"""Laplacian and effective-resistance positional encodings for OPF graphs.

The encodings are computed on the bus graph after OPF edge features have been
assembled.  Laplacian PE stores the smallest non-zero eigenpairs of the
unweighted bus Laplacian.  Effective-resistance PE stores, for each bus, the
diagonal-free ``[min, max, std, median, mean]`` summary of its resistance to
all other buses under the susceptance-weighted Laplacian.
"""

import copy
import fcntl
import hashlib
import json
import os
import re

import torch


_BUS_BRANCH_TYPES = (
    (("bus", "ac_line", "bus"), False),
    (("bus", "transformer", "bus"), True),
)
_SOURCE_ALIASES = {
    "laplacian": "laplacian",
    "lpe": "laplacian",
    "topological_laplacian": "laplacian",
    "effective_resistance": "effective_resistance",
    "resistance": "effective_resistance",
    "er": "effective_resistance",
}
_CACHE_VERSION = "opf-positional-encodings-v1"
_EXPECTED_STATISTICS = ["min", "max", "std", "median", "mean"]


def _canonical_sources(values):
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    result = []
    for value in values:
        key = str(value).lower()
        if key not in _SOURCE_ALIASES:
            raise ValueError(
                f"Unknown OPF positional encoding '{value}'. Expected "
                "'laplacian' or 'effective_resistance'."
            )
        canonical = _SOURCE_ALIASES[key]
        if canonical not in result:
            result.append(canonical)
    return result


def resolve_opf_positional_encoding_config(architecture_config):
    """Return a canonical, validated PE configuration."""

    architecture_config = architecture_config or {}
    raw = architecture_config.get("positional_encodings")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Architecture.positional_encodings must be a dictionary.")
    config = copy.deepcopy(raw)

    use = _canonical_sources(config.get("use", []))
    precompute = _canonical_sources(config.get("precompute", use))
    missing = set(use) - set(precompute)
    if missing:
        raise ValueError(
            "Every active positional encoding must also be precomputed; missing "
            f"from positional_encodings.precompute: {sorted(missing)}"
        )

    laplacian = copy.deepcopy(config.get("laplacian", {}))
    laplacian.setdefault("dim", 8)
    laplacian.setdefault("relative_tolerance", None)
    laplacian.setdefault("random_sign_flip", False)
    if "laplacian" in precompute and int(laplacian["dim"]) <= 0:
        raise ValueError("positional_encodings.laplacian.dim must be positive.")
    tolerance = laplacian.get("relative_tolerance")
    if tolerance is not None and float(tolerance) < 0.0:
        raise ValueError("laplacian.relative_tolerance must be non-negative.")

    resistance = copy.deepcopy(config.get("effective_resistance", {}))
    resistance.setdefault("statistics", _EXPECTED_STATISTICS)
    resistance.setdefault("std_correction", 0)
    resistance.setdefault("exclude_diagonal", True)
    if list(resistance["statistics"]) != _EXPECTED_STATISTICS:
        raise ValueError(
            "effective_resistance.statistics must be "
            f"{_EXPECTED_STATISTICS}, in that order."
        )
    if not bool(resistance["exclude_diagonal"]):
        raise ValueError("effective_resistance.exclude_diagonal must be true.")
    if int(resistance["std_correction"]) < 0:
        raise ValueError("effective_resistance.std_correction must be non-negative.")

    compute_device = str(config.get("compute_device", "auto")).lower()
    if compute_device != "auto":
        torch.device(compute_device)

    return {
        "precompute": precompute,
        "use": use,
        "cache_by_case": bool(config.get("cache_by_case", False)),
        "compute_device": compute_device,
        "laplacian": laplacian,
        "effective_resistance": resistance,
    }


def _resolve_compute_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PE compute_device requests CUDA, but CUDA is unavailable.")
    return device


def _edge_store(data, edge_type):
    return data[edge_type] if edge_type in data.edge_types else None


def _add_laplacian_branches(matrix, data, weighted):
    for edge_type, transformer in _BUS_BRANCH_TYPES:
        store = _edge_store(data, edge_type)
        if store is None or store.edge_index.numel() == 0:
            continue
        edge_index = store.edge_index.to(device=matrix.device, dtype=torch.long)
        src, dst = edge_index[0], edge_index[1]
        if weighted:
            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is None:
                raise ValueError(
                    f"Effective-resistance PE requires edge_attr for {edge_type}."
                )
            edge_attr = edge_attr.to(device=matrix.device, dtype=matrix.dtype)
            x_index = 3 if transformer else 5
            if edge_attr.size(1) <= x_index:
                raise ValueError(
                    f"edge_attr for {edge_type} lacks reactance column {x_index}."
                )
            reactance = edge_attr[:, x_index].abs()
            eps = torch.finfo(matrix.dtype).eps
            if torch.any(reactance <= eps):
                raise ValueError(f"Zero branch reactance encountered for {edge_type}.")
            weight = reactance.reciprocal()
            if transformer:
                if edge_attr.size(1) <= 7:
                    raise ValueError(
                        "Transformer edge_attr lacks the tap-ratio column (index 7)."
                    )
                tap = edge_attr[:, 7].abs()
                tap = torch.where(tap > eps, tap, torch.ones_like(tap))
                weight = weight / tap
        else:
            weight = torch.ones(src.numel(), dtype=matrix.dtype, device=matrix.device)

        matrix.index_put_((src, src), weight, accumulate=True)
        matrix.index_put_((dst, dst), weight, accumulate=True)
        matrix.index_put_((src, dst), -weight, accumulate=True)
        matrix.index_put_((dst, src), -weight, accumulate=True)


def build_topological_laplacian(data, dtype=torch.float64, device=None):
    num_bus = int(data["bus"].x.size(0))
    result = torch.zeros((num_bus, num_bus), dtype=dtype, device=device)
    _add_laplacian_branches(result, data, weighted=False)
    return result


def build_susceptance_laplacian(data, dtype=torch.float64, device=None):
    """Use 1/abs(x) for lines and 1/(abs(x) abs(tap)) for transformers."""

    num_bus = int(data["bus"].x.size(0))
    result = torch.zeros((num_bus, num_bus), dtype=dtype, device=device)
    _add_laplacian_branches(result, data, weighted=True)
    return result


def _eigenvalue_tolerance(values, matrix_size, relative_tolerance=None):
    if values.numel() == 0:
        return 0.0
    scale = values.abs().max()
    if relative_tolerance is not None:
        return float(scale) * float(relative_tolerance)
    return float(scale) * matrix_size * torch.finfo(values.dtype).eps


def compute_topological_laplacian_pe(
    data, k, relative_tolerance=None, compute_device="auto"
):
    """Compute the k smallest non-zero topological Laplacian eigenpairs."""

    k = int(k)
    num_bus = int(data["bus"].x.size(0))
    if k <= 0:
        raise ValueError("Laplacian PE dimension must be positive.")
    if num_bus == 0:
        return {
            "lap_eigvec": torch.empty((0, k), dtype=torch.float32),
            "lap_eigval": torch.zeros((1, k), dtype=torch.float32),
        }

    device = _resolve_compute_device(compute_device)
    laplacian = build_topological_laplacian(data, device=device)
    values, vectors = torch.linalg.eigh(laplacian)
    tolerance = _eigenvalue_tolerance(values, num_bus, relative_tolerance)
    selected = torch.nonzero(values > tolerance, as_tuple=False).flatten()[:k]

    eigvec = torch.zeros((num_bus, k), dtype=laplacian.dtype, device=device)
    eigval = torch.zeros(k, dtype=laplacian.dtype, device=device)
    count = int(selected.numel())
    if count:
        eigvec[:, :count] = vectors[:, selected]
        eigval[:count] = values[selected]
    return {
        "lap_eigvec": eigvec.float().cpu(),
        "lap_eigval": eigval.float().view(1, k).cpu(),
    }


def _num_bus_components(data):
    num_bus = int(data["bus"].x.size(0))
    parent = list(range(num_bus))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge_type, _ in _BUS_BRANCH_TYPES:
        store = _edge_store(data, edge_type)
        if store is None:
            continue
        for left, right in store.edge_index.detach().cpu().t().tolist():
            union(int(left), int(right))
    return len({find(node) for node in range(num_bus)}) if num_bus else 0


def compute_effective_resistance_matrix(data, compute_device="auto"):
    """Compute the dense resistance matrix from the weighted Laplacian."""

    num_bus = int(data["bus"].x.size(0))
    if num_bus == 0:
        return torch.empty((0, 0), dtype=torch.float64)
    components = _num_bus_components(data)
    if components != 1:
        raise ValueError(
            "Effective resistance is undefined between disconnected components; "
            f"the bus graph has {components} components."
        )

    device = _resolve_compute_device(compute_device)
    laplacian = build_susceptance_laplacian(data, device=device)
    centering = torch.full(
        (num_bus, num_bus),
        1.0 / float(num_bus),
        dtype=laplacian.dtype,
        device=device,
    )
    pseudoinverse = torch.linalg.inv(laplacian + centering) - centering
    diagonal = torch.diagonal(pseudoinverse)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * pseudoinverse
    resistance.clamp_min_(0.0)
    resistance.fill_diagonal_(0.0)
    return resistance


def _offdiagonal_rows(matrix):
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError(f"Expected a square matrix, got {tuple(matrix.shape)}.")
    num_bus = matrix.size(0)
    if num_bus < 2:
        raise ValueError("Diagonal-free resistance statistics require two buses.")
    mask = ~torch.eye(num_bus, dtype=torch.bool, device=matrix.device)
    return matrix.masked_select(mask).view(num_bus, num_bus - 1)


def _five_statistics(values, std_correction=0):
    correction = int(std_correction)
    if correction < 0 or correction >= values.size(1):
        raise ValueError(
            "std_correction must be smaller than the number of other buses "
            f"({values.size(1)}), got {correction}."
        )
    return torch.stack(
        (
            values.min(dim=1).values,
            values.max(dim=1).values,
            values.std(dim=1, correction=correction),
            torch.quantile(values, 0.5, dim=1),
            values.mean(dim=1),
        ),
        dim=1,
    )


def compute_effective_resistance_pe(data, std_correction=0, compute_device="auto"):
    num_bus = int(data["bus"].x.size(0))
    if num_bus == 0:
        return {"effective_resistance_pe": torch.empty((0, 5), dtype=torch.float32)}
    resistance = compute_effective_resistance_matrix(data, compute_device)
    stats = _five_statistics(
        _offdiagonal_rows(resistance), std_correction=std_correction
    )
    return {"effective_resistance_pe": stats.float().cpu()}


def _attach_artifact(data, source, artifact):
    bus = data["bus"]
    if source == "laplacian":
        bus.lap_eigvec = artifact["lap_eigvec"]
        bus.lap_eigval = artifact["lap_eigval"]
    elif source == "effective_resistance":
        bus.effective_resistance_pe = artifact["effective_resistance_pe"]
    else:
        raise ValueError(f"Unknown positional encoding '{source}'.")
    return data


def _update_hash_with_tensor(digest, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())


def _source_fingerprint(data, source, settings):
    digest = hashlib.sha256()
    digest.update(_CACHE_VERSION.encode())
    digest.update(source.encode())
    cache_settings = copy.deepcopy(settings)
    cache_settings.pop("random_sign_flip", None)
    digest.update(json.dumps(cache_settings, sort_keys=True).encode())
    digest.update(str(int(data["bus"].x.size(0))).encode())
    for edge_type, transformer in _BUS_BRANCH_TYPES:
        store = _edge_store(data, edge_type)
        digest.update(str(edge_type).encode())
        if store is None:
            digest.update(b"missing")
            continue
        _update_hash_with_tensor(digest, store.edge_index)
        if source == "effective_resistance":
            edge_attr = getattr(store, "edge_attr", None)
            if edge_attr is None:
                raise ValueError(f"Missing edge_attr for {edge_type}.")
            columns = [3, 7] if transformer else [5]
            if edge_attr.dim() != 2 or edge_attr.size(1) <= max(columns):
                raise ValueError(
                    f"edge_attr for {edge_type} lacks effective-resistance columns "
                    f"{columns}."
                )
            _update_hash_with_tensor(digest, edge_attr[:, columns])
    return digest.hexdigest()


def _safe_case_name(case_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name or "graph"))


class OPFPositionalEncodingPreprocessor:
    """Compute once per topology, cache, and attach configured PE tensors."""

    def __init__(self, architecture_config, cache_dir=None):
        self.config = resolve_opf_positional_encoding_config(architecture_config)
        self.cache_dir = cache_dir
        self._memory_cache = {}
        if cache_dir is not None and self.config["precompute"]:
            os.makedirs(cache_dir, exist_ok=True)

    @property
    def enabled(self):
        return bool(self.config["precompute"])

    def _compute(self, data, source):
        settings = self.config[source]
        if source == "laplacian":
            return compute_topological_laplacian_pe(
                data,
                settings["dim"],
                relative_tolerance=settings.get("relative_tolerance"),
                compute_device=self.config["compute_device"],
            )
        if source == "effective_resistance":
            return compute_effective_resistance_pe(
                data,
                std_correction=settings["std_correction"],
                compute_device=self.config["compute_device"],
            )
        raise ValueError(f"Unknown positional encoding '{source}'.")

    @staticmethod
    def _load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _load_or_compute(self, data, source, case_name):
        # Always hash the actual graph.  A case name is not a topology key when
        # topological perturbations are enabled, so trusting it can silently
        # attach an encoding computed for a different set of branches.
        fingerprint = _source_fingerprint(data, source, self.config[source])

        memory_key = (source, fingerprint)
        if memory_key in self._memory_cache:
            return self._memory_cache[memory_key]

        if self.cache_dir is None:
            artifact = self._compute(data, source)
        else:
            namespace = (
                _safe_case_name(case_name)
                if self.config["cache_by_case"]
                else "topology"
            )
            filename = f"{namespace}-{source}-{fingerprint[:20]}.pt"
            path = os.path.join(self.cache_dir, filename)
            with open(path + ".lock", "a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                if os.path.isfile(path):
                    artifact = self._load(path)
                else:
                    artifact = self._compute(data, source)
                    temporary = f"{path}.tmp-{os.getpid()}"
                    torch.save(artifact, temporary)
                    os.replace(temporary, path)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        self._memory_cache[memory_key] = artifact
        return artifact

    def __call__(self, data, case_name=None):
        for source in self.config["precompute"]:
            artifact = self._load_or_compute(data, source, case_name)
            _attach_artifact(data, source, artifact)
        return data
