#!/usr/bin/env python3
"""Measure how a trained OPF GNN's input-output Jacobian decays with distance.

For a model ``y = f_theta(u, G)``, this script evaluates exact Jacobian-vector
products at each stored operating point.  A direction changes one feature of
one source node, so the returned vector is one Jacobian column:

    J[:, j, k] = d f_theta(u, G) / d u[j, k].

The full Jacobian is never materialized.  Target responses are grouped by the
shortest-path distance between the source and target buses, producing

    A(d) = mean_{dist(i,j)=d} |d y_i / d u_j|.

By default every train/validation/test sample is visited, while up to 32 source
nodes are sampled per graph.  Use ``--max-sources-per-graph 0`` for every
source node in every graph.

Typical use from the HydraGNN repository root:

    python examples/opf/analyze_opf_jacobian.py \
        --modelname OPF_Solution \
        --checkpoint logs/OPF_Solution/OPF_Solution.pk \
        --data-root examples/opf/dataset

The differentiation is local to each stored data point; the OPF problem is not
re-solved and finite perturbations are not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch


SPLIT_LABELS = {
    "train": "trainset",
    "val": "valset",
    "test": "testset",
}

DEFAULT_COMPONENT_NAMES = {
    "load": {0: "pd", 1: "qd"},
    "bus": {0: "va", 1: "vm"},
    "generator": {0: "pg", 1: "qg"},
    "shunt": {0: "gs", 1: "bs"},
}


@dataclass
class RunningStats:
    """Streaming statistics for one (split, feature pair, distance) bin."""

    count: int = 0
    profiles: int = 0
    sum_signed: float = 0.0
    sum_abs: float = 0.0
    sum_square: float = 0.0
    max_abs: float = 0.0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        abs_values = np.abs(values)
        self.count += int(values.size)
        self.profiles += 1
        self.sum_signed += float(values.sum(dtype=np.float64))
        self.sum_abs += float(abs_values.sum(dtype=np.float64))
        self.sum_square += float(np.square(values).sum(dtype=np.float64))
        self.max_abs = max(self.max_abs, float(abs_values.max()))

    def row(self) -> dict[str, float | int]:
        if self.count:
            mean_signed = self.sum_signed / self.count
            mean_abs = self.sum_abs / self.count
            rms = math.sqrt(self.sum_square / self.count)
        else:
            mean_signed = mean_abs = rms = float("nan")
        return {
            "count": self.count,
            "profiles": self.profiles,
            "mean_signed": mean_signed,
            "mean_abs": mean_abs,
            "rms": rms,
            "max_abs": self.max_abs,
            "mean_shell_l1": (
                self.sum_abs / self.profiles if self.profiles else float("nan")
            ),
            "mean_shell_l2_energy": (
                self.sum_square / self.profiles
                if self.profiles
                else float("nan")
            ),
            "sum_signed": self.sum_signed,
            "sum_abs": self.sum_abs,
            "sum_square": self.sum_square,
        }


class RadialAccumulator:
    """Aggregate Jacobian entries without retaining individual responses."""

    def __init__(self) -> None:
        self._stats: dict[tuple[str, int, int, int], RunningStats] = defaultdict(
            RunningStats
        )

    def add_profile(
        self,
        scope: str,
        source_feature: int,
        output_feature: int,
        distances: np.ndarray,
        values: np.ndarray,
    ) -> None:
        distances = np.asarray(distances, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        valid = (distances >= 0) & np.isfinite(values)
        if not np.any(valid):
            return
        valid_distances = distances[valid]
        valid_values = values[valid]
        for distance in np.unique(valid_distances):
            shell = valid_values[valid_distances == distance]
            self._stats[
                (scope, source_feature, output_feature, int(distance))
            ].add(shell)

    def rows(
        self,
        source_node_type: str,
        target_node_type: str,
        output_head: int,
        source_names: dict[int, str],
        output_names: dict[int, str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(self._stats):
            scope, source_feature, output_feature, distance = key
            row: dict[str, Any] = {
                "scope": scope,
                "source_node_type": source_node_type,
                "source_feature_index": source_feature,
                "source_feature_name": source_names[source_feature],
                "target_node_type": target_node_type,
                "output_head": output_head,
                "output_feature_index": output_feature,
                "output_feature_name": output_names[output_feature],
                "distance": distance,
            }
            row.update(self._stats[key].row())
            rows.append(row)
        return rows


@dataclass
class TopologyInfo:
    fingerprint: str
    adjacency: tuple[tuple[int, ...], ...]
    source_to_bus: np.ndarray
    target_to_bus: np.ndarray
    source_permutation: np.ndarray


class DistanceCache:
    """Bounded LRU cache of source-bus to target-node graph distances."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def get(self, topology: TopologyInfo, source_bus: int) -> np.ndarray:
        key = (topology.fingerprint, int(source_bus))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        distances = _bfs_target_distances(
            topology.adjacency,
            int(source_bus),
            topology.target_to_bus,
        )
        if self.max_entries:
            self._cache[key] = distances
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return distances


class JVPComputer:
    """Try forward-mode AD first and fall back to PyTorch's functional JVP."""

    def __init__(self, backend: str) -> None:
        self.requested_backend = backend
        self.active_backend: str | None = None if backend == "auto" else backend
        self._fallback_reported = False

    @staticmethod
    def _func_jvp(function, primal: torch.Tensor, tangent: torch.Tensor):
        _, response = torch.func.jvp(function, (primal,), (tangent,))
        return response

    @staticmethod
    def _autograd_jvp(function, primal: torch.Tensor, tangent: torch.Tensor):
        _, response = torch.autograd.functional.jvp(
            function,
            primal,
            tangent,
            create_graph=False,
            strict=False,
        )
        return response

    def __call__(self, function, primal: torch.Tensor, tangent: torch.Tensor):
        if self.active_backend == "func":
            return self._func_jvp(function, primal, tangent)
        if self.active_backend == "autograd":
            return self._autograd_jvp(function, primal, tangent)

        try:
            response = self._func_jvp(function, primal, tangent)
            self.active_backend = "func"
            return response
        except (RuntimeError, NotImplementedError) as exc:
            self.active_backend = "autograd"
            if not self._fallback_reported:
                warnings.warn(
                    "torch.func.jvp is unsupported by an operation in this model; "
                    "falling back to torch.autograd.functional.jvp. Original "
                    f"error: {type(exc).__name__}: {exc}",
                    stacklevel=2,
                )
                self._fallback_reported = True
            return self._autograd_jvp(function, primal, tangent)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modelname",
        "--model-name",
        dest="modelname",
        default="OPF_Solution",
        help="Run name used to discover logs and the serialized dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset basename when it differs from --modelname.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit .pk checkpoint; otherwise search common log locations.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit JSON config; otherwise use config.json by the checkpoint.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Additional directory containing <modelname>/<modelname>.pk.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset parent, or the .pickle/.h5 dataset directory itself.",
    )
    parser.add_argument(
        "--format",
        choices=("pickle", "hdf5"),
        default="pickle",
        help="HydraGNN serialized dataset format.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_LABELS),
        default=list(SPLIT_LABELS),
        help="Dataset splits to analyze.",
    )
    parser.add_argument(
        "--source-node-type",
        default="load",
        help="Node type whose input features define perturbation directions.",
    )
    parser.add_argument(
        "--source-feature-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Columns of source-node x to differentiate with respect to.",
    )
    parser.add_argument(
        "--source-feature-names",
        nargs="+",
        default=None,
        help="Labels corresponding to --source-feature-indices.",
    )
    parser.add_argument(
        "--target-node-type",
        default=None,
        help="Output node type; defaults to Architecture.node_target_type.",
    )
    parser.add_argument(
        "--output-head",
        type=int,
        default=0,
        help="Index in the list returned by the HydraGNN model.",
    )
    parser.add_argument(
        "--output-feature-indices",
        nargs="+",
        type=int,
        default=None,
        help="Output columns to aggregate; default is every output column.",
    )
    parser.add_argument(
        "--output-feature-names",
        nargs="+",
        default=None,
        help="Labels corresponding to --output-feature-indices.",
    )
    parser.add_argument(
        "--bus-edge-relations",
        nargs="+",
        default=["ac_line", "transformer"],
        help="Bus-to-bus relation names used for shortest-path distance.",
    )
    parser.add_argument(
        "--max-sources-per-graph",
        type=int,
        default=32,
        help="Maximum source nodes per sample; 0 means all source nodes.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Maximum samples in each selected split; 0 means the whole split.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Analyze every Nth sample.",
    )
    parser.add_argument(
        "--source-unit-scales",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Physical units per stored input unit for selected source features. "
            "The reported Jacobian is multiplied by output_scale/source_scale."
        ),
    )
    parser.add_argument(
        "--output-unit-scales",
        nargs="+",
        type=float,
        default=None,
        help="Physical units per stored output unit for selected outputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete CUDA device such as cuda:1.",
    )
    parser.add_argument(
        "--jvp-backend",
        choices=("auto", "func", "autograd"),
        default="auto",
        help="Automatic-differentiation implementation.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--distance-cache-sources",
        type=int,
        default=256,
        help="Maximum cached BFS result vectors across topologies.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Rewrite partial CSV/JSON outputs every N processed samples; 0 disables.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N processed samples; 0 disables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory; defaults to <checkpoint-dir>/jacobian_analysis.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Record and skip malformed samples instead of stopping.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create the radial sensitivity PNG.",
    )
    return parser


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        # Match HydraGNN's OMPI/Slurm/PALS local-rank selection.  ROCm-enabled
        # PyTorch intentionally exposes AMD GPUs through the "cuda" device API.
        from hydragnn.utils.distributed import get_device

        device = get_device(use_gpu=True)
    else:
        device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({spec}), but CUDA is unavailable.")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device.index)
    return device


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.expanduser().resolve())
        if normalized not in seen:
            seen.add(normalized)
            result.append(Path(normalized))
    return result


def _discover_artifacts(
    args: argparse.Namespace,
    script_dir: Path,
) -> tuple[Path, Path]:
    repo_root = script_dir.parents[1]
    if args.checkpoint is not None:
        checkpoint_candidates = [args.checkpoint]
    else:
        log_roots = [
            Path.cwd() / "logs",
            repo_root / "logs",
            script_dir / "logs",
        ]
        if args.log_root is not None:
            log_roots.insert(0, args.log_root)
        checkpoint_candidates = [
            root / args.modelname / f"{args.modelname}.pk" for root in log_roots
        ]
        checkpoint_candidates.append(
            script_dir
            / "pretrained_models"
            / args.modelname
            / f"{args.modelname}.pk"
        )

    checkpoint_candidates = _unique_paths(checkpoint_candidates)
    checkpoint = next((p for p in checkpoint_candidates if p.is_file()), None)
    if checkpoint is None:
        searched = "\n  ".join(str(p) for p in checkpoint_candidates)
        raise FileNotFoundError(
            "Could not find a model checkpoint. Pass --checkpoint explicitly. "
            f"Searched:\n  {searched}"
        )

    if args.config is not None:
        config_candidates = [args.config]
    else:
        config_candidates = [
            checkpoint.parent / "config.json",
            script_dir / f"{args.modelname}.json",
            script_dir / "opf_solution_heterogeneous.json",
        ]
    config_candidates = _unique_paths(config_candidates)
    config_path = next((p for p in config_candidates if p.is_file()), None)
    if config_path is None:
        searched = "\n  ".join(str(p) for p in config_candidates)
        raise FileNotFoundError(
            "Could not find the architecture config needed to reconstruct the "
            f"model. Pass --config explicitly. Searched:\n  {searched}"
        )
    return checkpoint, config_path


def _dataset_directory(
    args: argparse.Namespace,
    script_dir: Path,
) -> Path:
    dataset_name = args.dataset_name or args.modelname
    suffix = ".pickle" if args.format == "pickle" else ".h5"
    root = (args.data_root or (script_dir / "dataset")).expanduser().resolve()
    if root.name.endswith(suffix):
        basedir = root
    else:
        basedir = root / f"{dataset_name}{suffix}"
    if not basedir.is_dir():
        raise FileNotFoundError(
            f"Missing {args.format} dataset directory: {basedir}. "
            "Pass either its parent or the directory itself via --data-root."
        )
    return basedir


def _edge_dim_from_config(config: dict[str, Any]) -> int | dict[str, int]:
    architecture = config["NeuralNetwork"]["Architecture"]
    raw = architecture.get("edge_dim")
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}
    if raw is None:
        raise RuntimeError("NeuralNetwork.Architecture.edge_dim is missing.")
    return int(raw)


def _load_datasets(
    args: argparse.Namespace,
    basedir: Path,
    target_node_type: str,
    edge_dim: int | dict[str, int],
) -> dict[str, Any]:
    from opf_solution_utils import NodeTargetDatasetAdapter

    if args.format == "pickle":
        from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
    else:
        from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset

    datasets: dict[str, Any] = {}
    for split in args.splits:
        label = SPLIT_LABELS[split]
        if args.format == "pickle":
            base = SimplePickleDataset(
                basedir=str(basedir),
                label=label,
                var_config=None,
            )
        else:
            base = HDF5Dataset(str(basedir), label)
        datasets[split] = NodeTargetDatasetAdapter(
            base,
            target_node_type,
            edge_dim=edge_dim,
        )
    return datasets


def _prepare_config(
    config: dict[str, Any],
    reference_dataset: Any,
    target_node_type: str,
) -> dict[str, Any]:
    """Apply HydraGNN's normal config completion without scanning the dataset."""

    from hydragnn.utils.input_config_parsing.config_utils import update_config
    from opf_solution_utils import compute_pna_deg_for_hetero_dataset

    architecture = config.setdefault("NeuralNetwork", {}).setdefault(
        "Architecture", {}
    )
    architecture["node_target_type"] = target_node_type

    holder = SimpleNamespace(dataset=reference_dataset)
    old_graph_size = os.environ.get("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE")
    os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = "0"
    try:
        config = update_config(config, holder, holder, holder)
    finally:
        if old_graph_size is None:
            os.environ.pop("HYDRAGNN_USE_VARIABLE_GRAPH_SIZE", None)
        else:
            os.environ["HYDRAGNN_USE_VARIABLE_GRAPH_SIZE"] = old_graph_size

    architecture = config["NeuralNetwork"]["Architecture"]
    architecture["output_dim"] = [int(value) for value in architecture["output_dim"]]
    if architecture.get("mpnn_type") == "HeteroPNA" and not architecture.get(
        "pna_deg"
    ):
        print("Computing HeteroPNA degree histogram from the reference split.")
        degree_histogram = compute_pna_deg_for_hetero_dataset(
            reference_dataset,
            verbosity=2,
        )
        architecture["pna_deg"] = degree_histogram
        architecture["max_neighbours"] = max(0, len(degree_histogram) - 1)
    return config


def _load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint at {path} is not a dictionary.")
    raw_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError(f"Checkpoint at {path} has no model state dictionary.")

    state = dict(raw_state)
    stripped_prefix = True
    while state and stripped_prefix:
        stripped_prefix = False
        for prefix in ("module.", "_orig_mod."):
            if all(key.startswith(prefix) for key in state):
                state = {key[len(prefix) :]: value for key, value in state.items()}
                stripped_prefix = True
                break
    return state, checkpoint


def _upgrade_legacy_heat_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map pre-adapter HeteroHEAT convolution keys to the current module path."""

    heat_parameter_roots = {
        "hetero_lin",
        "edge_type_emb",
        "edge_attr_emb",
        "att",
        "lin",
    }
    upgraded: dict[str, torch.Tensor] = {}
    remapped = 0
    for key, value in state.items():
        parts = key.split(".")
        try:
            conv_position = parts.index("graph_convs")
        except ValueError:
            upgraded[key] = value
            continue
        parameter_position = conv_position + 2
        if (
            parameter_position < len(parts)
            and parts[parameter_position] in heat_parameter_roots
        ):
            parts.insert(parameter_position, "heat_conv")
            key = ".".join(parts)
            remapped += 1
        upgraded[key] = value
    if remapped:
        warnings.warn(
            f"Remapped {remapped} legacy HeteroHEAT checkpoint keys to the "
            "current adapter layout.",
            stacklevel=2,
        )
    return upgraded


def _model_output(model, data, output_head: int) -> torch.Tensor:
    outputs = model(data)
    if isinstance(outputs, torch.Tensor):
        if output_head != 0:
            raise IndexError(
                f"Model returned one tensor, so --output-head={output_head} is invalid."
            )
        return outputs
    if output_head < 0 or output_head >= len(outputs):
        raise IndexError(
            f"--output-head={output_head} is outside model output range "
            f"[0, {len(outputs) - 1}]."
        )
    return outputs[output_head]


def _build_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    sample: Any,
    device: torch.device,
    output_head: int,
) -> tuple[torch.nn.Module, dict]:
    import hydragnn
    from opf_solution_utils import OPFDomainLoss, OPFEnhancedModelWrapper

    state, checkpoint = _load_checkpoint_state(checkpoint_path)
    architecture = config["NeuralNetwork"]["Architecture"]
    if architecture.get("mpnn_type") == "HeteroHEAT":
        state = _upgrade_legacy_heat_state(state)
    node_input_dims = architecture.get("node_input_dims")
    if not node_input_dims:
        node_input_dims = {
            str(node_type): int(sample[node_type].x.shape[-1])
            for node_type in sample.node_types
            if getattr(sample[node_type], "x", None) is not None
        }
        architecture["node_input_dims"] = node_input_dims

    base_model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config.get("Verbosity", {}).get("level", 1),
        use_gpu=device.type == "cuda",
        metadata=sample.metadata(),
        node_input_dims=node_input_dims,
    )
    # Heterogeneous models cache HydraGNN's initially selected device and use it
    # when checking lazily created modules.  Keep that cache synchronized with
    # an explicit analysis-device override.
    if hasattr(base_model, "device"):
        base_model.device = device

    checkpoint_has_opf_wrapper = bool(state) and all(
        key.startswith("model.") for key in state
    )
    if checkpoint_has_opf_wrapper:
        domain_config = config["NeuralNetwork"]["Training"].get("DomainLoss", {})
        target = architecture["node_target_type"]
        model: torch.nn.Module = OPFEnhancedModelWrapper(
            base_model,
            OPFDomainLoss(domain_config, node_target_type=target),
        )
    else:
        model = base_model

    model = model.to(device)
    model.eval()

    # Some heterogeneous modules create edge projectors lazily.  Materialize
    # them before state loading so checkpoint keys have matching destinations.
    dry_sample = sample.clone().to(device)
    with torch.no_grad():
        _model_output(model, dry_sample, output_head)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint does not match the model reconstructed from the "
            f"config. Checkpoint: {checkpoint_path}. Config architecture: "
            f"{architecture.get('mpnn_type')}. Original error:\n{exc}"
        ) from exc

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _num_nodes(data: Any, node_type: str) -> int:
    if node_type not in data.node_types:
        raise RuntimeError(
            f"Node type '{node_type}' is absent; available: {list(data.node_types)}."
        )
    store = data[node_type]
    if getattr(store, "num_nodes", None) is not None:
        return int(store.num_nodes)
    for field in ("x", "y"):
        value = getattr(store, field, None)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    raise RuntimeError(f"Cannot determine number of '{node_type}' nodes.")


def _entity_to_bus(data: Any, node_type: str) -> np.ndarray:
    count = _num_nodes(data, node_type)
    if node_type == "bus":
        return np.arange(count, dtype=np.int64)

    mapping = np.full(count, -1, dtype=np.int64)
    direct = [
        edge_type
        for edge_type in data.edge_types
        if edge_type[0] == node_type and edge_type[2] == "bus"
    ]
    reverse = [
        edge_type
        for edge_type in data.edge_types
        if edge_type[0] == "bus" and edge_type[2] == node_type
    ]
    candidates = [(edge_type, False) for edge_type in direct]
    if not candidates:
        candidates = [(edge_type, True) for edge_type in reverse]
    if not candidates:
        raise RuntimeError(
            f"No {node_type}->bus or bus->{node_type} relation is available "
            "to anchor nodes in the physical bus graph."
        )

    for edge_type, is_reverse in candidates:
        edge_index = data[edge_type].edge_index.detach().cpu().numpy()
        entities = edge_index[1] if is_reverse else edge_index[0]
        buses = edge_index[0] if is_reverse else edge_index[1]
        for entity, bus in zip(entities.tolist(), buses.tolist()):
            old_bus = mapping[entity]
            if old_bus >= 0 and old_bus != bus:
                raise RuntimeError(
                    f"{node_type} node {entity} maps to multiple buses "
                    f"({old_bus} and {bus})."
                )
            mapping[entity] = bus
    return mapping


def _canonical_bus_edges(
    data: Any,
    relation_names: set[str],
) -> np.ndarray:
    edge_arrays: list[np.ndarray] = []
    for edge_type in data.edge_types:
        source_type, relation, target_type = edge_type
        if (
            source_type == "bus"
            and target_type == "bus"
            and relation in relation_names
        ):
            edge_index = data[edge_type].edge_index.detach().cpu().numpy()
            if edge_index.size:
                pairs = np.sort(edge_index.astype(np.int64, copy=False), axis=0).T
                edge_arrays.append(pairs)
    if not edge_arrays:
        raise RuntimeError(
            "No bus-to-bus edges matched --bus-edge-relations="
            f"{sorted(relation_names)}."
        )
    pairs = np.concatenate(edge_arrays, axis=0)
    pairs = np.unique(pairs, axis=0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return np.ascontiguousarray(pairs[order], dtype=np.int64)


def _topology_info(
    data: Any,
    source_node_type: str,
    target_node_type: str,
    relation_names: set[str],
    seed: int,
) -> TopologyInfo:
    num_buses = _num_nodes(data, "bus")
    source_to_bus = _entity_to_bus(data, source_node_type)
    target_to_bus = _entity_to_bus(data, target_node_type)
    pairs = _canonical_bus_edges(data, relation_names)

    digest = hashlib.sha256()
    digest.update(np.asarray([num_buses], dtype=np.int64).tobytes())
    digest.update(pairs.tobytes())
    digest.update(source_to_bus.tobytes())
    digest.update(target_to_bus.tobytes())
    fingerprint = digest.hexdigest()

    adjacency_lists: list[list[int]] = [[] for _ in range(num_buses)]
    for first, second in pairs.tolist():
        if not (0 <= first < num_buses and 0 <= second < num_buses):
            raise RuntimeError(
                f"Bus edge ({first}, {second}) exceeds bus count {num_buses}."
            )
        if first == second:
            continue
        adjacency_lists[first].append(second)
        adjacency_lists[second].append(first)
    adjacency = tuple(tuple(sorted(set(items))) for items in adjacency_lists)

    valid_sources = np.flatnonzero(source_to_bus >= 0).astype(np.int64)
    topology_seed = seed ^ int(fingerprint[:16], 16)
    rng = np.random.default_rng(topology_seed)
    source_permutation = rng.permutation(valid_sources)
    return TopologyInfo(
        fingerprint=fingerprint,
        adjacency=adjacency,
        source_to_bus=source_to_bus,
        target_to_bus=target_to_bus,
        source_permutation=source_permutation,
    )


def _bfs_target_distances(
    adjacency: tuple[tuple[int, ...], ...],
    source_bus: int,
    target_to_bus: np.ndarray,
) -> np.ndarray:
    if source_bus < 0 or source_bus >= len(adjacency):
        return np.full(target_to_bus.shape, -1, dtype=np.int32)
    bus_distances = np.full(len(adjacency), -1, dtype=np.int32)
    bus_distances[source_bus] = 0
    queue: deque[int] = deque([source_bus])
    while queue:
        node = queue.popleft()
        next_distance = bus_distances[node] + 1
        for neighbor in adjacency[node]:
            if bus_distances[neighbor] < 0:
                bus_distances[neighbor] = next_distance
                queue.append(neighbor)

    result = np.full(target_to_bus.shape, -1, dtype=np.int32)
    valid = (target_to_bus >= 0) & (target_to_bus < len(adjacency))
    result[valid] = bus_distances[target_to_bus[valid]]
    return result


def _select_source_nodes(
    topology: TopologyInfo,
    maximum: int,
    graph_ordinal: int,
) -> np.ndarray:
    permutation = topology.source_permutation
    count = len(permutation)
    if maximum <= 0 or maximum >= count:
        return permutation
    start = (graph_ordinal * maximum) % count
    indices = (start + np.arange(maximum, dtype=np.int64)) % count
    return permutation[indices]


def _component_names(
    explicit_names: Sequence[str] | None,
    indices: Sequence[int],
    node_type: str,
    prefix: str,
) -> dict[int, str]:
    if explicit_names is not None and len(explicit_names) != len(indices):
        raise ValueError(
            f"Expected {len(indices)} {prefix} names, got {len(explicit_names)}."
        )
    defaults = DEFAULT_COMPONENT_NAMES.get(node_type, {})
    result: dict[int, str] = {}
    for position, index in enumerate(indices):
        result[index] = (
            explicit_names[position]
            if explicit_names is not None
            else defaults.get(index, f"{prefix}{index}")
        )
    return result


def _scales(
    values: Sequence[float] | None,
    indices: Sequence[int],
    description: str,
) -> dict[int, float]:
    if values is None:
        return {index: 1.0 for index in indices}
    if len(values) != len(indices):
        raise ValueError(
            f"Expected {len(indices)} {description} values, got {len(values)}."
        )
    if any(value == 0.0 for value in values):
        raise ValueError(f"{description} values must be nonzero.")
    return {index: float(value) for index, value in zip(indices, values)}


def _reshape_node_output(output: torch.Tensor, target_count: int) -> torch.Tensor:
    if output.ndim == 1:
        if output.numel() % target_count:
            raise RuntimeError(
                f"Flat output length {output.numel()} is not divisible by "
                f"target node count {target_count}."
            )
        output = output.reshape(target_count, -1)
    elif output.ndim != 2:
        raise RuntimeError(
            f"Expected a 1D or 2D node output tensor, got {tuple(output.shape)}."
        )
    if output.shape[0] != target_count:
        raise RuntimeError(
            f"Model produced {output.shape[0]} target rows, but "
            f"'{target_count}' target nodes were mapped to buses."
        )
    return output


def _atomic_write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "radial_jacobian.csv"
    csv_tmp = output_dir / ".radial_jacobian.csv.tmp"
    if rows:
        with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(csv_tmp, csv_path)

    json_path = output_dir / "radial_jacobian.json"
    json_tmp = output_dir / ".radial_jacobian.json.tmp"
    with json_tmp.open("w", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "rows": rows}, stream, indent=2)
    os.replace(json_tmp, json_path)


def _plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is unavailable; skipping the summary plot.")
        return

    all_rows = [row for row in rows if row["scope"] == "all"]
    source_features = sorted(
        {int(row["source_feature_index"]) for row in all_rows}
    )
    if not source_features:
        return

    fig, axes = plt.subplots(
        1,
        len(source_features),
        figsize=(6 * len(source_features), 4.5),
        squeeze=False,
    )
    for axis, source_feature in zip(axes[0], source_features):
        source_rows = [
            row
            for row in all_rows
            if int(row["source_feature_index"]) == source_feature
        ]
        output_features = sorted(
            {int(row["output_feature_index"]) for row in source_rows}
        )
        for output_feature in output_features:
            curve = sorted(
                (
                    row
                    for row in source_rows
                    if int(row["output_feature_index"]) == output_feature
                ),
                key=lambda row: int(row["distance"]),
            )
            axis.plot(
                [int(row["distance"]) for row in curve],
                [float(row["mean_abs"]) for row in curve],
                marker="o",
                label=str(curve[0]["output_feature_name"]),
            )
        # Exact zeros are meaningful for a finite-receptive-field GNN, so use a
        # symmetric-log scale rather than dropping them on a conventional log axis.
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.set_xlabel("Physical bus-graph distance")
        axis.set_ylabel("Mean absolute Jacobian entry")
        axis.set_title(
            f"source: {source_rows[0]['source_feature_name']}"
            if source_rows
            else f"source feature {source_feature}"
        )
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "radial_jacobian_mean_abs.png", dpi=200)
    plt.close(fig)


def _analysis_rows(
    accumulator: RadialAccumulator,
    args: argparse.Namespace,
    target_node_type: str,
    source_names: dict[int, str],
    output_names: dict[int, str],
) -> list[dict[str, Any]]:
    return accumulator.rows(
        source_node_type=args.source_node_type,
        target_node_type=target_node_type,
        output_head=args.output_head,
        source_names=source_names,
        output_names=output_names,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be at least 1.")
    if args.max_sources_per_graph < 0:
        raise ValueError("--max-sources-per-graph cannot be negative.")
    if args.max_samples_per_split < 0:
        raise ValueError("--max-samples-per-split cannot be negative.")

    script_dir = Path(__file__).resolve().parent
    # Make sibling OPF utilities importable when this is launched elsewhere.
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    checkpoint_path, config_path = _discover_artifacts(args, script_dir)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    architecture = config["NeuralNetwork"]["Architecture"]
    target_node_type = (
        args.target_node_type
        or architecture.get("node_target_type")
        or "bus"
    )
    edge_dim = _edge_dim_from_config(config)
    dataset_dir = _dataset_directory(args, script_dir)
    datasets = _load_datasets(
        args,
        dataset_dir,
        target_node_type,
        edge_dim,
    )
    reference_dataset = datasets[args.splits[0]]
    if len(reference_dataset) == 0:
        raise RuntimeError(f"Selected split '{args.splits[0]}' is empty.")
    reference_sample = reference_dataset[0]

    if args.source_node_type not in reference_sample.node_types:
        raise RuntimeError(
            f"Source node type '{args.source_node_type}' is absent; available "
            f"types: {list(reference_sample.node_types)}."
        )
    source_width = int(reference_sample[args.source_node_type].x.shape[-1])
    for index in args.source_feature_indices:
        if index < 0 or index >= source_width:
            raise IndexError(
                f"Source feature index {index} is outside [0, {source_width - 1}]."
            )

    config = _prepare_config(config, reference_dataset, target_node_type)
    training_config = config["NeuralNetwork"]["Training"]
    checkpoint_precision = training_config.get("precision", "fp32")
    oversmoothing_config = training_config.setdefault("Oversmoothing", {})
    checkpoint_oversmoothing_enabled = bool(
        oversmoothing_config.get("enabled", False)
    )
    # Oversmoothing collection is training instrumentation, not part of the
    # predictor.  Leaving it enabled would recompute diagnostics for every JVP.
    oversmoothing_config["enabled"] = False
    # The heterogeneous model path explicitly casts node inputs to float32.
    # Reconstructing in fp32 also avoids noisy bf16 Jacobians and dtype
    # mismatches when differentiation is run outside the training autocast loop.
    training_config["precision"] = "fp32"
    device = _resolve_device(args.device)
    model, checkpoint = _build_model(
        config,
        checkpoint_path,
        reference_sample,
        device,
        args.output_head,
    )

    # Determine and validate output columns with one loaded-model evaluation.
    probe = reference_sample.clone().to(device)
    with torch.no_grad():
        probe_output = _model_output(model, probe, args.output_head)
    probe_target_count = len(_entity_to_bus(probe, target_node_type))
    probe_output = _reshape_node_output(probe_output, probe_target_count)
    output_width = int(probe_output.shape[-1])
    output_features = (
        list(range(output_width))
        if args.output_feature_indices is None
        else args.output_feature_indices
    )
    for index in output_features:
        if index < 0 or index >= output_width:
            raise IndexError(
                f"Output feature index {index} is outside [0, {output_width - 1}]."
            )

    source_names = _component_names(
        args.source_feature_names,
        args.source_feature_indices,
        args.source_node_type,
        "x",
    )
    output_names = _component_names(
        args.output_feature_names,
        output_features,
        target_node_type,
        "y",
    )
    source_scales = _scales(
        args.source_unit_scales,
        args.source_feature_indices,
        "source-unit-scale",
    )
    output_scales = _scales(
        args.output_unit_scales,
        output_features,
        "output-unit-scale",
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "jacobian_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_path}")
    print(f"Dataset:    {dataset_dir}")
    print(f"Device:     {device}")
    print(
        f"Precision:  fp32 analysis (checkpoint config: {checkpoint_precision})"
    )
    if checkpoint_oversmoothing_enabled:
        print("Diagnostics: checkpoint oversmoothing collection disabled for JVPs")
    print(f"Results:    {output_dir}")
    if args.max_sources_per_graph:
        print(
            "Source sampling: up to "
            f"{args.max_sources_per_graph} nodes per sample, rotated across samples."
        )
    else:
        print("Source sampling: every valid source node in every sample.")

    accumulator = RadialAccumulator()
    jvp = JVPComputer(args.jvp_backend)
    distance_cache = DistanceCache(args.distance_cache_sources)
    topology_cache: dict[str, TopologyInfo] = {}
    relation_names = set(args.bus_edge_relations)
    counts: dict[str, int] = {
        "processed_samples": 0,
        "processed_directions": 0,
        "skipped_samples": 0,
        "unmapped_source_nodes": 0,
        "unreachable_target_entries": 0,
    }
    errors: list[dict[str, Any]] = []
    start_time = time.monotonic()
    graph_ordinal = 0

    def metadata(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "dataset": str(dataset_dir),
            "modelname": args.modelname,
            "splits": args.splits,
            "source_node_type": args.source_node_type,
            "source_feature_indices": args.source_feature_indices,
            "target_node_type": target_node_type,
            "output_head": args.output_head,
            "output_feature_indices": output_features,
            "bus_edge_relations": sorted(relation_names),
            "max_sources_per_graph": args.max_sources_per_graph,
            "max_samples_per_split": args.max_samples_per_split,
            "sample_stride": args.sample_stride,
            "jvp_backend_requested": args.jvp_backend,
            "jvp_backend_active": jvp.active_backend,
            "device": str(device),
            "checkpoint_config_precision": checkpoint_precision,
            "analysis_precision": "fp32",
            "checkpoint_oversmoothing_enabled": checkpoint_oversmoothing_enabled,
            "analysis_oversmoothing_enabled": False,
            "source_unit_scales": source_scales,
            "output_unit_scales": output_scales,
            "definition": (
                "A(d) pools abs(d model_output[i] / d source_x[j]) over "
                "source-target pairs whose anchored buses have shortest-path "
                "distance d."
            ),
            "counts": counts.copy(),
            "unique_topologies": len(topology_cache),
            "elapsed_seconds": time.monotonic() - start_time,
            "errors": errors,
            "checkpoint_fields": sorted(
                key for key in checkpoint if key != "model_state_dict"
            ),
        }

    for split in args.splits:
        dataset = datasets[split]
        candidate_indices = range(0, len(dataset), args.sample_stride)
        if args.max_samples_per_split:
            candidate_indices = list(candidate_indices)[
                : args.max_samples_per_split
            ]
        print(f"Analyzing {split}: {len(candidate_indices)} samples")

        for sample_index in candidate_indices:
            try:
                host_data = dataset[sample_index].clone()
                computed_topology = _topology_info(
                    host_data,
                    args.source_node_type,
                    target_node_type,
                    relation_names,
                    args.seed,
                )
                topology = topology_cache.setdefault(
                    computed_topology.fingerprint,
                    computed_topology,
                )
                counts["unmapped_source_nodes"] += int(
                    np.count_nonzero(topology.source_to_bus < 0)
                )
                selected_sources = _select_source_nodes(
                    topology,
                    args.max_sources_per_graph,
                    graph_ordinal,
                )
                if selected_sources.size == 0:
                    raise RuntimeError(
                        f"No '{args.source_node_type}' nodes map to the bus graph."
                    )

                data = host_data.to(device)
                source_x = data[args.source_node_type].x.detach()
                if not source_x.is_floating_point():
                    raise TypeError(
                        f"{args.source_node_type}.x must be floating point for AD."
                    )
                working_data = data.clone()

                def forward_from_source_x(input_x: torch.Tensor) -> torch.Tensor:
                    working_data[args.source_node_type].x = input_x
                    raw_output = _model_output(
                        model,
                        working_data,
                        args.output_head,
                    )
                    return _reshape_node_output(
                        raw_output,
                        len(topology.target_to_bus),
                    )

                for source_node in selected_sources.tolist():
                    source_bus = int(topology.source_to_bus[source_node])
                    distances = distance_cache.get(topology, source_bus)
                    counts["unreachable_target_entries"] += int(
                        np.count_nonzero(distances < 0)
                    )
                    for source_feature in args.source_feature_indices:
                        tangent = torch.zeros_like(source_x)
                        tangent[source_node, source_feature] = 1.0
                        with torch.enable_grad():
                            response = jvp(
                                forward_from_source_x,
                                source_x,
                                tangent,
                            )
                        response = response.detach().to(
                            device="cpu",
                            dtype=torch.float64,
                        )
                        for output_feature in output_features:
                            physical_factor = (
                                output_scales[output_feature]
                                / source_scales[source_feature]
                            )
                            values = (
                                response[:, output_feature].numpy()
                                * physical_factor
                            )
                            accumulator.add_profile(
                                split,
                                source_feature,
                                output_feature,
                                distances,
                                values,
                            )
                            accumulator.add_profile(
                                "all",
                                source_feature,
                                output_feature,
                                distances,
                                values,
                            )
                        counts["processed_directions"] += 1

                counts["processed_samples"] += 1
                graph_ordinal += 1

            except Exception as exc:
                if not args.skip_errors:
                    raise
                counts["skipped_samples"] += 1
                errors.append(
                    {
                        "split": split,
                        "sample_index": int(sample_index),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"Skipping {split}[{sample_index}]: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            if args.log_every and (
                counts["processed_samples"] + counts["skipped_samples"]
            ) % args.log_every == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"Progress: {counts['processed_samples']} samples, "
                    f"{counts['processed_directions']} JVPs, "
                    f"{len(topology_cache)} topologies, {elapsed:.1f} s",
                    flush=True,
                )

            if args.save_every and (
                counts["processed_samples"] + counts["skipped_samples"]
            ) % args.save_every == 0:
                rows = _analysis_rows(
                    accumulator,
                    args,
                    target_node_type,
                    source_names,
                    output_names,
                )
                _atomic_write_results(
                    output_dir,
                    rows,
                    metadata("running"),
                )

    rows = _analysis_rows(
        accumulator,
        args,
        target_node_type,
        source_names,
        output_names,
    )
    _atomic_write_results(output_dir, rows, metadata("complete"))
    if not args.no_plot:
        _plot_results(rows, output_dir)

    elapsed = time.monotonic() - start_time
    print(
        f"Complete: {counts['processed_samples']} samples and "
        f"{counts['processed_directions']} JVPs in {elapsed:.1f} s."
    )
    print(f"Wrote {output_dir / 'radial_jacobian.csv'}")
    print(f"Wrote {output_dir / 'radial_jacobian.json'}")
    if not args.no_plot:
        print(f"Wrote {output_dir / 'radial_jacobian_mean_abs.png'}")


if __name__ == "__main__":
    main()
