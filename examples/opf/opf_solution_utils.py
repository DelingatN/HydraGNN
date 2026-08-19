"""Shared utilities for OPF solution workflows (heterogeneous and homogeneous)."""

import copy
import logging
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import degree


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


class _LegacyStaticOPFDomainLoss:
    """Domain-informed regularization for OPF bus-level targets.

    Feasibility penalties (all zero on any strictly feasible OPF solution):
      - voltage_bound_weight           : Penalty for Vm (bus_pred[:, vm_output_index]) outside [v_min, v_max].
      - angle_diff_weight              : Penalty for predicted Va angle-difference outside line [theta_min, theta_max].
      - line_flow_weight               : Penalty for DC-approximate branch flow (DeltaVa / x_ij) exceeding rate_a.
      - line_flow_slack               : Tolerance subtracted from rate_a before penalising, absorbing the
                                        small linearisation error of the DC approximation on AC-feasible
                                        solutions.  Default 1e-4 (one decade above the ~1.3e-5 residual
                                        observed on pglib_opf_case10000_goc ground-truth data).

    Each raw penalty is normalized by a per-term exponential moving average (EMA)
    before the weight is applied.  This keeps every term near unit scale and makes
    the weights directly comparable to the task loss, regardless of the raw
    physical magnitudes (radians, per-unit power, etc.).
      - ema_momentum  (default 0.1): EMA decay.  Smaller = slower adaptation.

    Curriculum scheduling: domain-loss weights are ramped up gradually so the
    model first converges on the task loss before physics constraints are enforced.
      - warmup_epochs  (default 0): epochs with zero domain-loss weight.
      - ramp_epochs    (default 0): epochs over which weights linearly increase
                                    from 0 to their configured values.
    Example: warmup_epochs=3, ramp_epochs=3 with num_epoch=10 means:
      epochs 0-2: no domain loss, epochs 3-5: linear ramp, epochs 6-9: full weight.

    Feature-index conventions (derived from the gridopt/PyG OPFDataset schema):
      bus targets  : [Va (0), Vm (1)]
      ac_line attrs: [theta_min(0), theta_max(1), r_from(2), r_to(3), b_sh(4), x(5), rate_a(6), ...]
      transformer  : [theta_min(0), theta_max(1), r(2), x(3), rate_a(4), ...]
    """

    def __init__(self, config: dict | None = None, node_target_type: str = "bus"):
        cfg = copy.deepcopy(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.node_target_type = node_target_type
        self.voltage_bound_weight = float(cfg.get("voltage_bound_weight", 0.0))
        self.voltage_bound_feature_indices = cfg.get(
            "voltage_bound_feature_indices", None
        )
        # vm_output_index: index in bus_pred corresponding to voltage magnitude (Vm).
        # Default is 1 — bus targets are [Va, Vm] in the OPFDataset schema.
        self.voltage_output_index = int(cfg.get("voltage_output_index", 1))
        # va_output_index: index in bus_pred corresponding to voltage angle (Va).
        self.va_output_index = int(cfg.get("va_output_index", 0))
        self.angle_diff_weight = float(cfg.get("angle_diff_weight", 0.0))
        self.line_flow_weight = float(cfg.get("line_flow_weight", 0.0))
        # line_flow_slack: a small tolerance subtracted from rate_a before the DC thermal-limit
        # penalty is evaluated.  It exists because the DC power-flow formula
        #   P_ij = (Va_i - Va_j) / x_ij
        # is a linearisation of the full AC power-flow equations.  Even when the OPF solver
        # produces a strictly AC-feasible solution, the DC approximation introduces a residual
        # of ~1e-5 p.u. (empirically measured on pglib_opf_case10000_goc ground-truth data:
        # mean ~1.3e-5, max ~1.7e-5).  Without a slack the penalty is non-zero on ground truth,
        # which means the gradient incorrectly penalises physically correct predictions.
        # The default 1e-4 is one decade above the observed noise floor — large enough to zero
        # out the DC-approximation artefact but small enough to still penalise real violations.
        self.line_flow_slack = float(cfg.get("line_flow_slack", 1e-4))
        # EMA state for per-term scale normalization.
        self._ema_momentum = float(cfg.get("ema_momentum", 0.1))
        self._penalty_ema: dict[str, float] = {}
        # Curriculum scheduling.
        self.warmup_epochs = int(cfg.get("warmup_epochs", 0))
        self.ramp_epochs = int(cfg.get("ramp_epochs", 0))

        if self.voltage_bound_feature_indices is not None:
            if len(self.voltage_bound_feature_indices) != 2:
                raise RuntimeError(
                    "DomainLoss.voltage_bound_feature_indices must be [vmin_idx, vmax_idx]."
                )
            self.voltage_bound_feature_indices = tuple(
                int(v) for v in self.voltage_bound_feature_indices
            )


    def _curriculum_scale(self) -> float:
        """Return a [0, 1] multiplier for domain-loss weights based on current epoch.

        Reads os.environ["HYDRAGNN_EPOCH"] set by the HydraGNN training loop each
        epoch — no changes to shared training code are needed.
          - epoch < warmup_epochs          -> 0.0  (task-loss only)
          - warmup_epochs <= epoch < warmup + ramp -> linear ramp 0.0 -> 1.0
          - epoch >= warmup + ramp_epochs  -> 1.0  (full weight)
        """
        if self.warmup_epochs == 0 and self.ramp_epochs == 0:
            return 1.0
        try:
            epoch = int(os.environ.get("HYDRAGNN_EPOCH", "0"))
        except (ValueError, TypeError):
            return 1.0
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        progress = (epoch - self.warmup_epochs) / self.ramp_epochs
        return float(min(progress, 1.0))

    def _normalize(self, name: str, raw: torch.Tensor) -> torch.Tensor:
        """Normalize *raw* by its EMA so that the effective scale ≈ 1.0 on average.

        On the first call the EMA is seeded with the raw value, returning 1.0
        (or near-1.0 for non-zero values).  Subsequent calls use the smoothed
        estimate so the normalization adapts gradually as training progresses.
        """
        val = float(raw.detach())
        if name not in self._penalty_ema:
            # Seed: ema = raw value, normalized output = 1.0 on first step.
            self._penalty_ema[name] = max(val, 1e-8)
        else:
            m = self._ema_momentum
            self._penalty_ema[name] = max(
                m * val + (1.0 - m) * self._penalty_ema[name], 1e-8
            )
        # Floor at 1e-8 prevents division by zero when a penalty term is exactly zero
        # (e.g. the constraint is already satisfied for all samples in a batch).
        return raw / self._penalty_ema[name]

    def __call__(self, pred, value, head_index, data):
        if not self.enabled or data is None:
            return value.new_zeros(()), {}

        if self.node_target_type != "bus":
            return value.new_zeros(()), {}
        if not hasattr(data, "node_types") or "bus" not in data.node_types:
            return value.new_zeros(()), {}
        if len(pred) == 0:
            return value.new_zeros(()), {}

        bus_pred = pred[0]
        if bus_pred.dim() == 1:
            bus_pred = bus_pred.unsqueeze(-1)
        bus_true = value[head_index[0]]
        if bus_true.shape != bus_pred.shape:
            bus_true = bus_true.reshape_as(bus_pred)
        bus_true = bus_true.to(bus_pred.device)

        total_penalty = bus_pred.new_zeros(())
        metrics = {}
        curriculum = self._curriculum_scale()
        metrics["opf_curriculum_scale"] = torch.tensor(curriculum)

        if curriculum == 0.0:
            metrics["opf_domain_total"] = total_penalty.detach()
            return total_penalty, metrics

        if (
            self.voltage_bound_weight > 0.0
            and self.voltage_bound_feature_indices is not None
            and hasattr(data["bus"], "x")
        ):
            vmin_idx, vmax_idx = self.voltage_bound_feature_indices
            bus_x = data["bus"].x
            if bus_x.dim() >= 2 and bus_x.shape[1] > max(vmin_idx, vmax_idx):
                lower = bus_x[:, vmin_idx].reshape(-1)
                upper = bus_x[:, vmax_idx].reshape(-1)
                voltage = bus_pred[:, self.voltage_output_index].reshape(-1)
                # F.relu zeros out values that already satisfy the bound, so the gradient
                # is zero for feasible predictions and proportional to the violation otherwise.
                # Squaring gives a smooth (C1) penalty with growing gradient for larger violations.
                bound_penalty = torch.mean(
                    F.relu(lower - voltage).pow(2)
                    + F.relu(voltage - upper).pow(2)
                )
                total_penalty = (
                    total_penalty + curriculum * self.voltage_bound_weight * self._normalize("voltage_bound", bound_penalty)
                )
                metrics["opf_voltage_bound"] = bound_penalty.detach()

        # ── Angle difference limit penalty ──────────────────────────────────
        # Penalise predicted Va angle-differences that violate per-line bounds.
        #   ac_line  edge_attr: [theta_min(0), theta_max(1), ...]
        #   transformer edge_attr: [theta_min(0), theta_max(1), ...]
        if self.angle_diff_weight > 0.0 and bus_pred.shape[-1] > self.va_output_index:
            Va = bus_pred[:, self.va_output_index].reshape(-1)
            for rel, rel_tag in [
                (("bus", "ac_line", "bus"), "ac"),
                (("bus", "transformer", "bus"), "tr"),
            ]:
                if rel not in data.edge_types:
                    continue
                ea = getattr(data[rel], "edge_attr", None)
                ei = getattr(data[rel], "edge_index", None)
                if ea is None or ei is None or ea.numel() == 0 or ea.shape[1] < 2:
                    continue
                theta_min = ea[:, 0].to(Va.device)
                theta_max = ea[:, 1].to(Va.device)
                src, dst = ei
                delta_theta = Va[src] - Va[dst]
                # Same relu-squared form as voltage_bound: zero gradient inside the
                # feasible region [theta_min, theta_max], growing penalty outside it.
                # No slack is needed here: verified empirically that this term is exactly
                # zero on OPFDataset ground-truth solutions (Va and theta bounds share units).
                angdiff_p = torch.mean(
                    F.relu(delta_theta - theta_max).pow(2)
                    + F.relu(theta_min - delta_theta).pow(2)
                )
                total_penalty = total_penalty + curriculum * self.angle_diff_weight * self._normalize(f"{rel_tag}_angle_diff", angdiff_p)
                metrics[f"opf_{rel_tag}_angle_diff"] = angdiff_p.detach()

        # ── DC thermal limit penalty ─────────────────────────────────────────
        # Penalise approximate DC branch flows that exceed the thermal limit.
        #   P_ij = (Va_i - Va_j) / x_ij   (DC power flow approximation)
        #   ac_line:     x = edge_attr[:,5], rate_a = edge_attr[:,6]
        #   transformer: x = edge_attr[:,3], rate_a = edge_attr[:,4]
        if self.line_flow_weight > 0.0 and bus_pred.shape[-1] > self.va_output_index:
            Va = bus_pred[:, self.va_output_index].reshape(-1)
            for rel, x_idx, ra_idx, rel_tag in [
                (("bus", "ac_line", "bus"),    5, 6, "ac"),
                (("bus", "transformer", "bus"), 3, 4, "tr"),
            ]:
                if rel not in data.edge_types:
                    continue
                ea = getattr(data[rel], "edge_attr", None)
                ei = getattr(data[rel], "edge_index", None)
                if ea is None or ei is None or ea.numel() == 0 or ea.shape[1] <= max(x_idx, ra_idx):
                    continue
                # clamp x_ij away from zero to avoid division-by-zero in the DC formula;
                # 1e-6 p.u. is several orders of magnitude below any physical reactance.
                x_ij   = ea[:, x_idx].to(Va.device).clamp(min=1e-6)
                # clamp rate_a to be non-negative; negative thermal limits are nonsensical
                # and could arise from edge cases in dataset normalisation.
                rate_a = ea[:, ra_idx].to(Va.device).clamp(min=0.0)
                src, dst = ei
                # DC power-flow approximation: P_ij ≈ (Va_i - Va_j) / x_ij  [per unit].
                # This linearises the full AC formula sin(Va_i - Va_j) / x_ij and is only
                # exact in the flat-voltage, small-angle limit.
                P_ij = (Va[src] - Va[dst]) / x_ij
                # line_flow_slack is subtracted from rate_a to absorb the residual introduced
                # by the DC linearisation on AC-feasible solutions (see __init__ for details).
                # Without it, ground-truth predictions would incur a spurious non-zero penalty.
                flow_p = torch.mean(F.relu(P_ij.abs() - rate_a - self.line_flow_slack).pow(2))
                total_penalty = total_penalty + curriculum * self.line_flow_weight * self._normalize(f"{rel_tag}_line_flow", flow_p)
                metrics[f"opf_{rel_tag}_line_flow"] = flow_p.detach()

        metrics["opf_domain_total"] = total_penalty.detach()
        return total_penalty, metrics


_EQUALITY_CONSTRAINTS = (
    "power_balance_p",
    "power_balance_q",
)
_INEQUALITY_CONSTRAINTS = (
    "voltage_bounds",
    "ac_line_angle_bounds",
    "transformer_angle_bounds",
    "ac_line_apparent_power_limit",
    "transformer_apparent_power_limit",
    "generator_active_power_bounds",
    "generator_reactive_power_bounds",
)
_DIAGNOSTIC_CONSTRAINTS = (
    "ac_line_dc_flow_proxy",
    "transformer_dc_flow_proxy",
)
_LOSS_CONSTRAINTS = _EQUALITY_CONSTRAINTS + _INEQUALITY_CONSTRAINTS
_ALL_PHYSICS_CONSTRAINTS = _LOSS_CONSTRAINTS + _DIAGNOSTIC_CONSTRAINTS


def _safe_series_admittance(resistance, reactance):
    denominator = resistance.square() + reactance.square()
    denominator = denominator.clamp_min(1e-12)
    return resistance / denominator, -reactance / denominator


def wrapped_angle_difference(left, right):
    """Return the signed angular difference in [-pi, pi]."""

    difference = left - right
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def _physics_edge_attr(store):
    edge_attr = getattr(store, "edge_attr", None)
    if edge_attr is None:
        edge_attr = getattr(store, "physics_edge_attr", None)
    return edge_attr


def _signed_nonzero(values, floor=1e-6):
    floor_values = torch.where(
        values >= 0.0,
        values.new_full(values.shape, floor),
        values.new_full(values.shape, -floor),
    )
    return torch.where(values.abs() < floor, floor_values, values)


def _ac_branch_powers(voltage_angle, voltage_magnitude, edge_index, edge_attr, kind):
    """Return complex power leaving the from and to ends of each branch."""

    source, destination = edge_index.to(voltage_angle.device)
    magnitude_from = voltage_magnitude[source].float().clamp_min(1e-6)
    magnitude_to = voltage_magnitude[destination].float().clamp_min(1e-6)
    voltage_from = torch.polar(magnitude_from, voltage_angle[source].float())
    voltage_to = torch.polar(magnitude_to, voltage_angle[destination].float())
    attributes = edge_attr.to(device=voltage_angle.device, dtype=torch.float32)

    if kind == "ac_line":
        if attributes.dim() != 2 or attributes.size(1) < 7:
            raise RuntimeError(
                "AC-line physics requires edge_attr columns "
                "[angmin, angmax, b_fr, b_to, r, x, rate_a, ...]."
            )
        b_from, b_to = attributes[:, 2], attributes[:, 3]
        resistance, reactance = attributes[:, 4], attributes[:, 5]
        tap_complex = None
    elif kind == "transformer":
        if attributes.dim() != 2 or attributes.size(1) < 11:
            raise RuntimeError(
                "Transformer physics requires edge_attr columns "
                "[angmin, angmax, r, x, rates, tap, shift, b_fr, b_to]."
            )
        resistance, reactance = attributes[:, 2], attributes[:, 3]
        tap = attributes[:, 7].abs().clamp_min(1e-6)
        shift = attributes[:, 8]
        b_from, b_to = attributes[:, 9], attributes[:, 10]
        tap_complex = torch.polar(tap, shift)
    else:
        raise ValueError(f"Unknown AC branch kind '{kind}'.")

    reactance = _signed_nonzero(reactance)
    conductance, susceptance = _safe_series_admittance(resistance, reactance)
    series_admittance = torch.complex(conductance, susceptance)
    shunt_from = torch.complex(torch.zeros_like(b_from), b_from)
    shunt_to = torch.complex(torch.zeros_like(b_to), b_to)

    if tap_complex is None:
        current_from = (
            (series_admittance + shunt_from) * voltage_from
            - series_admittance * voltage_to
        )
        current_to = (
            (series_admittance + shunt_to) * voltage_to
            - series_admittance * voltage_from
        )
    else:
        tap_magnitude_squared = tap_complex.abs().square()
        current_from = (
            (series_admittance + shunt_from)
            / tap_magnitude_squared
            * voltage_from
            - series_admittance / torch.conj(tap_complex) * voltage_to
        )
        current_to = (
            (series_admittance + shunt_to) * voltage_to
            - series_admittance / tap_complex * voltage_from
        )

    power_from = voltage_from * torch.conj(current_from)
    power_to = voltage_to * torch.conj(current_to)
    return power_from, power_to


def _required_bus_generation(voltage_angle, voltage_magnitude, data):
    """Return complex generation required by full AC nodal balance."""

    num_bus = voltage_angle.numel()
    required = torch.zeros(
        num_bus, dtype=torch.complex64, device=voltage_angle.device
    )

    for relation, kind in (
        (("bus", "ac_line", "bus"), "ac_line"),
        (("bus", "transformer", "bus"), "transformer"),
    ):
        if relation not in data.edge_types:
            continue
        store = data[relation]
        edge_index = getattr(store, "edge_index", None)
        edge_attr = _physics_edge_attr(store)
        if edge_index is None or edge_attr is None:
            raise RuntimeError(
                f"AC power balance requires edge_index and edge_attr for {relation}."
            )
        if edge_index.numel() == 0:
            continue
        power_from, power_to = _ac_branch_powers(
            voltage_angle, voltage_magnitude, edge_index, edge_attr, kind
        )
        source, destination = edge_index.to(voltage_angle.device)
        required.index_add_(0, source, power_from)
        required.index_add_(0, destination, power_to)

    load_relation = ("load", "load_link", "bus")
    if "load" in data.node_types and int(data["load"].num_nodes) > 0:
        if load_relation not in data.edge_types:
            raise RuntimeError("AC power balance requires load_link edges.")
        load_features = data["load"].x.to(voltage_angle.device, torch.float32)
        if load_features.dim() != 2 or load_features.size(1) < 2:
            raise RuntimeError("Load physics requires [Pd, Qd] node features.")
        load_edges = data[load_relation].edge_index.to(voltage_angle.device)
        load_nodes, load_buses = load_edges
        if (
            load_nodes.numel() != load_features.size(0)
            or torch.unique(load_nodes).numel() != load_features.size(0)
            or int(load_nodes.min()) < 0
            or int(load_nodes.max()) >= load_features.size(0)
        ):
            raise RuntimeError("Each load must have exactly one load_link edge.")
        required.index_add_(
            0,
            load_buses,
            torch.complex(
                load_features[load_nodes, 0], load_features[load_nodes, 1]
            ),
        )

    shunt_relation = ("shunt", "shunt_link", "bus")
    if "shunt" in data.node_types and int(data["shunt"].num_nodes) > 0:
        if shunt_relation not in data.edge_types:
            raise RuntimeError("AC power balance requires shunt_link edges.")
        shunt_features = data["shunt"].x.to(voltage_angle.device, torch.float32)
        if shunt_features.dim() != 2 or shunt_features.size(1) < 2:
            raise RuntimeError(
                "Shunt physics requires [susceptance, conductance] node features."
            )
        shunt_edges = data[shunt_relation].edge_index.to(voltage_angle.device)
        shunt_nodes, shunt_buses = shunt_edges
        if (
            shunt_nodes.numel() != shunt_features.size(0)
            or torch.unique(shunt_nodes).numel() != shunt_features.size(0)
            or int(shunt_nodes.min()) < 0
            or int(shunt_nodes.max()) >= shunt_features.size(0)
        ):
            raise RuntimeError("Each shunt must have exactly one shunt_link edge.")
        # S_shunt = V^2 * conjugate(g + j b) = V^2 * (g - j b).
        shunt_power = torch.complex(
            shunt_features[shunt_nodes, 1], -shunt_features[shunt_nodes, 0]
        ) * voltage_magnitude[shunt_buses].float().square()
        required.index_add_(0, shunt_buses, shunt_power)

    return required


class OPFDomainLoss(torch.nn.Module):
    """Static legacy loss or family-level augmented-Lagrangian OPF loss.

    Omitting ``mode`` preserves the original fixed-weight/EMA behavior.  With
    ``mode='augmented_lagrangian'``, ``constraints`` explicitly selects the
    equality and inequality families included in the objective.  Every selected
    family owns one scalar dual variable, saved as a module buffer.
    """

    def __init__(self, config: dict | None = None, node_target_type="bus"):
        super().__init__()
        config = copy.deepcopy(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.node_target_type = node_target_type
        self.mode = str(config.get("mode", "static")).lower()
        aliases = {
            "al": "augmented_lagrangian",
            "monitor_only": "monitor",
        }
        self.mode = aliases.get(self.mode, self.mode)
        if self.mode not in {"static", "augmented_lagrangian", "monitor"}:
            raise ValueError(
                "DomainLoss.mode must be 'static', 'augmented_lagrangian', or "
                "'monitor'."
            )

        self._legacy = None
        if self.mode == "static":
            self._legacy = _LegacyStaticOPFDomainLoss(config, node_target_type)
            # Keep the legacy wrapper's checkpoint surface unchanged: the old
            # static loss held no tensors or module state of its own.
            self.selected_constraints = ()
            self.monitor_constraints = ()
            self.metric_names = ()
            self._epoch_sums = {}
            self._epoch_counts = {}
            self._warned_missing = set()
            return

        selected = config.get("constraints", [])
        if isinstance(selected, str):
            selected = [selected]
        self.selected_constraints = tuple(str(name) for name in selected)
        unknown_selected = set(self.selected_constraints) - set(_LOSS_CONSTRAINTS)
        if unknown_selected:
            raise ValueError(
                "Unknown or monitoring-only DomainLoss constraint(s): "
                f"{sorted(unknown_selected)}. Supported loss constraints are "
                f"{list(_LOSS_CONSTRAINTS)}."
            )
        if (
            self.enabled
            and self.mode == "augmented_lagrangian"
            and not self.selected_constraints
        ):
            raise ValueError(
                "Augmented-Lagrangian mode requires a non-empty constraints list."
            )

        monitor = config.get("monitor_constraints", "all")
        monitor_all = bool(config.get("monitor_all_constraints", True))
        if monitor_all or monitor == "all":
            monitor = list(_ALL_PHYSICS_CONSTRAINTS)
        elif isinstance(monitor, str):
            monitor = [monitor]
        else:
            monitor = list(monitor or [])
        unknown_monitor = set(monitor) - set(_ALL_PHYSICS_CONSTRAINTS)
        if unknown_monitor:
            raise ValueError(
                f"Unknown monitor constraint(s): {sorted(unknown_monitor)}."
            )
        self.monitor_constraints = tuple(
            dict.fromkeys(list(monitor) + list(self.selected_constraints))
        )

        self.rho = float(config.get("rho", config.get("al_rho", 1e-3)))
        if self.rho <= 0.0:
            raise ValueError("DomainLoss.rho must be positive.")
        self.va_output_index = int(config.get("va_output_index", 0))
        self.vm_output_index = int(config.get("voltage_output_index", 1))
        self.pg_output_index = int(config.get("pg_output_index", 0))
        self.qg_output_index = int(config.get("qg_output_index", 1))
        self.voltage_bound_feature_indices = tuple(
            int(value)
            for value in config.get("voltage_bound_feature_indices", [2, 3])
        )
        if len(self.voltage_bound_feature_indices) != 2:
            raise ValueError(
                "voltage_bound_feature_indices must contain [vmin, vmax]."
            )
        generator_indices = config.get(
            "generator_bound_feature_indices",
            {"pmin": 2, "pmax": 3, "qmin": 5, "qmax": 6},
        )
        self.generator_bound_feature_indices = {
            key: int(generator_indices[key])
            for key in ("pmin", "pmax", "qmin", "qmax")
        }
        self.dc_flow_slack = float(config.get("dc_flow_slack", 1e-4))
        initial_duals = config.get("initial_duals", {})
        for name in _EQUALITY_CONSTRAINTS:
            self.register_buffer(
                f"lambda_{name}",
                torch.tensor(float(initial_duals.get(name, 0.0))),
            )
        for name in _INEQUALITY_CONSTRAINTS:
            self.register_buffer(
                f"mu_{name}",
                torch.tensor(max(0.0, float(initial_duals.get(name, 0.0)))),
            )

        self._epoch_sums = {name: 0.0 for name in self.selected_constraints}
        self._epoch_counts = {name: 0 for name in self.selected_constraints}
        self._warned_missing = set()
        metric_names = ["physics_penalty_total"]
        for name in self.monitor_constraints:
            metric_names.extend(
                (
                    f"physics_{name}_mean_violation",
                    f"physics_{name}_max_violation",
                    f"physics_{name}_mse",
                    f"physics_{name}_signed_mean",
                    f"physics_{name}_loss",
                    f"physics_{name}_dual",
                )
            )
        self.metric_names = tuple(metric_names)

    def _target_head_index(self, target_type):
        targets = self.node_target_type
        if isinstance(targets, (list, tuple)):
            try:
                return list(targets).index(target_type)
            except ValueError:
                return None
        return 0 if targets == target_type else None

    def _missing(self, name, message):
        if name in self.selected_constraints:
            raise RuntimeError(f"Selected physics constraint '{name}': {message}")
        if name not in self._warned_missing:
            info(
                f"Skipping monitored physics constraint '{name}': {message}",
                logtype="warning",
            )
            self._warned_missing.add(name)

    def _prediction_for_type(self, pred, target_type):
        index = self._target_head_index(target_type)
        if index is None or index >= len(pred):
            return None
        result = pred[index]
        if isinstance(result, (tuple, list)):
            result = result[0]
        if result.dim() == 1:
            result = result.unsqueeze(-1)
        return result

    def _constraint_residuals(self, pred, data):
        requested = set(self.monitor_constraints) | set(self.selected_constraints)
        residuals = {}
        bus_pred = self._prediction_for_type(pred, "bus")
        if bus_pred is None:
            for name in requested:
                self._missing(name, "the model has no bus prediction head.")
            return residuals
        required_bus_width = max(self.va_output_index, self.vm_output_index) + 1
        if bus_pred.size(1) < required_bus_width:
            for name in requested:
                self._missing(
                    name,
                    f"bus predictions need at least {required_bus_width} columns.",
                )
            return residuals

        voltage_angle = bus_pred[:, self.va_output_index].reshape(-1)
        voltage_magnitude = bus_pred[:, self.vm_output_index].reshape(-1)
        generator_pred = self._prediction_for_type(pred, "generator")

        if "voltage_bounds" in requested:
            bus_features = getattr(data["bus"], "x", None)
            vmin_index, vmax_index = self.voltage_bound_feature_indices
            if (
                bus_features is None
                or bus_features.dim() != 2
                or bus_features.size(1) <= max(vmin_index, vmax_index)
            ):
                self._missing("voltage_bounds", "bus voltage-bound features are absent.")
            else:
                bus_features = bus_features.to(voltage_magnitude.device)
                lower = bus_features[:, vmin_index]
                upper = bus_features[:, vmax_index]
                residuals["voltage_bounds"] = (
                    "inequality",
                    torch.cat(
                        (lower - voltage_magnitude, voltage_magnitude - upper)
                    ),
                )

        relation_specs = (
            (
                "ac_line",
                ("bus", "ac_line", "bus"),
                "ac_line_angle_bounds",
                "ac_line_apparent_power_limit",
                "ac_line_dc_flow_proxy",
                5,
                6,
            ),
            (
                "transformer",
                ("bus", "transformer", "bus"),
                "transformer_angle_bounds",
                "transformer_apparent_power_limit",
                "transformer_dc_flow_proxy",
                3,
                4,
            ),
        )
        for (
            kind,
            relation,
            angle_name,
            apparent_name,
            dc_name,
            reactance_index,
            rate_index,
        ) in relation_specs:
            relation_requested = requested & {angle_name, apparent_name, dc_name}
            if not relation_requested:
                continue
            minimum_width = 11 if kind == "transformer" else 7
            # Report absence against each requested family, not an arbitrary one.
            inputs = None
            if relation in data.edge_types:
                store = data[relation]
                edge_index = getattr(store, "edge_index", None)
                edge_attr = _physics_edge_attr(store)
                if (
                    edge_index is not None
                    and edge_attr is not None
                    and edge_index.numel() > 0
                    and edge_attr.dim() == 2
                    and edge_attr.size(1) >= minimum_width
                ):
                    inputs = (edge_index, edge_attr)
            if inputs is None:
                for name in relation_requested:
                    self._missing(
                        name,
                        f"{relation} with edge_attr width {minimum_width} is required.",
                    )
                continue

            edge_index, edge_attr = inputs
            edge_index_device = edge_index.to(voltage_angle.device)
            edge_attr_device = edge_attr.to(voltage_angle.device, torch.float32)
            source, destination = edge_index_device
            angle_difference = wrapped_angle_difference(
                voltage_angle[source], voltage_angle[destination]
            )

            if angle_name in requested:
                minimum_angle = edge_attr_device[:, 0]
                maximum_angle = edge_attr_device[:, 1]
                residuals[angle_name] = (
                    "inequality",
                    torch.cat(
                        (
                            angle_difference - maximum_angle,
                            minimum_angle - angle_difference,
                        )
                    ),
                )

            if apparent_name in requested:
                power_from, power_to = _ac_branch_powers(
                    voltage_angle,
                    voltage_magnitude,
                    edge_index_device,
                    edge_attr_device,
                    kind,
                )
                rate = edge_attr_device[:, rate_index]
                rated = rate > 0.0
                if not torch.any(rated):
                    self._missing(apparent_name, "no branch has a positive rate_a.")
                else:
                    residuals[apparent_name] = (
                        "inequality",
                        torch.cat(
                            (
                                power_from.abs()[rated] - rate[rated],
                                power_to.abs()[rated] - rate[rated],
                            )
                        ),
                    )

            if dc_name in requested:
                reactance = _signed_nonzero(
                    edge_attr_device[:, reactance_index]
                )
                dc_angle = angle_difference
                denominator = reactance
                if kind == "transformer":
                    dc_angle = wrapped_angle_difference(
                        angle_difference, edge_attr_device[:, 8]
                    )
                    denominator = denominator * edge_attr_device[:, 7].abs().clamp_min(
                        1e-6
                    )
                rate = edge_attr_device[:, rate_index]
                rated = rate > 0.0
                if not torch.any(rated):
                    self._missing(dc_name, "no branch has a positive rate_a.")
                else:
                    proxy = (dc_angle / denominator).abs()
                    residuals[dc_name] = (
                        "diagnostic",
                        proxy[rated] - rate[rated] - self.dc_flow_slack,
                    )

        balance_requested = requested & set(_EQUALITY_CONSTRAINTS)
        if balance_requested:
            generator_relation = ("generator", "generator_link", "bus")
            if generator_pred is None:
                for name in balance_requested:
                    self._missing(name, "the model has no generator prediction head.")
            elif generator_relation not in data.edge_types:
                for name in balance_requested:
                    self._missing(name, "generator_link edges are absent.")
            elif generator_pred.size(1) <= max(
                self.pg_output_index, self.qg_output_index
            ):
                for name in balance_requested:
                    self._missing(name, "generator predictions need Pg and Qg columns.")
            else:
                required_generation = _required_bus_generation(
                    voltage_angle, voltage_magnitude, data
                )
                generator_edges = data[generator_relation].edge_index.to(
                    voltage_angle.device
                )
                generator_nodes, generator_buses = generator_edges
                if (
                    generator_nodes.numel() != generator_pred.size(0)
                    or torch.unique(generator_nodes).numel() != generator_pred.size(0)
                    or int(generator_nodes.min()) < 0
                    or int(generator_nodes.max()) >= generator_pred.size(0)
                ):
                    for name in balance_requested:
                        self._missing(
                            name,
                            "each generator must have exactly one generator_link edge.",
                        )
                else:
                    predicted_generation = torch.zeros_like(required_generation)
                    generator_values = generator_pred[generator_nodes]
                    predicted_generation.index_add_(
                        0,
                        generator_buses,
                        torch.complex(
                            generator_values[:, self.pg_output_index].float(),
                            generator_values[:, self.qg_output_index].float(),
                        ),
                    )
                    mismatch = predicted_generation - required_generation
                    if "power_balance_p" in requested:
                        residuals["power_balance_p"] = ("equality", mismatch.real)
                    if "power_balance_q" in requested:
                        residuals["power_balance_q"] = ("equality", mismatch.imag)

        generator_requested = requested & {
            "generator_active_power_bounds",
            "generator_reactive_power_bounds",
        }
        if generator_requested:
            generator_features = (
                getattr(data["generator"], "x", None)
                if "generator" in data.node_types
                else None
            )
            maximum_feature_index = max(self.generator_bound_feature_indices.values())
            if generator_pred is None:
                for name in generator_requested:
                    self._missing(name, "the model has no generator prediction head.")
            elif (
                generator_features is None
                or generator_features.dim() != 2
                or generator_features.size(1) <= maximum_feature_index
            ):
                for name in generator_requested:
                    self._missing(name, "generator bound features are absent.")
            elif generator_pred.size(1) <= max(
                self.pg_output_index, self.qg_output_index
            ):
                for name in generator_requested:
                    self._missing(name, "generator predictions need Pg and Qg columns.")
            else:
                generator_features = generator_features.to(generator_pred.device)
                indices = self.generator_bound_feature_indices
                if "generator_active_power_bounds" in requested:
                    pg = generator_pred[:, self.pg_output_index]
                    residuals["generator_active_power_bounds"] = (
                        "inequality",
                        torch.cat(
                            (
                                generator_features[:, indices["pmin"]] - pg,
                                pg - generator_features[:, indices["pmax"]],
                            )
                        ),
                    )
                if "generator_reactive_power_bounds" in requested:
                    qg = generator_pred[:, self.qg_output_index]
                    residuals["generator_reactive_power_bounds"] = (
                        "inequality",
                        torch.cat(
                            (
                                generator_features[:, indices["qmin"]] - qg,
                                qg - generator_features[:, indices["qmax"]],
                            )
                        ),
                    )

        return residuals

    def _dual(self, name):
        prefix = "lambda" if name in _EQUALITY_CONSTRAINTS else "mu"
        return getattr(self, f"{prefix}_{name}")

    def _accumulate_dual_statistic(self, name, values):
        self._epoch_sums[name] += float(values.detach().double().sum().cpu())
        self._epoch_counts[name] += int(values.numel())

    def update_duals(self):
        """Apply one globally synchronized dual-ascent update for the epoch."""

        if not self.enabled or self.mode != "augmented_lagrangian":
            return {}
        if not self.selected_constraints:
            return {}
        device = self._dual(self.selected_constraints[0]).device
        statistics = torch.tensor(
            [
                [self._epoch_sums[name], float(self._epoch_counts[name])]
                for name in self.selected_constraints
            ],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)

        updated = {}
        with torch.no_grad():
            for index, name in enumerate(self.selected_constraints):
                count = statistics[index, 1]
                if count <= 0:
                    raise RuntimeError(
                        f"No training residuals were accumulated for selected "
                        f"constraint '{name}'."
                    )
                mean = statistics[index, 0] / count
                dual = self._dual(name)
                dual.add_(self.rho * mean.to(dual.dtype))
                if name in _INEQUALITY_CONSTRAINTS:
                    dual.clamp_(min=0.0)
                updated[name] = float(dual.detach().cpu())

        for name in self.selected_constraints:
            self._epoch_sums[name] = 0.0
            self._epoch_counts[name] = 0
        return updated

    def forward(self, pred, value, head_index, data, update_state=False):
        if not self.enabled or data is None:
            return value.new_zeros(()), {}
        if self.mode == "static":
            return self._legacy(pred, value, head_index, data)
        if not hasattr(data, "node_types") or "bus" not in data.node_types:
            raise RuntimeError("Physics constraints require heterogeneous bus data.")

        residuals = self._constraint_residuals(pred, data)
        reference = self._prediction_for_type(pred, "bus")
        total = reference.new_zeros(())
        metrics = {}
        contributions = {}

        for name in self.selected_constraints:
            if name not in residuals:
                raise RuntimeError(
                    f"Selected physics constraint '{name}' produced no residuals."
                )
            kind, raw = residuals[name]
            if raw.numel() == 0:
                raise RuntimeError(
                    f"Selected physics constraint '{name}' has an empty residual."
                )
            if not torch.isfinite(raw).all():
                raise RuntimeError(
                    f"Selected physics constraint '{name}' produced non-finite values."
                )
            values = raw if kind == "equality" else F.relu(raw)
            dual = self._dual(name).to(values.device, values.dtype)
            contribution = dual * values.mean() + 0.5 * self.rho * values.square().mean()
            total = total + contribution
            contributions[name] = contribution
            if update_state:
                self._accumulate_dual_statistic(name, values)

        for name in self.monitor_constraints:
            if name not in residuals:
                continue
            kind, raw = residuals[name]
            violation = raw.abs() if kind == "equality" else F.relu(raw)
            if violation.numel() == 0:
                continue
            metrics[f"physics_{name}_mean_violation"] = violation.mean().detach()
            metrics[f"physics_{name}_max_violation"] = violation.max().detach()
            metrics[f"physics_{name}_mse"] = violation.square().mean().detach()
            metrics[f"physics_{name}_signed_mean"] = raw.mean().detach()
            metrics[f"physics_{name}_loss"] = contributions.get(
                name, total.new_zeros(())
            ).detach()
            metrics[f"physics_{name}_dual"] = (
                self._dual(name).detach()
                if name in _LOSS_CONSTRAINTS
                else total.new_zeros(()).detach()
            )

        metrics["physics_penalty_total"] = total.detach()
        if self.mode == "monitor":
            return total.new_zeros(()), metrics
        return total, metrics


class OPFEnhancedModelWrapper(torch.nn.Module):
    """Add OPF physics loss, dual updates, and per-split diagnostics."""

    def __init__(self, original_model, domain_loss: OPFDomainLoss):
        super().__init__()
        self.model = original_model
        self.domain_loss = domain_loss
        self._last_batch = None
        self.last_extra_loss_metrics = {}
        self._split_accum = {
            split: {} for split in ("train", "val", "test")
        }

    def _accumulate_metric(self, split, name, value):
        bucket = self._split_accum.setdefault(split, {})
        entry = bucket.setdefault(name, [0.0, 0])
        scalar = float(value.detach().double().cpu())
        if name.endswith("_max_violation"):
            entry[0] = max(entry[0], scalar)
        else:
            entry[0] += scalar
        entry[1] += 1

    def update_physics_duals(self, epoch=None, writer=None):
        updated = self.domain_loss.update_duals()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and updated:
            fields = "  ".join(
                f"{name}={value:.8g}" for name, value in updated.items()
            )
            print(f"0: PhysicsDuals epoch={int(epoch):02d}  {fields}", flush=True)
            if writer is not None:
                for name, value in updated.items():
                    writer.add_scalar(f"physics/dual/{name}", value, epoch)
        return updated

    def finalize_physics_epoch(self, epoch, writer=None):
        """DDP-reduce and log train/validation/test physics diagnostics."""

        fixed_keys = ["data_driven_mse"] + list(self.domain_loss.metric_names)
        # Preserve visibility of the original static-loss metrics too.
        fixed_keys.extend(
            (
                "opf_domain_total",
                "opf_curriculum_scale",
                "opf_voltage_bound",
                "opf_ac_angle_diff",
                "opf_tr_angle_diff",
                "opf_ac_line_flow",
                "opf_tr_line_flow",
            )
        )
        fixed_keys = list(dict.fromkeys(fixed_keys))
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        rank = dist.get_rank() if dist.is_initialized() else 0
        for split in ("train", "val", "test"):
            bucket = self._split_accum.get(split, {})
            sums = torch.zeros(len(fixed_keys), dtype=torch.float64, device=device)
            counts = torch.zeros(len(fixed_keys), dtype=torch.float64, device=device)
            maxima = torch.zeros(len(fixed_keys), dtype=torch.float64, device=device)
            for index, name in enumerate(fixed_keys):
                if name in bucket:
                    counts[index] = bucket[name][1]
                    if name.endswith("_max_violation"):
                        maxima[index] = bucket[name][0]
                    else:
                        sums[index] = bucket[name][0]
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)

            if rank == 0:
                fields = []
                for index, name in enumerate(fixed_keys):
                    count = float(counts[index])
                    if count <= 0.0:
                        continue
                    aggregate = (
                        float(maxima[index])
                        if name.endswith("_max_violation")
                        else float(sums[index] / count)
                    )
                    fields.append(f"{name}={aggregate:.8g}")
                    if writer is not None:
                        writer.add_scalar(
                            f"physics/{split}/{name}", aggregate, epoch
                        )
                if fields:
                    print(
                        f"0: PhysicsBreakdown epoch={int(epoch):02d} "
                        f"split={split}  " + "  ".join(fields),
                        flush=True,
                    )
        for bucket in self._split_accum.values():
            bucket.clear()

    def _flush_epoch_log(self, epoch):
        """Compatibility alias for older OPF training scripts."""

        self.finalize_physics_epoch(epoch)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, data):
        self._last_batch = data
        return self.model(data)

    def loss(self, pred, value, head_index):
        total_loss, tasks_loss = self.model.loss(pred, value, head_index)
        if self._last_batch is None:
            info(
                "[OPFEnhancedModelWrapper] loss() called before forward(); "
                "domain penalty will be zero for this batch.",
                logtype="warning",
            )
        extra_loss, extra_metrics = self.domain_loss(
            pred,
            value,
            head_index,
            self._last_batch,
            update_state=(
                self.training
                and os.environ.get("HYDRAGNN_PHASE", "train") == "train"
            ),
        )
        self.last_extra_loss_metrics = extra_metrics

        phase = os.environ.get("HYDRAGNN_PHASE")
        if phase not in {"train", "val", "test"}:
            phase = "train" if self.training else "val"
        self._accumulate_metric(phase, "data_driven_mse", total_loss)
        for key, val in extra_metrics.items():
            self._accumulate_metric(phase, key, val)

        return total_loss + extra_loss, tasks_loss


def build_solution_target(data, node_target_type: str):
    """Extract the solution target tensor for the given node type."""
    if hasattr(data, "node_types") and node_target_type in data.node_types:
        node_store = data[node_target_type]
        if not hasattr(node_store, "y") or node_store.y is None:
            raise RuntimeError(
                f"No targets found for node type '{node_target_type}' in OPF sample."
            )
        return node_store.y.to(torch.float32)

    if hasattr(data, "_node_type_names") and hasattr(data, "node_type"):
        if node_target_type not in data._node_type_names:
            raise RuntimeError(
                f"Node type '{node_target_type}' not found in OPF sample."
            )
        type_index = data._node_type_names.index(node_target_type)
        if not hasattr(data, "y") or data.y is None:
            raise RuntimeError(
                f"No homogeneous targets found for node type '{node_target_type}'."
            )
        mask = data.node_type == type_index
        return data.y[mask].to(torch.float32)

    raise RuntimeError(f"Node type '{node_target_type}' not found in OPF sample.")


def pack_node_targets(data, node_target_types):
    """Pack differently sized node targets head-by-head for HydraGNN."""

    if isinstance(node_target_types, str):
        node_target_types = [node_target_types]
    if not node_target_types:
        raise ValueError("node_target_types must contain at least one node type.")

    targets = [build_solution_target(data, name) for name in node_target_types]
    locations = [0]
    for target in targets:
        locations.append(locations[-1] + int(target.numel()))
    data.y = torch.cat([target.reshape(-1) for target in targets], dim=0)
    data.y_loc = torch.tensor([locations], dtype=torch.int64, device=data.y.device)
    data.y_num_nodes = torch.tensor(
        [int(target.shape[0]) for target in targets],
        dtype=torch.int64,
        device=data.y.device,
    )
    return data


def ensure_node_y_loc(data):
    if not hasattr(data, "y") or data.y is None:
        raise RuntimeError("Missing node targets (data.y) for OPF sample.")
    if data.y.dim() == 1:
        data.y = data.y.unsqueeze(-1)
    num_nodes = int(data.y.shape[0])
    target_dim = int(data.y.shape[1])
    data.y_num_nodes = torch.tensor(
        [num_nodes], dtype=torch.int64, device=data.y.device
    )
    data.y_loc = torch.tensor(
        [[0, num_nodes * target_dim]],
        dtype=torch.int64,
        device=data.y.device,
    )


def resolve_node_target_type(data, requested: str) -> str:
    if hasattr(data, "node_types"):
        if requested in data.node_types:
            return requested
        if hasattr(data, "_node_type_names") and requested in data._node_type_names:
            idx = data._node_type_names.index(requested)
            if idx < len(data.node_types):
                return data.node_types[idx]
        raise RuntimeError(
            f"Requested node_target_type '{requested}' not found in data. "
            f"Available node types: {list(data.node_types)}."
        )
    if hasattr(data, "_node_type_names") and requested in data._node_type_names:
        return requested
    raise RuntimeError(
        f"Cannot resolve node_target_type '{requested}': data has no node_types."
    )


def _as_edge_feature(value, num_edges: int, device):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except Exception:
            return None
    if value.numel() == 0:
        return None
    if value.dim() == 0:
        return None
    if value.dim() == 1:
        if int(value.shape[0]) != int(num_edges):
            return None
        value = value.view(-1, 1)
    elif value.dim() >= 2:
        if int(value.shape[0]) != int(num_edges):
            return None
        value = value.reshape(num_edges, -1)
    if value.dtype not in (torch.float16, torch.float32, torch.float64):
        value = value.to(torch.float32)
    return value.to(device=device, dtype=torch.float32)


def resolve_edge_feature_schema(
    configured_feature_names=None,
    configured_edge_dim=None,
):
    if configured_feature_names is None or len(configured_feature_names) == 0:
        raise RuntimeError(
            "edge_feature_names must be explicitly provided in the config. "
            "No implicit defaults are used."
        )
    schema = [str(name) for name in configured_feature_names if str(name).strip()]
    if not schema:
        raise RuntimeError("edge_feature_names contains only empty/whitespace entries.")
    if configured_edge_dim is not None:
        edge_dim = int(configured_edge_dim)
        if edge_dim != len(schema):
            raise RuntimeError(
                f"edge_dim={edge_dim} does not match the number of "
                f"edge_feature_names ({len(schema)}). They must be equal."
            )
    return tuple(schema)


def validate_voi_node_features(config: dict, node_target_type: str | None = None):
    """Validate that node feature config is fully specified.  Crash on anything missing."""
    nn_config = config.get("NeuralNetwork")
    if nn_config is None:
        raise RuntimeError("Config is missing 'NeuralNetwork' section.")
    var_config = nn_config.get("Variables_of_interest")
    if var_config is None:
        raise RuntimeError("Config is missing 'NeuralNetwork.Variables_of_interest'.")

    input_node_features = var_config.get("input_node_features")
    if not isinstance(input_node_features, list) or len(input_node_features) == 0:
        raise RuntimeError(
            "'input_node_features' must be an explicit non-empty list in the config."
        )

    node_feature_dims = var_config.get("node_feature_dims")
    if not isinstance(node_feature_dims, list) or len(node_feature_dims) == 0:
        raise RuntimeError(
            "'node_feature_dims' must be an explicit non-empty list in the config."
        )

    if "node_feature_names" not in var_config:
        raise RuntimeError(
            "'node_feature_names' must be explicitly provided in the config."
        )

    return config


def compute_pna_deg_for_hetero_dataset(dataset, verbosity: int = 2):
    from hydragnn.utils.print.print_utils import iterate_tqdm

    num_samples = len(dataset)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        start = (num_samples * rank) // world_size
        end = (num_samples * (rank + 1)) // world_size
    else:
        start = 0
        end = num_samples

    local_indices = range(start, end)

    max_deg_local = 0
    for idx in iterate_tqdm(local_indices, verbosity, desc="HeteroPNA degree max"):
        data = dataset[idx]
        data_h = data.to_homogeneous(add_node_type=True, add_edge_type=True)
        d = degree(data_h.edge_index[1], num_nodes=data_h.num_nodes, dtype=torch.long)
        if d.numel() > 0:
            max_deg_local = max(max_deg_local, int(d.max().item()))

    if dist.is_initialized():
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        max_deg_tensor = torch.tensor(
            [max_deg_local], dtype=torch.long, device=reduce_device
        )
        dist.all_reduce(max_deg_tensor, op=dist.ReduceOp.MAX)
        max_deg = int(max_deg_tensor.item())
    else:
        max_deg = max_deg_local

    deg_local = torch.zeros(max_deg + 1, dtype=torch.long)
    for idx in iterate_tqdm(local_indices, verbosity, desc="HeteroPNA degree bincount"):
        data = dataset[idx]
        data_h = data.to_homogeneous(add_node_type=True, add_edge_type=True)
        d = degree(data_h.edge_index[1], num_nodes=data_h.num_nodes, dtype=torch.long)
        deg_local += torch.bincount(d, minlength=deg_local.numel())

    if dist.is_initialized():
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        deg_tensor = deg_local.to(device=reduce_device)
        dist.all_reduce(deg_tensor, op=dist.ReduceOp.SUM)
        deg = deg_tensor.cpu()
    else:
        deg = deg_local

    return deg.tolist()


def _assemble_edge_attr_hetero(data, edge_dim_dict):
    """Heterogeneous route: keep per-edge-type native widths.

    Edge types whose relation name appears in *edge_dim_dict* must carry a
    pre-assembled ``edge_attr`` tensor with the declared width.  Edge types
    absent from the dict are treated as featureless — any stale ``edge_attr``
    is removed so that ``data.edge_attr_dict`` only contains featured types.

    Returns ``(data, edge_dim_dict)`` unchanged.
    """
    for edge_type in data.edge_types:
        _, rel, _ = edge_type
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue

        expected_dim = edge_dim_dict.get(rel)
        existing = getattr(edge_store, "edge_attr", None)

        if expected_dim is None:
            # Featureless — remove any edge_attr so it stays out of
            # data.edge_attr_dict during training.
            if existing is not None:
                try:
                    delattr(edge_store, "edge_attr")
                except AttributeError:
                    pass
            continue

        if not isinstance(existing, torch.Tensor) or existing.dim() != 2:
            raise RuntimeError(
                f"Edge type {edge_type} (rel={rel}) expects edge_attr with "
                f"{expected_dim} columns but found no valid 2-D tensor."
            )
        if existing.size(1) != expected_dim:
            raise RuntimeError(
                f"Edge type {edge_type} (rel={rel}) has edge_attr width "
                f"{existing.size(1)}, expected {expected_dim} from edge_dim config."
            )

    return data, edge_dim_dict


def assemble_edge_attr(data, edge_dim, feature_schema=None):
    """One-time assembly during preprocessing.

    *edge_dim* determines the route:

    * **int** — *homogeneous* route.  Every edge type is zero-padded (or
      assembled from named columns via *feature_schema*) to a uniform width
      equal to *edge_dim*.
    * **dict** — *heterogeneous* route.  Keys are relation names (the middle
      element of an edge-type triple); values are the expected widths of
      pre-assembled ``edge_attr`` tensors.  Edge types absent from the dict
      are treated as featureless.

    Returns ``(data, edge_dim)``.
    """
    if not hasattr(data, "edge_types"):
        return data, edge_dim

    if isinstance(edge_dim, dict):
        return _assemble_edge_attr_hetero(data, edge_dim)

    target_dim = int(edge_dim)
    if target_dim <= 0:
        raise RuntimeError("int edge_dim must be positive.")

    schema = None
    if feature_schema is not None:
        schema = tuple(str(n) for n in feature_schema if str(n).strip())
        if not schema:
            schema = None

    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))
        device = edge_index.device

        # Already assembled — accept as-is or zero-pad to target_dim.
        existing = getattr(edge_store, "edge_attr", None)
        if (
            isinstance(existing, torch.Tensor)
            and existing.dim() == 2
            and existing.size(0) == num_edges
        ):
            w = existing.size(1)
            if w == target_dim:
                continue  # exact match
            if w < target_dim:
                pad = torch.zeros(
                    num_edges, target_dim - w, device=device, dtype=existing.dtype
                )
                data[edge_type].edge_attr = torch.cat([existing, pad], dim=1)
                continue
            raise RuntimeError(
                f"edge_attr for {edge_type} has {w} columns, exceeding edge_dim={target_dim}."
            )

        # Try named-column assembly if a schema was provided.
        if schema is not None:
            has_any = any(
                getattr(edge_store, name, None) is not None for name in schema
            )
            if not has_any and existing is None:
                data[edge_type].edge_attr = torch.zeros(
                    num_edges, target_dim, device=device, dtype=torch.float32
                )
                continue

            cols = []
            for attr_name in schema:
                col = _as_edge_feature(
                    getattr(edge_store, attr_name, None), num_edges, device
                )
                if col is None:
                    raise RuntimeError(
                        f"Missing or invalid edge attribute '{attr_name}' "
                        f"for edge type {edge_type}."
                    )
                if int(col.shape[1]) != 1:
                    raise RuntimeError(
                        f"Edge attribute '{attr_name}' for edge type {edge_type} has "
                        f"{int(col.shape[1])} columns; expected exactly 1."
                    )
                cols.append(col)

            data[edge_type].edge_attr = torch.cat(cols, dim=1).contiguous()

            for attr_name in schema:
                try:
                    delattr(edge_store, attr_name)
                except AttributeError:
                    pass
            continue

        # No schema and no existing tensor — zero-fill.
        if existing is None:
            data[edge_type].edge_attr = torch.zeros(
                num_edges, target_dim, device=device, dtype=torch.float32
            )

    return data, target_dim


def _validate_edge_attr_hetero(data, edge_dim_dict):
    """Check per-edge-type widths for the heterogeneous route."""
    for edge_type in data.edge_types:
        _, rel, _ = edge_type
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))

        expected_dim = edge_dim_dict.get(rel)
        edge_attr = getattr(edge_store, "edge_attr", None)

        if expected_dim is None:
            # Featureless — must NOT have edge_attr.
            if isinstance(edge_attr, torch.Tensor):
                raise RuntimeError(
                    f"Featureless edge type {edge_type} (rel={rel}) should not "
                    f"have edge_attr, but found tensor with shape {list(edge_attr.shape)}."
                )
            continue

        if not isinstance(edge_attr, torch.Tensor):
            raise RuntimeError(
                f"Edge type {edge_type} (rel={rel}) is missing edge_attr; "
                f"expected width {expected_dim}."
            )
        if edge_attr.dim() != 2:
            raise RuntimeError(
                f"edge_attr for edge type {edge_type} has {edge_attr.dim()} "
                f"dimensions; expected 2."
            )
        if edge_attr.size(0) != num_edges:
            raise RuntimeError(
                f"edge_attr row count mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(0)}, expected {num_edges}."
            )
        if edge_attr.size(1) != expected_dim:
            raise RuntimeError(
                f"edge_attr dim mismatch for edge type {edge_type} (rel={rel}): "
                f"got {edge_attr.size(1)}, expected {expected_dim}."
            )

    return data


def validate_edge_attr(data, edge_dim):
    """Validate that every edge type carries properly shaped ``edge_attr``.

    *edge_dim* can be:

    * **int** — every edge type must have ``edge_attr`` with that many columns
      (featureless types that have no ``edge_attr`` are silently skipped).
    * **dict** — per-relation-name widths; featureless types (absent from the
      dict) must NOT carry ``edge_attr``.
    """
    if not hasattr(data, "edge_types"):
        return data

    if isinstance(edge_dim, dict):
        return _validate_edge_attr_hetero(data, edge_dim)

    target_dim = int(edge_dim)

    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))

        edge_attr = getattr(edge_store, "edge_attr", None)
        if not isinstance(edge_attr, torch.Tensor):
            continue
        if edge_attr.dim() != 2:
            raise RuntimeError(
                f"edge_attr for edge type {edge_type} has "
                f"{edge_attr.dim()} dimensions; expected 2."
            )
        if edge_attr.size(0) != num_edges:
            raise RuntimeError(
                f"edge_attr row count mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(0)}, expected {num_edges}."
            )
        if edge_attr.size(1) != target_dim:
            raise RuntimeError(
                f"edge_attr dim mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(1)}, expected {target_dim}."
            )

    return data


class HeteroFromHomogeneousDataset:
    """Wraps an ADIOS-loaded homogeneous dataset, converting each sample to
    heterogeneous and validating ``edge_attr`` shape.
    """

    def __init__(self, base, edge_dim: int):
        self.base = base
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        hetero = data.to_heterogeneous()
        if hasattr(data, "y"):
            hetero.y = data.y
        if hasattr(data, "y_loc"):
            hetero.y_loc = data.y_loc
        if hasattr(data, "y_num_nodes"):
            hetero.y_num_nodes = data.y_num_nodes
        if hasattr(data, "graph_attr"):
            hetero.graph_attr = data.graph_attr
        validate_edge_attr(hetero, self.edge_dim)
        return hetero


class EdgeAttrDatasetAdapter:
    """Validates ``edge_attr`` on every access — no assembly, just shape check."""

    def __init__(self, base, edge_dim: int):
        self.base = base
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        validate_edge_attr(data, self.edge_dim)
        return data

    def __getattr__(self, name):
        return getattr(self.base, name)


class NodeTargetDatasetAdapter:
    def __init__(self, base, node_target_type, edge_dim: int):
        self.base = base
        self.node_target_types = (
            [node_target_type]
            if isinstance(node_target_type, str)
            else list(node_target_type)
        )
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        validate_edge_attr(data, self.edge_dim)
        for node_type in self.node_target_types:
            if not hasattr(data, "node_types") or node_type not in data.node_types:
                raise RuntimeError(f"Node type '{node_type}' not found in OPF sample.")
            if not hasattr(data[node_type], "y") or data[node_type].y is None:
                raise RuntimeError(
                    f"No targets found for node type '{node_type}' in OPF sample."
                )
        pack_node_targets(data, self.node_target_types)
        return data

    def __getattr__(self, name):
        return getattr(self.base, name)


class NodeBatchAdapter:
    def __init__(self, loader, node_target_type, edge_dim: int):
        self.loader = loader
        self.node_target_types = (
            [node_target_type]
            if isinstance(node_target_type, str)
            else list(node_target_type)
        )
        self.edge_dim = edge_dim
        self.dataset = loader.dataset
        self.sampler = getattr(loader, "sampler", None)

    def __iter__(self):
        for data in self.loader:
            validate_edge_attr(data, self.edge_dim)
            if (
                not hasattr(data, "node_types")
                or any(name not in data.node_types for name in self.node_target_types)
            ):
                raise RuntimeError(
                    f"Node target types {self.node_target_types} not found in OPF sample."
                )

            if not hasattr(data, "batch"):
                node_store = data[self.node_target_types[0]]
                if hasattr(node_store, "batch"):
                    data.batch = node_store.batch
                elif (
                    hasattr(data, "batch_dict")
                    and self.node_target_types[0] in data.batch_dict
                ):
                    data.batch = data.batch_dict[self.node_target_types[0]]
                else:
                    raise RuntimeError(
                        f"Cannot find batch vector for node type "
                        f"'{self.node_target_types[0]}' in batched OPF data."
                    )

            if not hasattr(data, "y_loc") or data.y_loc.shape[1] != len(
                self.node_target_types
            ) + 1:
                raise RuntimeError(
                    "Batched node targets are missing a valid multi-head y_loc."
                )
            yield data

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)
