import os
import sys

import pytest
import torch
from torch_geometric.data import HeteroData


OPF_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "opf")
if OPF_DIR not in sys.path:
    sys.path.insert(0, OPF_DIR)

from opf_solution_utils import OPFDomainLoss  # noqa: E402


def _two_bus_graph():
    data = HeteroData()
    data["bus"].x = torch.tensor(
        [[100.0, 3.0, 0.9, 1.1], [100.0, 1.0, 0.9, 1.1]]
    )
    generator = torch.zeros((1, 11))
    generator[:, 2] = -1.0
    generator[:, 3] = 1.0
    generator[:, 5] = -1.0
    generator[:, 6] = 1.0
    data["generator"].x = generator
    data[("generator", "generator_link", "bus")].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )

    data[("bus", "ac_line", "bus")].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long
    )
    edge_attr = torch.zeros((1, 9))
    edge_attr[:, 0] = -0.5
    edge_attr[:, 1] = 0.5
    edge_attr[:, 4] = 0.0
    edge_attr[:, 5] = 0.1
    edge_attr[:, 6] = 10.0
    data[("bus", "ac_line", "bus")].edge_attr = edge_attr
    return data


def _call(loss, bus_prediction, generator_prediction, update_state=False):
    prediction = [bus_prediction, generator_prediction]
    value = torch.zeros(6)
    head_index = [torch.arange(4), torch.arange(4, 6)]
    return loss(
        prediction,
        value,
        head_index,
        _two_bus_graph(),
        update_state=update_state,
    )


@pytest.mark.mpi_skip()
def pytest_augmented_lagrangian_matches_family_mean_formula():
    loss = OPFDomainLoss(
        {
            "enabled": True,
            "mode": "augmented_lagrangian",
            "rho": 2.0,
            "constraints": ["voltage_bounds"],
            "monitor_all_constraints": False,
            "monitor_constraints": ["voltage_bounds"],
        },
        node_target_type=["bus", "generator"],
    )
    bus_prediction = torch.tensor(
        [[0.0, 0.5], [0.0, 1.5]], requires_grad=True
    )
    generator_prediction = torch.zeros((1, 2), requires_grad=True)

    penalty, metrics = _call(
        loss, bus_prediction, generator_prediction, update_state=True
    )
    # Positive h values are [0.4, 0, 0, 0.4]. With mu=0 and rho=2:
    # rho/2 * mean(relu(h)^2) = 1 * 0.08.
    assert torch.allclose(penalty, torch.tensor(0.08), atol=1e-7)
    assert torch.allclose(
        metrics["physics_voltage_bounds_mean_violation"],
        torch.tensor(0.2),
        atol=1e-7,
    )
    penalty.backward()
    assert bus_prediction.grad is not None

    updated = loss.update_duals()
    # mu <- mu + rho * mean(relu(h)) = 0 + 2 * 0.2.
    assert updated["voltage_bounds"] == pytest.approx(0.4)
    assert float(loss.mu_voltage_bounds) == pytest.approx(0.4)


@pytest.mark.mpi_skip()
def pytest_equality_dual_is_signed_and_updates_only_when_requested():
    loss = OPFDomainLoss(
        {
            "enabled": True,
            "mode": "augmented_lagrangian",
            "rho": 2.0,
            "constraints": ["power_balance_p"],
            "monitor_all_constraints": False,
            "monitor_constraints": ["power_balance_p"],
        },
        node_target_type=["bus", "generator"],
    )
    bus_prediction = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    generator_prediction = torch.tensor([[1.0, 0.0]])

    penalty, _ = _call(
        loss, bus_prediction, generator_prediction, update_state=False
    )
    # P residual is [1, 0]: rho/2 * mean(r^2) = 1 * 0.5.
    assert torch.allclose(penalty, torch.tensor(0.5), atol=1e-7)
    with pytest.raises(RuntimeError, match="No training residuals"):
        loss.update_duals()

    _call(loss, bus_prediction, generator_prediction, update_state=True)
    updated = loss.update_duals()
    assert updated["power_balance_p"] == pytest.approx(1.0)
    assert float(loss.lambda_power_balance_p) == pytest.approx(1.0)


@pytest.mark.mpi_skip()
def pytest_power_balance_uses_link_source_indices_not_edge_order():
    data = _two_bus_graph()
    data["generator"].x = torch.zeros((2, 11))
    data[("generator", "generator_link", "bus")].edge_index = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.long
    )
    data["load"].x = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    data[("load", "load_link", "bus")].edge_index = torch.tensor(
        [[1, 0], [1, 0]], dtype=torch.long
    )
    loss = OPFDomainLoss(
        {
            "enabled": True,
            "mode": "augmented_lagrangian",
            "rho": 2.0,
            "constraints": ["power_balance_p"],
            "monitor_all_constraints": False,
            "monitor_constraints": ["power_balance_p"],
        },
        node_target_type=["bus", "generator"],
    )
    bus_prediction = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    generator_prediction = torch.tensor([[1.0, 0.0], [2.0, 0.0]])

    penalty, metrics = loss(
        [bus_prediction, generator_prediction],
        torch.zeros(8),
        [torch.arange(4), torch.arange(4, 8)],
        data,
    )
    assert float(penalty) == pytest.approx(0.0, abs=1e-8)
    assert float(metrics["physics_power_balance_p_max_violation"]) == pytest.approx(
        0.0, abs=1e-8
    )


@pytest.mark.mpi_skip()
def pytest_duals_are_checkpointed_and_inequality_duals_are_nonnegative():
    config = {
        "enabled": True,
        "mode": "augmented_lagrangian",
        "rho": 1.0,
        "constraints": ["voltage_bounds"],
        "monitor_all_constraints": False,
        "monitor_constraints": ["voltage_bounds"],
        "initial_duals": {"voltage_bounds": -4.0},
    }
    first = OPFDomainLoss(config, node_target_type=["bus", "generator"])
    assert float(first.mu_voltage_bounds) == 0.0
    first.mu_voltage_bounds.fill_(3.25)

    second = OPFDomainLoss(config, node_target_type=["bus", "generator"])
    second.load_state_dict(first.state_dict())
    assert float(second.mu_voltage_bounds) == pytest.approx(3.25)


@pytest.mark.mpi_skip()
def pytest_omitted_mode_keeps_legacy_static_configuration():
    loss = OPFDomainLoss(
        {
            "enabled": True,
            "voltage_bound_weight": 0.1,
            "voltage_bound_feature_indices": [2, 3],
        },
        node_target_type="bus",
    )
    assert loss.mode == "static"
    assert loss.state_dict() == {}
