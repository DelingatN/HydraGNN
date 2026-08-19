import os
import sys
from types import SimpleNamespace

import pytest
import torch
from mpi4py import MPI
from torch_geometric.data import Batch, HeteroData

import hydragnn.utils.datasets.hdf5dataset as hdf5dataset_module
import hydragnn.utils.datasets.pickledataset as pickledataset_module
from hydragnn.models.create import create_model
from hydragnn.train.train_validate_test import get_head_indices
from hydragnn.utils.datasets.hdf5dataset import HDF5Dataset, HDF5Writer
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleDataset,
    SimplePickleWriter,
)
from hydragnn.utils.model.model import update_multibranch_heads

OPF_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "opf")
if OPF_DIR not in sys.path:
    sys.path.insert(0, OPF_DIR)

from opf_positional_encodings import (  # noqa: E402
    OPFPositionalEncodingPreprocessor,
    build_topological_laplacian,
    compute_effective_resistance_pe,
    compute_topological_laplacian_pe,
    resolve_opf_positional_encoding_config,
)
from opf_solution_utils import pack_node_targets  # noqa: E402


def _path_graph():
    data = HeteroData()
    data["bus"].x = torch.zeros((3, 4))
    data[("bus", "ac_line", "bus")].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    edge_attr = torch.zeros((2, 9))
    edge_attr[:, 0] = -0.5
    edge_attr[:, 1] = 0.5
    edge_attr[:, 4] = 0.1
    edge_attr[:, 5] = 1.0
    edge_attr[:, 6] = 10.0
    data[("bus", "ac_line", "bus")].edge_attr = edge_attr
    return data


@pytest.mark.mpi_skip()
def pytest_opf_laplacian_pe_uses_smallest_nonzero_modes():
    data = _path_graph()
    artifact = compute_topological_laplacian_pe(
        data, k=2, compute_device="cpu"
    )
    eigenvectors = artifact["lap_eigvec"].double()
    eigenvalues = artifact["lap_eigval"].reshape(-1).double()
    laplacian = build_topological_laplacian(data)

    assert torch.allclose(
        eigenvalues, torch.tensor([1.0, 3.0], dtype=torch.float64), atol=1e-6
    )
    assert torch.allclose(
        laplacian @ eigenvectors,
        eigenvectors * eigenvalues.view(1, -1),
        atol=1e-6,
    )
    assert torch.allclose(
        eigenvectors.T @ eigenvectors, torch.eye(2, dtype=torch.float64), atol=1e-6
    )


@pytest.mark.mpi_skip()
def pytest_effective_resistance_five_stat_summary_excludes_diagonal():
    artifact = compute_effective_resistance_pe(
        _path_graph(), std_correction=0, compute_device="cpu"
    )
    expected = torch.tensor(
        [
            [1.0, 2.0, 0.5, 1.5, 1.5],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 0.5, 1.5, 1.5],
        ]
    )
    assert torch.allclose(artifact["effective_resistance_pe"], expected, atol=1e-5)


@pytest.mark.mpi_skip()
def pytest_opf_pe_preprocessor_caches_and_attaches_both_encodings(tmp_path):
    architecture = {
        "positional_encodings": {
            "precompute": ["laplacian", "effective_resistance"],
            "use": ["effective_resistance"],
            "cache_by_case": True,
            "compute_device": "cpu",
            "laplacian": {"dim": 2},
        }
    }
    resolved = resolve_opf_positional_encoding_config(architecture)
    assert resolved["precompute"] == ["laplacian", "effective_resistance"]
    assert resolved["use"] == ["effective_resistance"]

    preprocessor = OPFPositionalEncodingPreprocessor(
        architecture, cache_dir=str(tmp_path)
    )
    first = preprocessor(_path_graph(), case_name="three_bus")
    second = preprocessor(_path_graph(), case_name="three_bus")

    assert first["bus"].lap_eigvec.shape == (3, 2)
    assert first["bus"].lap_eigval.shape == (1, 2)
    assert first["bus"].effective_resistance_pe.shape == (3, 5)
    assert torch.equal(
        first["bus"].effective_resistance_pe,
        second["bus"].effective_resistance_pe,
    )
    assert len(list(tmp_path.glob("*.pt"))) == 2


@pytest.mark.mpi_skip()
def pytest_pe_cache_fingerprints_perturbed_topology_with_same_case_name(tmp_path):
    architecture = {
        "positional_encodings": {
            "precompute": ["laplacian"],
            "use": ["laplacian"],
            "cache_by_case": True,
            "compute_device": "cpu",
            "laplacian": {"dim": 2},
        }
    }
    preprocessor = OPFPositionalEncodingPreprocessor(
        architecture, cache_dir=str(tmp_path)
    )
    original = preprocessor(_path_graph(), case_name="three_bus")
    perturbed = _path_graph()
    perturbed[("bus", "ac_line", "bus")].edge_index = torch.tensor(
        [[0, 0], [1, 2]], dtype=torch.long
    )
    perturbed = preprocessor(perturbed, case_name="three_bus")

    assert not torch.equal(
        original["bus"].lap_eigvec, perturbed["bus"].lap_eigvec
    )
    assert len(list(tmp_path.glob("*.pt"))) == 2


@pytest.mark.mpi_skip()
def pytest_active_pe_must_be_precomputed():
    with pytest.raises(ValueError, match="must also be precomputed"):
        resolve_opf_positional_encoding_config(
            {
                "positional_encodings": {
                    "precompute": ["laplacian"],
                    "use": ["effective_resistance"],
                }
            }
        )


@pytest.mark.mpi_skip()
def pytest_joint_node_targets_pack_head_by_head_across_a_batch():
    samples = []
    for num_bus, num_generator in ((3, 1), (4, 2)):
        data = HeteroData()
        data["bus"].x = torch.zeros((num_bus, 4))
        data["bus"].y = torch.arange(num_bus * 2).view(num_bus, 2).float()
        data["generator"].x = torch.zeros((num_generator, 11))
        data["generator"].y = (
            100.0 + torch.arange(num_generator * 2).view(num_generator, 2)
        )
        pack_node_targets(data, ["bus", "generator"])
        samples.append(data)

    batch = Batch.from_data_list(samples)
    batch.batch = batch["bus"].batch
    dummy_model = SimpleNamespace(
        module=SimpleNamespace(
            num_heads=2,
            head_type=["node", "node"],
        )
    )
    head_index = get_head_indices(dummy_model, batch)

    expected_bus = torch.cat([sample["bus"].y for sample in samples])
    expected_generator = torch.cat([sample["generator"].y for sample in samples])
    assert torch.equal(batch.y[head_index[0]].reshape_as(expected_bus), expected_bus)
    assert torch.equal(
        batch.y[head_index[1]].reshape_as(expected_generator), expected_generator
    )


@pytest.mark.mpi_skip()
def pytest_joint_hgt_gps_uses_laplacian_and_effective_resistance_pe():
    data = HeteroData()
    data["bus"].x = torch.randn(3, 4)
    data["generator"].x = torch.randn(1, 11)
    data["load"].x = torch.randn(1, 2)
    data["shunt"].x = torch.randn(1, 2)
    data["bus"].lap_eigvec = torch.randn(3, 2)
    data["bus"].lap_eigval = torch.tensor([[1.0, 3.0]])
    data["bus"].effective_resistance_pe = torch.randn(3, 5)

    data[("bus", "ac_line", "bus")].edge_index = torch.tensor(
        [[0, 1], [1, 2]]
    )
    data[("bus", "ac_line", "bus")].edge_attr = torch.randn(2, 9)
    data[("bus", "transformer", "bus")].edge_index = torch.tensor([[2], [0]])
    data[("bus", "transformer", "bus")].edge_attr = torch.randn(1, 11)
    for source, relation, destination, edge_index in (
        ("generator", "generator_link", "bus", [[0], [0]]),
        ("bus", "generator_link", "generator", [[0], [0]]),
        ("load", "load_link", "bus", [[0], [1]]),
        ("bus", "load_link", "load", [[1], [0]]),
        ("shunt", "shunt_link", "bus", [[0], [2]]),
        ("bus", "shunt_link", "shunt", [[2], [0]]),
    ):
        data[(source, relation, destination)].edge_index = torch.tensor(edge_index)

    output_heads = update_multibranch_heads(
        {
            "node": {
                "num_headlayers": 2,
                "dim_headlayers": [16, 8],
                "type": "mlp",
            }
        }
    )
    model = create_model(
        mpnn_type="HeteroHGT",
        input_dim=4,
        hidden_dim=16,
        output_dim=[2, 2],
        pe_dim=0,
        global_attn_engine="GPS",
        global_attn_type="multihead",
        global_attn_heads=4,
        output_type=["node", "node"],
        output_heads=output_heads,
        activation_function="relu",
        loss_function_type="mse",
        task_weights=[1.0, 1.0],
        num_conv_layers=2,
        equivariance=False,
        node_target_type=["bus", "generator"],
        metadata=data.metadata(),
        node_input_dims={"bus": 4, "generator": 11, "load": 2, "shunt": 2},
        hetero_attention_heads=4,
        positional_encodings={
            "use": ["laplacian", "effective_resistance"],
            "laplacian": {"dim": 2, "random_sign_flip": True},
            "effective_resistance": {
                "statistics": ["min", "max", "std", "median", "mean"]
            },
        },
    )
    model.eval()
    output = model(data)
    assert output[0].shape == (3, 2)
    assert output[1].shape == (1, 2)


@pytest.mark.mpi_skip()
def pytest_pe_tensors_round_trip_through_pickle_and_hdf5(tmp_path, monkeypatch):
    # Both writers use a progress helper that normally expects the training
    # process group to exist.  Serialization itself only needs COMM_SELF here.
    def no_progress(iterator, *_args, **_kwargs):
        return iterator

    monkeypatch.setattr(pickledataset_module, "iterate_tqdm", no_progress)
    monkeypatch.setattr(hdf5dataset_module, "iterate_tqdm", no_progress)
    architecture = {
        "positional_encodings": {
            "precompute": ["laplacian", "effective_resistance"],
            "use": ["effective_resistance"],
            "compute_device": "cpu",
            "laplacian": {"dim": 2},
        }
    }
    data = OPFPositionalEncodingPreprocessor(architecture)(
        _path_graph(), case_name="three_bus"
    )

    pickle_dir = tmp_path / "pickle"
    SimplePickleWriter(
        [data], str(pickle_dir), label="trainset", comm=MPI.COMM_SELF
    )
    pickle_data = SimplePickleDataset(
        str(pickle_dir), "trainset", var_config=None
    )[0]

    hdf5_dir = tmp_path / "hdf5"
    hdf5_writer = HDF5Writer(str(hdf5_dir), comm=MPI.COMM_SELF)
    hdf5_writer.add("trainset", [data])
    hdf5_writer.save()
    hdf5_data = HDF5Dataset(str(hdf5_dir), "trainset")[0]

    for restored in (pickle_data, hdf5_data):
        assert torch.equal(restored["bus"].lap_eigvec, data["bus"].lap_eigvec)
        assert torch.equal(restored["bus"].lap_eigval, data["bus"].lap_eigval)
        assert torch.equal(
            restored["bus"].effective_resistance_pe,
            data["bus"].effective_resistance_pe,
        )
