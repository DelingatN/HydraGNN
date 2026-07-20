##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import torch
import torch.nn.functional as F


def _normalize_edge_type(edge_type):
    if isinstance(edge_type, str):
        parts = edge_type.split("__")
        return tuple(parts) if len(parts) == 3 else edge_type
    if isinstance(edge_type, (list, tuple)):
        return tuple(edge_type)
    return edge_type


def _edge_type_tag(edge_type):
    if isinstance(edge_type, tuple):
        return "__".join(str(part) for part in edge_type)
    return str(edge_type)


def _mean_by_batch(x, batch):
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    sums = x.new_zeros((num_graphs, x.size(-1)))
    counts = x.new_zeros((num_graphs, 1))
    sums.index_add_(0, batch, x)
    counts.index_add_(
        0,
        batch,
        torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype),
    )
    return sums / counts.clamp_min(1.0)


@torch.no_grad()
def compute_hetero_oversmoothing_metrics(
    x_dict,
    edge_index_dict,
    batch_dict,
    layer_idx,
    node_types=None,
    edge_types=None,
    metrics=None,
):
    """Compute lightweight oversmoothing diagnostics for hetero hidden states."""

    if metrics is None:
        metrics = {
            "feature_variance",
            "mean_cos_to_centroid",
            "dirichlet_energy",
        }
    else:
        metrics = set(metrics)

    selected_node_types = set(str(node_type) for node_type in node_types or x_dict.keys())
    selected_edge_types = (
        set(_normalize_edge_type(edge_type) for edge_type in edge_types)
        if edge_types is not None
        else None
    )

    layer_tag = "layer_input" if layer_idx < 0 else f"layer_{layer_idx:02d}"
    out = {}

    for node_type, x in x_dict.items():
        node_type_str = str(node_type)
        if node_type_str not in selected_node_types or x.numel() == 0:
            continue

        x_float = x.detach().float()
        batch = batch_dict.get(node_type)
        if batch is None:
            batch = torch.zeros(x_float.size(0), dtype=torch.long, device=x_float.device)
        else:
            batch = batch.to(device=x_float.device, dtype=torch.long)

        centroid = _mean_by_batch(x_float, batch)[batch]
        centered = x_float - centroid
        prefix = f"{layer_tag}/{node_type_str}"

        if "feature_variance" in metrics:
            out[f"{prefix}/feature_variance"] = centered.pow(2).mean()

        if "mean_cos_to_centroid" in metrics:
            out[f"{prefix}/mean_cos_to_centroid"] = F.cosine_similarity(
                x_float,
                centroid,
                dim=-1,
                eps=1e-12,
            ).mean()

    if "dirichlet_energy" in metrics:
        for edge_type, edge_index in edge_index_dict.items():
            normalized_edge_type = _normalize_edge_type(edge_type)
            if (
                selected_edge_types is not None
                and normalized_edge_type not in selected_edge_types
            ):
                continue
            if (
                not isinstance(normalized_edge_type, tuple)
                or len(normalized_edge_type) != 3
            ):
                continue

            src_type, relation, dst_type = normalized_edge_type
            if src_type not in x_dict or dst_type not in x_dict or edge_index.numel() == 0:
                continue

            src = x_dict[src_type].detach().float()
            dst = x_dict[dst_type].detach().float()
            edge_index = edge_index.to(device=src.device, dtype=torch.long)
            diff = src[edge_index[0]] - dst[edge_index[1]]
            edge_tag = _edge_type_tag((src_type, relation, dst_type))
            out[f"{layer_tag}/{edge_tag}/dirichlet_energy"] = diff.pow(2).mean()

    return out
