"""
coarsening_eigen_experiment.py
================================

Graph-hashing / coarsening experiment driver  (MULTI-LEVEL version).

For a given dataset it builds the hashing feature matrix `data.z` in three
different ways (modes) and under three different homophily-coefficient schemes,
runs the hashing -> binary-search-cutoff -> partition -> coarsening pipeline for
every (mode x scheme) combination, and records the RELATIVE EIGEN ERROR of the
coarsened graph. All results are written to an .xlsx file.

NEW: the coarsening is now done in *stages* (a cascade of coarsening levels).

  * Level 1 hashes the original homophily-mixed feature matrix `z`, exactly as
    before.
  * Each later level REBUILDS its hashing matrix the same way `Z_o` was built:
    it takes the coarsened feature `z_c` (the normalized aggregate `P^T z` of
    the member features -- "the normalized feature vector of all the nodes
    falling into that supernode"), the coarsened adjacency `A_c` obtained from
    the previous level, and `A_c @ z_c`, and mixes them with the SAME mode and
    the SAME homophily coefficients (cX, cA, cAx) via build_z. So every level
    runs build_z, not just the first.
  * The coarsened adjacency carries a size-aware diagonal "mass" equal to how
    many ORIGINAL nodes fall into each supernode, so the strength of connection
    between supernodes grows with their membership across levels. At level 1
    this mass equals the old immediate-member count, so the level-1 output is
    identical to the original single-level code.

You give two inputs -- the number of stages (`--stages`) and the FINAL overall
reduction ratio (`--cr`, where cr=0.5 means the final graph keeps N/2 nodes) --
and the code decides the in-between cumulative targets in DECREASING order
automatically (linear spacing by default). If you want to control the schedule
yourself, pass `--stage_fractions` (a decreasing list of cumulative
fractions-of-N remaining per stage), e.g. `--stage_fractions 0.9,0.5`.

bucketDistType is fixed to "L2" everywhere (as requested).

Run it like:

    # auto schedule: 2 stages down to N/2  (-> 75% then 50% of N)
    python coarsening_eigen_experiment.py --dataname Cora --stages 2 --cr 0.5

    # your exact example: first level to 90% of N, then to 50% of N
    python coarsening_eigen_experiment.py --dataname Cora --stage_fractions 0.9,0.5

    # original single-level behaviour
    python coarsening_eigen_experiment.py --dataname Cora --stages 1 --cr 0.5

Supported datasets (case-insensitive):
    cora, citeseer, pubmed, squirrel, chameleon, texas, computers, cs

Requires your local `utils` and `spectral_properties` modules to be importable
(same directory / PYTHONPATH), exactly like the original script.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    WikipediaNetwork,
    WebKB,
)

import utils                 # your local module (used in build_coarsened_graph)
import spectral_properties   # your local module (eigen_error)

import warnings
warnings.simplefilter("ignore")


# ----------------------------------------------------------------------------- #
#  Dataset loading
# ----------------------------------------------------------------------------- #

# name (lower-case)  ->  (dataset_class, name_arg_passed_to_the_class)
DATASET_REGISTRY = {
    "cora":      (Planetoid,         "Cora"),
    "citeseer":  (Planetoid,         "Citeseer"),
    "pubmed":    (Planetoid,         "Pubmed"),
    "squirrel":  (WikipediaNetwork,  "squirrel"),
    "chameleon": (WikipediaNetwork,  "chameleon"),
    "texas":     (WebKB,             "Texas"),
    "computers": (Amazon,            "Computers"),
    "cs":        (Coauthor,          "CS"),
}


def make_random_split(data, train_ratio=0.6, val_ratio=0.2, seed=0):
    """Create boolean train/val/test masks with a fresh random split.

    We always regenerate the split (instead of relying on dataset-provided
    masks) so that every dataset is treated identically -- this mirrors what
    the original script did for Cora, and avoids the multi-column (N x 10)
    masks that ship with WikipediaNetwork / WebKB.
    """
    g = torch.Generator().manual_seed(seed)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes, generator=g)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    return data


def load_dataset(name, root, seed=0):
    """Load a supported dataset by name and return (data, num_features, num_classes).

    A fresh random 60/20/20 split is attached. Everything is kept on CPU; the
    heavy coarsening / numpy steps assume CPU tensors.
    """
    key = name.strip().lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(DATASET_REGISTRY)}"
        )

    cls, name_arg = DATASET_REGISTRY[key]
    # PyG dataset classes already create their own per-name subfolder under
    # `root` (e.g. <root>/Cora/raw), so we pass `root` directly -- this matches
    # the original `Planetoid(root=dataset_path, name="Cora")` layout and reuses
    # any data you already downloaded.
    ds_root = root

    if cls is WikipediaNetwork:
        # geom_gcn_preprocess=True gives the standard cleaned features/labels
        dataset = cls(root=ds_root, name=name_arg, geom_gcn_preprocess=True)
    else:
        dataset = cls(root=ds_root, name=name_arg)

    data = dataset[0]

    # num_classes / num_features (fall back to deriving them if needed)
    try:
        num_classes = int(dataset.num_classes)
    except Exception:
        num_classes = int(data.y.max().item()) + 1
    try:
        num_features = int(dataset.num_features)
    except Exception:
        num_features = int(data.x.shape[1])

    data = make_random_split(data, seed=seed)

    print(f"\nLoaded dataset: {name}")
    print("-------------------")
    print("Number of nodes    :", data.num_nodes)
    print("Number of edges    :", data.num_edges)
    print("Number of features :", num_features)
    print("Number of classes  :", num_classes)
    print("Train / Val / Test :",
          int(data.train_mask.sum()),
          int(data.val_mask.sum()),
          int(data.test_mask.sum()))

    return data, num_features, num_classes


# ----------------------------------------------------------------------------- #
#  Core helpers (kept faithful to the original script)
# ----------------------------------------------------------------------------- #

def compute_homophily(data):
    edge_index = data.edge_index
    labels = data.y
    src, dst = edge_index[0], edge_index[1]
    same_class = (labels[src] == labels[dst]).sum().item()
    total_edges = edge_index.shape[1]
    return same_class / total_edges


def make_sparse_jl_left(out_dim, in_dim, s, device, dtype):
    cols = torch.arange(in_dim, device=device)
    h = torch.randint(low=0, high=out_dim, size=(s, in_dim), device=device)
    g = (torch.randint(0, 2, size=(s, in_dim), device=device) * 2 - 1).to(dtype) / (s ** 0.5)
    row_idx = h.reshape(-1)
    col_idx = cols.repeat(s)
    vals = g.reshape(-1)
    indices = torch.stack([row_idx, col_idx], dim=0)
    J = torch.sparse_coo_tensor(indices, vals, (out_dim, in_dim),
                                device=device, dtype=dtype).coalesce()
    return J


def jl_project_A_stream(A, d_proj, s=1):
    device = A.device
    if A.is_sparse:
        A = A.coalesce()
        row, col = A.indices()
        val = A.values()
        N = A.size(0)
        A_proj = torch.zeros((N, d_proj), device=device, dtype=val.dtype)
        h_all = torch.randint(0, d_proj, size=(s, N), device=device)
        g_all = (torch.randint(0, 2, size=(s, N), device=device) * 2 - 1).to(val.dtype) / (s ** 0.5)
        for k in range(s):
            tgt = h_all[k, col]
            w = g_all[k, col] * val
            A_proj.index_put_((row, tgt), w, accumulate=True)
        return A_proj
    else:
        N = A.size(0)
        H_left = make_sparse_jl_left(d_proj, N, s, device=A.device, dtype=A.dtype)
        A_proj_T = H_left @ A.T
        return A_proj_T.T


def standardize_tensor(tensor):
    mean_val = torch.mean(tensor, dim=0)
    std_val = torch.std(tensor, dim=0)
    epsilon = 1e-8
    return (tensor - mean_val) / (std_val + epsilon)


def compute_bin_values(Z, no_of_hash, hash_norm="L2-norm",
                       projectors_distribution="normal", device="cpu"):
    """Hash the feature matrix Z into Bin_values via random projectors.

    Returns Bin_values on `device` (shape: num_nodes x no_of_hash).
    """
    feature_size = Z.shape[-1]
    Z = Z.to(device)

    if projectors_distribution == "normal":
        Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0, 1).to(device)
    elif projectors_distribution == "uniform":
        Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0, 1).to(device)
    elif projectors_distribution == "VAEs":
        learned_mean, learned_sigma = -0.0017, 0.29
        Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(learned_mean, learned_sigma).to(device)
    else:
        Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0, 1).to(device)

    if hash_norm == "L2-norm":
        Bin_values = torch.cdist(Z, Wl, p=2)
    elif hash_norm == "L1-norm":
        Bin_values = torch.cdist(Z, Wl, p=1)
    else:  # "dot" (dot-product / matmul projection)
        Bin_values = torch.matmul(Z, Wl.T)
    return Bin_values


def partition(Bin_values, cutoffBuckDist=1.0, bucketDistType="L2"):
    """Density-rank partition. Device-aware (works whether Bin_values is on
    CPU or GPU). Returns a dict {node_idx -> supernode_id}.
    """
    n = len(Bin_values)
    feat_dim = Bin_values.shape[-1]
    device = Bin_values.device

    if bucketDistType == "L1":
        D = torch.cdist(Bin_values, Bin_values, p=1) / feat_dim
    else:  # L2
        D = torch.cdist(Bin_values, Bin_values, p=2) / feat_dim

    neighbor_mask = D < cutoffBuckDist          # (N, N) bool
    density = neighbor_mask.sum(dim=1)          # (N,)

    clusterAssigned = -1 * torch.ones(n, device=device)
    rank = -1 * torch.ones(n, device=device)
    currClusterIdx = 0

    for i in range(n):
        indicesBuckDist = neighbor_mask[i]
        rankCurrCluster = density[i]
        indicesRankCluster = (rank < rankCurrCluster)

        updateIndices = indicesBuckDist & indicesRankCluster
        orgRank = torch.sum(updateIndices)

        clusterAssigned[updateIndices] = currClusterIdx
        currClusterIdx += 1
        rank[updateIndices] = orgRank.float()

    return {idx: int(v.item()) for idx, v in enumerate(clusterAssigned)}


def reduction_ratio(summary):
    values = summary.values()
    return 1 - len(set(values)) / len(values)


def binarySearchCutoff(minVal, maxVal, cr, epsilon, bucketDistType, Bin_values,
                       verbose=False):
    """Bisection search for the cutoff that yields reduction ratio ~ cr.

    Tolerance-based: loops until the achieved reduction ratio is within
    `epsilon` of the target `cr`.
    """
    rr = 0
    mid = (minVal + maxVal) / 2
    current_bin_width_summary = None
    while abs(rr - cr) > epsilon:
        if verbose:
            print("Finding avgBucketDist....")
        mid = (minVal + maxVal) / 2
        current_bin_width_summary = partition(
            Bin_values, cutoffBuckDist=mid, bucketDistType=bucketDistType
        )
        values = current_bin_width_summary.values()
        unique_values = set(values)
        rr = 1 - len(unique_values) / len(values)

        if rr > cr:
            maxVal = mid
        else:
            minVal = mid
    return mid, current_bin_width_summary


def get_key(val, g_coarsened):
    keys = [k for k, v in g_coarsened.items() if v == val]
    return len(keys), keys


def build_coarsened_graph(current_bin_width_summary, data, num_classes,
                          random_coarsening=False):
    """Build a coarsened PyG graph from the partition output.

    NOTE: This is the ORIGINAL single-level routine. It is kept here unchanged
    for reference / re-use by your GCN training script. The multi-level eigen
    experiment below uses `coarsen_one_level` instead (which is equivalent to
    this at level 1, plus weighted-adjacency propagation across levels).

    Returns: data_coarsen, P, P_hat, zero_list, rr
    (uses data.z for features).
    """
    values = current_bin_width_summary.values()
    unique_values = set(values)
    num_supernodes = len(unique_values)
    rr = 1 - num_supernodes / len(values)

    print(f"  Graph reduced by: {rr * 100:.2f}%  "
          f"(supernodes={num_supernodes}, original={len(values)})")

    # Step 1: supernode sizes + member lists
    C_diag = torch.zeros(num_supernodes)
    dict_blabla = {}
    for idx, supernode_id in enumerate(unique_values):
        size, members = get_key(supernode_id, current_bin_width_summary)
        C_diag[idx] = size
        dict_blabla[idx] = members

    # Step 2: P_hat (N x M binary assignment)
    P_hat = torch.zeros((data.num_nodes, num_supernodes))
    zero_list = torch.ones(num_supernodes, dtype=torch.bool)  # True = no train node inside

    if not random_coarsening:
        for supernode_idx, members in dict_blabla.items():
            for node_idx in members:
                P_hat[node_idx, supernode_idx] = 1
                if data.train_mask[node_idx]:
                    zero_list[supernode_idx] = False
    else:
        import random as _random
        for supernode_idx in dict_blabla:
            sampled = _random.sample(range(data.num_nodes), 1)
            for node_idx in sampled:
                P_hat[node_idx, supernode_idx] = 1
                if data.train_mask[node_idx]:
                    zero_list[supernode_idx] = False

    P_hat = P_hat.to_sparse()

    # Step 3: normalized assignment matrix P = P_hat * C^{-1/2}
    P = torch.sparse.mm(P_hat, torch.diag(torch.pow(C_diag, -0.5)))

    # Step 4: coarsened features X_c = P^T * Z
    cor_feat = torch.sparse.mm(P.t(), data.z)

    # Step 5: coarsened adjacency A_c = P_hat^T * A * P_hat
    i = data.edge_index
    v = torch.ones(data.edge_index.shape[1])
    shape = torch.Size([data.num_nodes, data.num_nodes])
    A_sparse = torch.sparse_coo_tensor(i, v, shape)
    g_coarse_adj = torch.sparse.mm(P_hat.t(), torch.sparse.mm(A_sparse, P_hat))

    C_diag_matrix = np.diag(C_diag.numpy().astype(np.float32))
    g_coarse_dense = (
        g_coarse_adj.to_dense().numpy()
        + C_diag_matrix
        - np.eye(num_supernodes, dtype=np.float32)
    )

    # Step 6: edge index + weights
    nz_rows, nz_cols = np.nonzero(g_coarse_dense)
    edge_weight = g_coarse_dense[nz_rows, nz_cols]
    edge_index_coarse = torch.stack([
        torch.from_numpy(nz_rows),
        torch.from_numpy(nz_cols),
    ])
    edge_features = torch.from_numpy(edge_weight)

    # Step 7: coarsened labels
    Y = np.array(data.y.cpu())
    Y = utils.one_hot(Y, num_classes)
    Y[~data.train_mask] = torch.tensor([0.0] * num_classes)
    labels_coarse = torch.argmax(
        torch.sparse.mm(P.t().double(), Y.double()), dim=1
    )

    data_coarsen = Data(x=cor_feat, edge_index=edge_index_coarse, y=labels_coarse)
    data_coarsen.edge_attr = edge_features

    return data_coarsen, P, P_hat, zero_list, rr


# ----------------------------------------------------------------------------- #
#  Multi-level coarsening
# ----------------------------------------------------------------------------- #

def compute_stage_targets(stages, final_rr, stage_fractions=None):
    """Decide the per-stage coarsening schedule from two inputs.

    Parameters
    ----------
    stages : int
        Number of successive coarsening levels.
    final_rr : float
        Overall target reduction ratio of the FINAL graph relative to the
        ORIGINAL graph. e.g. final_rr=0.5 -> final graph keeps 50% of the
        nodes (N -> N/2).
    stage_fractions : list[float] | None
        Optional explicit list of CUMULATIVE fractions-of-N REMAINING after
        each stage, strictly decreasing, e.g. [0.9, 0.5]. If given it overrides
        the automatic schedule (and defines `stages` and `final_rr`).

    Returns
    -------
    cum_fractions : list[float]
        Cumulative fraction-of-N remaining target after each stage (decreasing).
    per_stage_rr : list[float]
        Reduction ratio to REQUEST at each stage, relative to that stage's
        INPUT graph, so that the cumulative targets are met.
    """
    if stage_fractions is not None:
        cum = [float(x) for x in stage_fractions]
        stages = len(cum)
    else:
        # Automatic schedule: spread the fraction-remaining linearly from 1.0
        # down to (1 - final_rr) across `stages` levels  -> decreasing order.
        cum = [1.0 - (i + 1) * final_rr / stages for i in range(stages)]

    # sanity: strictly decreasing starting from the full graph (1.0)
    prev = 1.0
    for f in cum:
        if not (0.0 < f < prev):
            raise ValueError(
                f"stage targets must be strictly decreasing in (0, 1): got {cum}"
            )
        prev = f

    # cumulative fractions -> per-stage reduction ratios (relative to the input
    # graph that each stage actually receives)
    per_stage_rr = []
    prev = 1.0
    for f in cum:
        per_stage_rr.append(1.0 - f / prev)
        prev = f

    return cum, per_stage_rr


def coarsen_one_level(z, edge_index, edge_weight, node_weight, num_nodes,
                      target_rr, args, device, mass_mode="accumulate"):
    """Apply ONE hashing-based coarsening level to the current graph.

    The current graph is described by:
      z           : (n, d) float feature matrix of the current nodes
      edge_index  : (2, E) long
      edge_weight : (E,) float, or None for an unweighted graph (all ones)
      node_weight : (n,) float, number of ORIGINAL nodes each current node
                    represents (all ones at the very first level)
      num_nodes   : n
      target_rr   : reduction ratio to hit at THIS level, relative to n

    `mass_mode` controls how the size-aware diagonal is carried forward:
      "accumulate" : the full coarsened adjacency (off-diagonal + diagonal
                     mass) is fed into the next level. This literally "uses the
                     adjacency matrix obtained through the previous level", so
                     the diagonal mass accumulates as levels deepen.
      "fresh"      : only the genuine inter-supernode connections are fed
                     forward; the size-aware diagonal mass is recomputed from
                     the cumulative node counts at each level (no compounding).

    Returns
    -------
    z_c            : (m, d) coarsened (normalized-aggregate) features
    ei_eig, ew_eig : coarsened adjacency WITH the size-aware diagonal mass
                     (this is what feeds spectral_properties.eigen_error if this
                     turns out to be the last level)
    ei_prop, ew_prop : adjacency that feeds the NEXT level (depends on mass_mode)
    node_weight_c  : (m,) cumulative original-node count per supernode
    m              : number of supernodes
    achieved_rr    : reduction ratio actually reached this level
    cutoff         : the bucket-distance cutoff that produced it
    """
    n = num_nodes

    # ---- 1. hash the current features ---------------------------------------
    Bin_values = compute_bin_values(
        z, no_of_hash=args.no_of_hash, hash_norm=args.hash_norm,
        projectors_distribution=args.projectors, device=device,
    )

    maxVal = 2 * torch.max(
        torch.sqrt(torch.sum(torch.pow(Bin_values, 2), axis=1)) / Bin_values.shape[-1]
    ).item()

    # ---- 2. cutoff search to reach target_rr at this level ------------------
    cutoff, summary = binarySearchCutoff(
        0.0, maxVal, target_rr, args.epsilon, "L2", Bin_values,
        verbose=args.verbose,
    )

    values = summary.values()
    unique_values = set(values)
    m = len(unique_values)
    achieved_rr = 1 - m / len(values)

    # ---- 3. assignment matrix P_hat (n x m) + member bookkeeping ------------
    C_diag = torch.zeros(m)             # immediate members per supernode
    node_weight_c = torch.zeros(m)      # cumulative original nodes per supernode
    members_of = {}
    for new_idx, supernode_id in enumerate(unique_values):
        size, members = get_key(supernode_id, summary)
        C_diag[new_idx] = size
        members_of[new_idx] = members
        idx_t = torch.tensor(members, dtype=torch.long)
        node_weight_c[new_idx] = float(node_weight[idx_t].sum().item())

    P_hat = torch.zeros((n, m))
    for new_idx, members in members_of.items():
        for node_idx in members:
            P_hat[node_idx, new_idx] = 1.0
    P_hat = P_hat.to_sparse()

    # normalized assignment P = P_hat * C^{-1/2}  (immediate-size normalization,
    # exactly as in the original single-level code)
    P = torch.sparse.mm(P_hat, torch.diag(torch.pow(C_diag, -0.5)))

    # ---- 4. coarsened features: normalized aggregate of member features -----
    #         ("the normalized feature vector of all the nodes falling into
    #          that supernode" -- same construction as the original code)
    z_c = torch.sparse.mm(P.t(), z)

    # ---- 5. coarsened WEIGHTED adjacency ------------------------------------
    if edge_weight is None:
        v = torch.ones(edge_index.shape[1], dtype=torch.float32)
    else:
        v = edge_weight.float()
    A_sparse = torch.sparse_coo_tensor(
        edge_index, v, torch.Size([n, n])
    )
    # genuine inter-/intra-supernode connection strength (off-diagonal carries
    # the accumulated edge weights -> grows with how many member nodes connect)
    g_off = torch.sparse.mm(P_hat.t(), torch.sparse.mm(A_sparse, P_hat)) \
                 .to_dense().numpy().astype(np.float32)

    # size-aware diagonal "mass": how many ORIGINAL nodes fall into each
    # supernode. At level 1 this equals C_diag, so the level-1 output matches
    # the original build_coarsened_graph exactly.
    mass = np.diag(node_weight_c.numpy().astype(np.float32))
    g_eig = g_off + mass - np.eye(m, dtype=np.float32)

    # adjacency that feeds the next level
    g_prop = g_eig if mass_mode == "accumulate" else g_off

    def _to_edges(g_dense):
        nz_rows, nz_cols = np.nonzero(g_dense)
        w = g_dense[nz_rows, nz_cols]
        ei = torch.stack([torch.from_numpy(nz_rows), torch.from_numpy(nz_cols)])
        ew = torch.from_numpy(w)
        return ei, ew

    ei_eig, ew_eig = _to_edges(g_eig)
    if mass_mode == "accumulate":
        ei_prop, ew_prop = ei_eig, ew_eig
    else:
        ei_prop, ew_prop = _to_edges(g_prop)

    # dense coarsened adjacency that the NEXT level will treat as its "A_c"
    # (used to rebuild that level's hashing matrix via build_z). This is the
    # same adjacency that feeds the next coarsening step (g_prop), so the
    # feature mixing and the coarsening see a consistent graph.
    A_prop_dense = torch.from_numpy(g_prop)

    # z_c here is the coarsened FEATURE ("normalized feature vector of all the
    # nodes falling into that supernode", i.e. P^T z). It plays the role of X
    # in the next level's build_z, NOT the next level's hashing matrix.
    return (z_c, A_prop_dense, ei_eig, ew_eig, ei_prop, ew_prop,
            node_weight_c, m, achieved_rr, cutoff)


# ----------------------------------------------------------------------------- #
#  Feature construction: modes + coefficient schemes
# ----------------------------------------------------------------------------- #

# Three ways of building data.z
# MODES = ["x_aproj", "x_a", "x_aproj_ax"]
MODES = ["x_aproj_ax"]

MODE_DESC = {
    "x_aproj":    "concat(X, A_proj)",
    "x_a":        "concat(X, raw A)",
    "x_aproj_ax": "concat(X, A_proj, Ax=A.X)",
}

# Three homophily-coefficient schemes.  h = homophily.
# Each returns (coeff_X, coeff_A, coeff_Ax).
SCHEMES = {
    # original: X*(1-h), adjacency*h, Ax*h
    "hf_adj":          lambda h: (1 - h, h,     h),
    # h on adjacency, (1-h) on Ax
    "hf_adj_1mhf_ax":  lambda h: (1 - h, h,     1 - h),
    # vice versa: (1-h) on adjacency, h on Ax
    "1mhf_adj_hf_ax":  lambda h: (1 - h, 1 - h, h),
}


def build_components_from(X_feat, A_dense, aproj_dim, s=1, project_adj=True,
                          normalize=True):
    """Build the hashing components (X, A_proj, A_raw, hop=A.X) for an ARBITRARY
    feature matrix and dense adjacency.

    This is the single source of truth for "how the building blocks are made",
    used both for the original graph (level 1) and for every coarsened graph
    (level >= 2). For a coarsened graph, `X_feat` is the coarsened feature
    (P^T z, "the normalized feature vector of all the nodes falling into that
    supernode") and `A_dense` is the coarsened adjacency obtained from the
    previous level, so `hop = A_c @ X_c` is exactly the "A_c Z_c" term.

    `project_adj`:
      True  (level 1)  -> the huge N-dim ORIGINAL adjacency is compressed with a
                          random JL projection to `aproj_dim`.
      False (level>=2) -> the coarsened adjacency A_c is used DIRECTLY (no JL
                          re-projection); its entries are the current supernode
                          connection strengths.

    `normalize`:
      True  (level 1)  -> row-wise L2 normalize every block. The raw inputs
                          (data.x, raw A) need it, and this is their SINGLE
                          normalization.
      False (level>=2) -> NO second normalization. The coarsened feature z_c is
                          already the normalized aggregate (P^T z carries the
                          C^{-1/2} size normalization), and A_c / A_c@z_c are
                          used AS PRODUCED -- which also keeps A_c's real
                          supernode connection-strength magnitudes intact. So
                          every quantity is normalized exactly once in its
                          lineage, never twice.
    """
    if project_adj:
        # JL-compress the original adjacency (level 1 only).
        A_proj = jl_project_A_stream(A_dense, aproj_dim, s=s)
    else:
        # use the coarsened adjacency itself -- its weights ARE the current
        # supernode connection strengths; do NOT JL-project it again.
        A_proj = A_dense
    hop = A_dense @ X_feat  # one-hop aggregation (Ax  /  A_c X_c)

    if normalize:
        # ONE normalization, applied to the raw level-1 inputs.
        return {
            "X":      F.normalize(X_feat, p=2, dim=1, eps=1e-12),
            "A_proj": F.normalize(A_proj, p=2, dim=1, eps=1e-12),
            "A_raw":  F.normalize(A_dense, p=2, dim=1, eps=1e-12),
            "hop":    F.normalize(hop, p=2, dim=1, eps=1e-12),
        }
    # level >= 2: components are already normalized at production (z_c via the
    # coarsening aggregation) or are intentionally left raw (A_c connection
    # strengths). No second normalization is applied.
    return {
        "X":      X_feat,
        "A_proj": A_proj,
        "A_raw":  A_dense,
        "hop":    hop,
    }


def precompute_components(data, aproj_dim, s=1):
    """Compute and normalize the level-1 building blocks once per dataset.

    Returns a dict with normalized X, A_proj, A_raw, hop (=A.X). All on CPU.
    The original adjacency IS JL-projected here, and the raw inputs get their
    single row-wise L2 normalization (project_adj=True, normalize=True).
    """
    N = data.num_nodes
    A = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0)
    return build_components_from(data.x, A, aproj_dim, s=s,
                                 project_adj=True, normalize=True)


def build_z(comps, mode, coeffs):
    """Assemble data.z for a given mode and coefficient tuple (cX, cA, cAx)."""
    cX, cA, cAx = coeffs
    if mode == "x_aproj":
        parts = [cX * comps["X"], cA * comps["A_proj"]]
    elif mode == "x_a":
        parts = [cX * comps["X"], cA * comps["A_raw"]]
    elif mode == "x_aproj_ax":
        parts = [cX * comps["X"], cA * comps["A_proj"], cAx * comps["hop"]]
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return torch.cat(parts, dim=1)


# ----------------------------------------------------------------------------- #
#  Single run of the full (multi-level) pipeline for one (mode, scheme)
# ----------------------------------------------------------------------------- #

def run_single(data, comps, num_classes, mode, scheme_name, h_f, args, device):
    """Run the multi-level cascade for one (mode, scheme):

        build_z -> hash -> cutoff -> partition -> coarsen   (x `args.stages`)

    At level 1 build_z mixes the original X / A_proj / A.X. At every later level
    the building blocks are rebuilt from the coarsened feature (z_c = P^T z),
    the coarsened adjacency A_c, and A_c @ z_c, then mixed by build_z with the
    SAME coefficients -- so the same recipe is applied to the coarsened graph.

    Finally the relative eigen error of the FINAL coarsened graph is measured
    against the ORIGINAL graph. Returns a results dict.
    """
    coeffs = SCHEMES[scheme_name](h_f)
    cX, cA, cAx = coeffs

    # Decide the multilevel schedule from (stages, final reduction ratio) or
    # from an explicit list of cumulative fractions.
    stage_fractions = getattr(args, "stage_fractions_parsed", None)
    cum_fractions, per_stage_rr = compute_stage_targets(
        args.stages, args.cr, stage_fractions
    )

    # ----- run the cascade -----
    # `comps_cur` holds the (X, A_proj, A_raw, hop) building blocks for the graph
    # the current level operates on. At level 1 these come from the ORIGINAL
    # graph; at every later level they are REBUILT from the coarsened feature
    # (X_c = P^T z) and the coarsened adjacency A_c via build_components_from,
    # then mixed by build_z with the SAME (cX, cA, cAx) coefficients and mode.
    comps_cur = comps
    ei = data.edge_index          # current adjacency feeding the next level
    ew = None                     # None == unweighted (original graph)
    nw = torch.ones(data.num_nodes)
    n = data.num_nodes

    feature_size = None           # hashing-matrix width at level 1 (for the table)
    ei_eig, ew_eig = None, None   # latest "with-mass" adjacency (for eigen err)
    achieved_cum = []
    n_stages = len(per_stage_rr)
    for lvl, t_rr in enumerate(per_stage_rr, start=1):
        # (re)build THIS level's hashing matrix exactly the way Z_o was built
        Z = build_z(comps_cur, mode, coeffs)
        if lvl == 1:
            feature_size = Z.shape[-1]

        (z_c, A_c_dense, ei_eig, ew_eig, ei, ew, nw, n, achieved_rr, cutoff) = \
            coarsen_one_level(Z, ei, ew, nw, n, t_rr, args, device,
                              mass_mode=args.mass_mode)

        # rebuild the building blocks on the coarsened graph for the NEXT level:
        # coarsened feature z_c plays the role of X, the coarsened adjacency A_c
        # gives A_proj / A_raw, and hop = A_c @ z_c is the "A_c Z_c" term.
        # project_adj=False -> A_c is used directly (NOT JL-projected again).
        # normalize=False    -> NO second normalization: z_c is already the
        # normalized aggregate and A_c / A_c@z_c are used as produced, so each
        # quantity is normalized exactly once in its lineage.
        if lvl < n_stages:
            comps_cur = build_components_from(
                z_c, A_c_dense, args.aproj_dim, s=1,
                project_adj=False, normalize=False,
            )

        cum_remaining = n / data.num_nodes
        achieved_cum.append(round(cum_remaining * 100, 2))
        print(f"  [stage {lvl}/{n_stages}] "
              f"target cum {cum_fractions[lvl-1]*100:.1f}% of N  |  "
              f"hash dim {Z.shape[-1]}  |  "
              f"this-level rr requested {t_rr*100:.2f}% / achieved {achieved_rr*100:.2f}%  |  "
              f"nodes {n}  (cumulative {cum_remaining*100:.2f}% of N)  "
              f"[cutoff={cutoff:.6g}]")

    final_num_nodes = n
    overall_rr = 1 - final_num_nodes / data.num_nodes

    # Eigen error (relative) of the FINAL coarsened graph vs the ORIGINAL graph.
    if data.num_nodes < 100:
        num_eigen = data.num_nodes // 2
    else:
        num_eigen = 100
    if args.dataname == "texas":
        num_eigen = 80

    eigen_error = spectral_properties.eigen_error(
        data.edge_index,   # original graph
        ei_eig,            # final coarsened edge index (with size-aware mass)
        ew_eig,            # final coarsened edge weights
        num_eigen,
    )
    eigen_error = np.asarray(eigen_error, dtype=np.float64)

    return {
        "mode": mode,
        "mode_desc": MODE_DESC[mode],
        "coeff_scheme": scheme_name,
        "hash_norm": args.hash_norm,
        "coeff_X": round(float(cX), 4),
        "coeff_A": round(float(cA), 4),
        "coeff_Ax": round(float(cAx), 4),
        "feature_size": int(feature_size),
        "num_nodes": int(data.num_nodes),
        "stages": int(len(per_stage_rr)),
        "mass_mode": args.mass_mode,
        "stage_targets_pct": ", ".join(f"{f*100:.1f}" for f in cum_fractions),
        "stage_achieved_pct": ", ".join(str(a) for a in achieved_cum),
        "final_num_nodes": int(final_num_nodes),
        "overall_reduction_ratio_pct": round(float(overall_rr) * 100, 2),
        "num_eigen": int(num_eigen),
        "mean_relative_eigen_error": float(np.mean(eigen_error)),
        "std_relative_eigen_error": float(np.std(eigen_error)),
        "max_relative_eigen_error": float(np.max(eigen_error)),
    }


# ----------------------------------------------------------------------------- #
#  Output: write per-dataset xlsx + append to master
# ----------------------------------------------------------------------------- #

def write_results(df, dataname, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    per_dataset_path = os.path.join(out_dir, f"eigen_results_{dataname.lower()}.xlsx")
    master_path = os.path.join(out_dir, "eigen_results_master.xlsx")

    def _to_excel(frame, path):
        try:
            frame.to_excel(path, index=False, engine="openpyxl")
            return path
        except Exception as e:
            csv_path = path.replace(".xlsx", ".csv")
            frame.to_csv(csv_path, index=False)
            print(f"  [warn] could not write xlsx ({e}); wrote CSV instead: {csv_path}")
            return csv_path

    p1 = _to_excel(df, per_dataset_path)
    print(f"\nSaved per-dataset results -> {p1}")

    # Append to / refresh the master file (drop any old rows for this dataset).
    try:
        if os.path.exists(master_path):
            old = pd.read_excel(master_path, engine="openpyxl")
            old = old[old["dataset"].str.lower() != dataname.lower()]
            combined = pd.concat([old, df], ignore_index=True)
        else:
            combined = df.copy()
        p2 = _to_excel(combined, master_path)
        print(f"Updated master results     -> {p2}")
    except Exception as e:
        print(f"  [warn] could not update master file: {e}")


# ----------------------------------------------------------------------------- #
#  Main
# ----------------------------------------------------------------------------- #

def resolve_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available -> falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-level graph hashing/coarsening eigen-error experiment."
    )
    parser.add_argument("--dataname", "--dataset", dest="dataname", required=True,
                        help="Dataset name: cora, citeseer, pubmed, squirrel, "
                             "chameleon, texas, computers, cs")
    parser.add_argument("--device", default="cpu",
                        help="torch device, e.g. 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--root", default="/home/h9ri/Notebooks/Final/graph_hashing/data",
                        help="Root directory for dataset downloads/cache")
    parser.add_argument("--out", default="./ugc_coar_same_lavel",
                        help="Output directory for xlsx files (per-dataset + "
                             "master). Default: ./ugc_coar_same_lavel")
    parser.add_argument("--no_of_hash", type=int, default=1000)
    parser.add_argument("--aproj_dim", type=int, default=256)
    parser.add_argument("--hash_norm", default="dot",
                        choices=["dot", "L2-norm", "L1-norm"],
                        help="How Bin_values are computed from Z (default: dot)")
    # ---- multilevel controls ----
    parser.add_argument("--stages", type=int, default=2,
                        help="Number of successive coarsening levels (>=1). "
                             "With --stages 1 you get the original single-level "
                             "behaviour.")
    parser.add_argument("--cr", type=float, default=0.5,
                        help="FINAL overall reduction ratio of the whole "
                             "cascade, relative to the original graph. "
                             "cr=0.5 -> final graph keeps N/2 nodes.")
    parser.add_argument("--stage_fractions", default=None,
                        help="Optional explicit schedule: comma-separated, "
                             "strictly-decreasing CUMULATIVE fractions-of-N "
                             "remaining after each stage, e.g. '0.9,0.5'. "
                             "Overrides --stages and --cr.")
    parser.add_argument("--mass_mode", default="accumulate",
                        choices=["accumulate", "fresh"],
                        help="How the size-aware diagonal is carried across "
                             "levels. 'accumulate' (default) feeds the full "
                             "coarsened adjacency (incl. diagonal mass) into "
                             "the next level; 'fresh' re-derives the mass from "
                             "cumulative node counts each level (no compounding).")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Tolerance for the per-level reduction-ratio bisection")
    parser.add_argument("--projectors", default="normal",
                        choices=["normal", "uniform", "VAEs"],
                        help="Random projector distribution for hashing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Parse the explicit schedule (if any) and let it define stages / cr.
    if args.stage_fractions:
        fracs = [float(x) for x in args.stage_fractions.split(",")]
        args.stage_fractions_parsed = fracs
        args.stages = len(fracs)
        args.cr = 1.0 - fracs[-1]
    else:
        args.stage_fractions_parsed = None

    if args.stages < 1:
        raise ValueError("--stages must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Load data.
    data, num_features, num_classes = load_dataset(args.dataname, args.root, seed=args.seed)

    # Homophily.
    h_f = compute_homophily(data)
    print(f"Edge Homophily: {h_f:.4f}")

    # Report the coarsening schedule once up front.
    cum_fractions, per_stage_rr = compute_stage_targets(
        args.stages, args.cr, args.stage_fractions_parsed
    )
    print("Coarsening schedule (cumulative % of N remaining):",
          " -> ".join(f"{f*100:.1f}%" for f in cum_fractions))
    print("Per-level reduction ratios requested        :",
          ", ".join(f"{r*100:.2f}%" for r in per_stage_rr))
    print(f"Diagonal mass mode: {args.mass_mode}")

    # Precompute the building blocks once.
    print("Precomputing feature components (X, A_proj, A_raw, hop)...")
    comps = precompute_components(data, aproj_dim=args.aproj_dim, s=1)

    # Run the full grid: 3 modes x 3 coefficient schemes.
    rows = []
    for mode in MODES:
        for scheme_name in SCHEMES:
            tag = f"[{args.dataname} | mode={mode} | scheme={scheme_name}]"
            print(f"\n=== {tag} ===")
            t0 = time.time()
            try:
                rec = run_single(data, comps, num_classes, mode, scheme_name,
                                 h_f, args, device)
                rec["dataset"] = args.dataname
                rec["homophily"] = round(float(h_f), 4)
                rec["status"] = "ok"
                rec["seconds"] = round(time.time() - t0, 2)
                print(f"  mean relative eigen error = "
                      f"{rec['mean_relative_eigen_error']:.6f}  "
                      f"({rec['seconds']}s)")
                rows.append(rec)
            except Exception as e:
                print(f"  [error] {tag} failed: {e}")
                rows.append({
                    "dataset": args.dataname,
                    "mode": mode,
                    "mode_desc": MODE_DESC[mode],
                    "coeff_scheme": scheme_name,
                    "homophily": round(float(h_f), 4),
                    "status": f"error: {e}",
                    "mean_relative_eigen_error": np.nan,
                })

    # Order columns nicely.
    col_order = [
        "dataset", "homophily", "mode", "mode_desc", "coeff_scheme", "hash_norm",
        "coeff_X", "coeff_A", "coeff_Ax", "feature_size",
        "num_nodes", "stages", "mass_mode",
        "stage_targets_pct", "stage_achieved_pct",
        "final_num_nodes", "overall_reduction_ratio_pct",
        "num_eigen",
        "mean_relative_eigen_error", "std_relative_eigen_error",
        "max_relative_eigen_error", "seconds", "status",
    ]
    df = pd.DataFrame(rows)
    df = df.reindex(columns=[c for c in col_order if c in df.columns])

    print("\n================= SUMMARY =================")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        cols = [c for c in ["mode", "coeff_scheme", "stages",
                            "overall_reduction_ratio_pct",
                            "mean_relative_eigen_error"] if c in df.columns]
        print(df[cols].to_string(index=False))

    write_results(df, args.dataname, args.out)


if __name__ == "__main__":
    main()