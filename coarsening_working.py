"""
Tri3GN-UGC Graph Coarsening
Datasets: Cora, Citeseer, Pubmed, Texas, Wisconsin, Cornell,
          Computers, CS, DBLP, Physics, Film, Squirrel, Chameleon

Device-aware: runs on CPU or CUDA. Heavy tensor ops (branches, projections,
distances) run on the selected device; sparse eigen/coarsening stays on CPU.
Random matrices are drawn on CPU with a fixed seed then moved to the device,
so output is reproducible across CPU and CUDA.
"""

import os
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch_geometric.datasets import (
    Planetoid, WebKB, Amazon, Coauthor, CitationFull, WikipediaNetwork, Actor,
    Flickr, Yelp, Reddit
)
from torch_geometric.nn import GCNConv

# ── Path configuration — change this to your dataset directory ─────────────────
DATA_ROOT = "./data"
# ──────────────────────────────────────────────────────────────────────────────

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
COARSENING_RATIO = 0.5
K_EIGENVALUES    = 100
BRANCH_DIM       = 128      # d
HOPS             = 3        # R
HOP_WEIGHTS      = [1/6, 2/6, 3/6]   # β
SKETCH_REPEATS   = 2
LARGE_THRESHOLD  = 6000
N_PROJECTIONS    = 12
SEED             = 7
# ──────────────────────────────────────────────────────────────────────────────


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(name: str, root: str):
    """Return a single PyG Data object for the given dataset name.
    Dataset names and loader classes match the server code exactly.
    """
    name_lower = name.lower()
    if name_lower == "cora":
        ds = Planetoid(root=root, name="Cora")
    elif name_lower == "citeseer":
        ds = Planetoid(root=root, name="CiteSeer")
    elif name_lower == "pubmed":
        ds = Planetoid(root=root, name="PubMed")
    elif name_lower == "texas":
        ds = WebKB(root=root, name="Texas")
    elif name_lower == "wisconsin":
        ds = WebKB(root=root, name="Wisconsin")
    elif name_lower == "cornell":
        ds = WebKB(root=root, name="Cornell")
    elif name_lower == "computers":
        ds = Amazon(root=root, name="Computers")
    elif name_lower == "cs":
        ds = Coauthor(root=root, name="CS")
    elif name_lower == "dblp":
        ds = CitationFull(root=root, name="DBLP")
    elif name_lower == "physics":
        ds = Coauthor(root=root, name="Physics")
    elif name_lower == "film":
        ds = Actor(root=os.path.join(root, "Actor"))
    elif name_lower == "squirrel":
        ds = WikipediaNetwork(root=root, name="squirrel", geom_gcn_preprocess=True)
    elif name_lower == "chameleon":
        ds = WikipediaNetwork(root=root, name="chameleon", geom_gcn_preprocess=True)
    elif name_lower == "flickr":
        ds = Flickr(root=os.path.join(root, "Flickr"))
    elif name_lower == "yelp":
        ds = Yelp(root=os.path.join(root, "Yelp"))
    elif name_lower == "reddit":
        ds = Reddit(root=os.path.join(root, "Reddit"))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return ds[0]


# ── Random helpers (CPU-seeded, device-portable) ───────────────────────────────

def make_generator():
    """CPU generator seeded for reproducibility across devices."""
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    return g


def randn_projection(f: int, gen: torch.Generator, device) -> torch.Tensor:
    """JL projection Π ∈ R^{f×128}, entries ~ N(0, 1/128)."""
    Pi = torch.randn((f, BRANCH_DIM), generator=gen, dtype=torch.float32)
    Pi = Pi / np.sqrt(BRANCH_DIM)
    return Pi.to(device)


# ── Adjacency helpers ──────────────────────────────────────────────────────────

def build_sparse_adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    """Build binary symmetric CSR adjacency (CPU/SciPy) from edge_index."""
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A = A + A.T                       # symmetrize
    A.data = np.ones_like(A.data)     # binarize
    A.setdiag(0)                      # remove self-loops
    A.eliminate_zeros()
    return A


def scipy_to_torch_sparse(A: sp.csr_matrix, device) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a torch sparse_coo tensor on device."""
    Acoo = A.tocoo()
    idx  = np.vstack([Acoo.row, Acoo.col])
    indices = torch.tensor(idx, dtype=torch.long)
    values  = torch.tensor(Acoo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, A.shape).coalesce().to(device)


def normalized_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} A D^{-1/2}."""
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ── Step 2: Edge homophily ─────────────────────────────────────────────────────

def compute_homophily(A: sp.csr_matrix, y: np.ndarray) -> float:
    cx = A.tocoo()
    same = np.sum(y[cx.row] == y[cx.col])
    total = cx.nnz
    return float(same) / float(total) if total > 0 else 0.0


# ── Step 3: Bernstein weights ──────────────────────────────────────────────────

def bernstein_weights(h: float):
    alpha_X  = (1 - h) ** 2
    alpha_A  = 2 * h * (1 - h)
    alpha_AX = h ** 2
    return alpha_X, alpha_A, alpha_AX


# ── Row normalization (torch) ──────────────────────────────────────────────────

def row_norm(X: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(X, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return X / norms


# ── Step 4a: Feature branch ────────────────────────────────────────────────────

def feature_branch(X: torch.Tensor, gen: torch.Generator, device) -> torch.Tensor:
    f = X.shape[1]
    Pi = randn_projection(f, gen, device)
    return row_norm(row_norm(X) @ Pi)


# ── Step 4b: CountSketch branch ───────────────────────────────────────────────

def count_sketch_branch(A: sp.csr_matrix, gen: torch.Generator, device) -> torch.Tensor:
    n = A.shape[0]
    cx = A.tocoo()
    src = torch.tensor(cx.row, dtype=torch.long, device=device)
    tgt = torch.tensor(cx.col, dtype=torch.long, device=device)
    vals = torch.tensor(cx.data, dtype=torch.float32, device=device)

    sketch_sum = torch.zeros((n, BRANCH_DIM), dtype=torch.float32, device=device)
    for _ in range(SKETCH_REPEATS):
        h_idx = torch.randint(0, BRANCH_DIM, (n,), generator=gen).to(device)
        signs = (torch.randint(0, 2, (n,), generator=gen).to(device).float() * 2 - 1)
        # per-edge contribution: signs[tgt] * val into bin h_idx[tgt] of row src
        contrib   = signs[tgt] * vals
        flat_idx  = src * BRANCH_DIM + h_idx[tgt]
        sketch    = torch.zeros(n * BRANCH_DIM, dtype=torch.float32, device=device)
        sketch.index_add_(0, flat_idx, contrib)
        sketch_sum += sketch.view(n, BRANCH_DIM)
    return row_norm(sketch_sum / SKETCH_REPEATS)


# ── Step 4c: Multi-hop branch ─────────────────────────────────────────────────

def multihop_branch(A: sp.csr_matrix, X: torch.Tensor,
                    gen: torch.Generator, device) -> torch.Tensor:
    f = X.shape[1]
    A_norm = scipy_to_torch_sparse(normalized_adjacency(A), device)
    H  = torch.zeros_like(X)
    AX = X
    for r in range(1, HOPS + 1):
        AX = torch.sparse.mm(A_norm, AX)
        H = H + HOP_WEIGHTS[r - 1] * AX
    Pi = randn_projection(f, gen, device)
    return row_norm(row_norm(H) @ Pi)


# ── Step 5: Fuse branches ──────────────────────────────────────────────────────

def fuse(Z_X, Z_A, Z_AX, alpha_X, alpha_A, alpha_AX) -> torch.Tensor:
    Z = alpha_X * Z_X + alpha_A * Z_A + alpha_AX * Z_AX
    return row_norm(Z)


# ── Step 7: Small-graph candidate generation (n <= 6000) ──────────────────────

def candidates_small(Z: torch.Tensor, top_k: int = 8):
    """Full pairwise distance on device, keep top-k neighbors per node."""
    n = Z.shape[0]
    D = torch.cdist(Z, Z)                       # (n, n) on device
    D.fill_diagonal_(float("inf"))
    k = min(top_k, n - 1)
    dists, nbrs = torch.topk(D, k, dim=1, largest=False)
    dists = dists.cpu().numpy()
    nbrs  = nbrs.cpu().numpy()

    candidates = {}
    for i in range(n):
        for col in range(k):
            j = int(nbrs[i, col])
            if i == j:
                continue
            pair  = (min(i, j), max(i, j))
            score = float(dists[i, col])
            if pair not in candidates or candidates[pair] > score:
                candidates[pair] = score
    return candidates


# ── Step 8: Large-graph candidate generation (n > 6000) ───────────────────────

def candidates_large(Z: torch.Tensor, gen: torch.Generator, device):
    """12 random projections, consecutive pairs in sorted order."""
    n = Z.shape[0]
    candidates = {}
    for _ in range(N_PROJECTIONS):
        v = torch.randn(BRANCH_DIM, generator=gen, dtype=torch.float32).to(device)
        v = v / torch.linalg.norm(v)
        scores = (Z @ v)                          # (n,) on device
        order  = torch.argsort(scores)
        scores_np = scores.cpu().numpy()
        order_np  = order.cpu().numpy()
        for k in range(n):
            for offset in (1, 2):
                if k + offset >= n:
                    break
                i = int(order_np[k])
                j = int(order_np[k + offset])
                pair  = (min(i, j), max(i, j))
                score = abs(float(scores_np[i]) - float(scores_np[j]))
                if pair not in candidates or candidates[pair] > score:
                    candidates[pair] = score
    return candidates


# ── Step 9: Greedy disjoint matching ──────────────────────────────────────────

def greedy_matching(candidates: dict, n: int, q: int):
    sorted_pairs = sorted(candidates.items(), key=lambda x: x[1])
    assigned = np.zeros(n, dtype=bool)
    supernodes = []
    for (i, j), _ in sorted_pairs:
        if not assigned[i] and not assigned[j]:
            supernodes.append((i, j))
            assigned[i] = True
            assigned[j] = True
        if len(supernodes) == q:
            break
    return supernodes, assigned


# ── Step 10: Fallback ──────────────────────────────────────────────────────────

def fallback_small(Z: torch.Tensor, assigned: np.ndarray, supernodes: list, q: int):
    remaining = np.where(~assigned)[0]
    if len(remaining) < 2:
        return supernodes, assigned
    Z_rem = Z[torch.tensor(remaining, device=Z.device)]
    D = torch.cdist(Z_rem, Z_rem)
    D.fill_diagonal_(float("inf"))
    nn = torch.argmin(D, dim=1).cpu().numpy()
    used = np.zeros(len(remaining), dtype=bool)
    for i in range(len(remaining)):
        if len(supernodes) >= q:
            break
        if used[i]:
            continue
        j = int(nn[i])
        if not used[j]:
            supernodes.append((int(remaining[i]), int(remaining[j])))
            assigned[remaining[i]] = True
            assigned[remaining[j]] = True
            used[i] = True
            used[j] = True
    return supernodes, assigned


def fallback_large(Z: torch.Tensor, A: sp.csr_matrix,
                   assigned: np.ndarray, supernodes: list,
                   q: int, gen: torch.Generator, device):
    remaining = np.where(~assigned)[0]
    if len(remaining) < 2:
        return supernodes, assigned
    v = torch.randn(BRANCH_DIM, generator=gen, dtype=torch.float32).to(device)
    v = v / torch.linalg.norm(v)
    Z_rem  = Z[torch.tensor(remaining, device=Z.device)]
    scores = (Z_rem @ v).cpu().numpy()
    order  = remaining[np.argsort(scores)]
    used   = np.zeros(len(order), dtype=bool)
    # consecutive pairs in sorted order
    for k in range(len(order) - 1):
        if len(supernodes) >= q:
            break
        if not used[k] and not used[k + 1]:
            supernodes.append((int(order[k]), int(order[k + 1])))
            assigned[order[k]] = True
            assigned[order[k + 1]] = True
            used[k] = True
            used[k + 1] = True
    # edge-connected pairs among still-remaining
    if len(supernodes) < q:
        still_set = set(np.where(~assigned)[0].tolist())
        cx = A.tocoo()
        for src, tgt in zip(cx.row, cx.col):
            if len(supernodes) >= q:
                break
            if src in still_set and tgt in still_set and src != tgt:
                supernodes.append((int(src), int(tgt)))
                assigned[src] = True
                assigned[tgt] = True
                still_set.discard(src)
                still_set.discard(tgt)
    return supernodes, assigned


# ── Steps 11-12: Assignment matrix P ──────────────────────────────────────────

def build_assignment(supernodes: list, n: int):
    """Returns cluster vector c and normalized assignment matrix P (sparse, CPU)."""
    c = np.full(n, -1, dtype=np.int32)
    sid = 0
    for (i, j) in supernodes:
        c[i] = sid
        c[j] = sid
        sid += 1
    for i in range(n):
        if c[i] == -1:
            c[i] = sid
            sid += 1
    m = sid

    rows = np.arange(n)
    data = np.ones(n, dtype=np.float32)
    C = sp.coo_matrix((data, (rows, c)), shape=(n, m)).tocsr()

    sizes = np.array(C.sum(axis=0)).flatten()
    scale = 1.0 / np.sqrt(np.maximum(sizes, 1))
    P = C @ sp.diags(scale)
    return c, P, m


# ── Step 13: Coarse adjacency ─────────────────────────────────────────────────

def coarsen_adjacency(A: sp.csr_matrix, P: sp.spmatrix) -> sp.csr_matrix:
    Ac = P.T @ A @ P
    Ac = (Ac + Ac.T) / 2
    Ac = Ac.tocsr()
    Ac.setdiag(0)
    Ac.eliminate_zeros()
    return Ac


# ── Spectral metric: REE ───────────────────────────────────────────────────────

def laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.array(A.sum(axis=1)).flatten()
    return sp.diags(deg) - A


def top_k_eigenvalues(L: sp.csr_matrix, k: int) -> np.ndarray:
    k = min(k, L.shape[0] - 2)
    if k < 1:
        return np.array([])
    vals, _ = eigsh(L, k=k, which="LM")
    return np.sort(vals)[::-1]


def compute_ree(A: sp.csr_matrix, Ac: sp.csr_matrix, k: int = K_EIGENVALUES) -> float:
    lam  = top_k_eigenvalues(laplacian(A),  k)
    lamc = top_k_eigenvalues(laplacian(Ac), k)
    k    = min(len(lam), len(lamc))
    if k < 1:
        return float("nan")
    lam, lamc = lam[:k], lamc[:k]
    gamma = lam.mean() / (lamc.mean() + 1e-12)
    return float(np.mean(np.abs(lam - gamma * lamc) / (np.abs(lam) + 1e-12)))


# ── Main coarsening pipeline ───────────────────────────────────────────────────

def coarsen(data, device):
    gen = make_generator()

    X_np = data.x.cpu().numpy().astype(np.float32)
    y_raw = data.y.cpu().numpy()
    # multi-label case (e.g. Yelp): y is (n, num_classes) — collapse to 1D via argmax
    if y_raw.ndim == 2:
        y = y_raw.argmax(axis=1).astype(np.int32)
    else:
        y = y_raw.astype(np.int32)
    n    = X_np.shape[0]
    X    = torch.tensor(X_np, dtype=torch.float32, device=device)

    # Step 1
    A = build_sparse_adjacency(data.edge_index, n)

    # Step 2-3
    h = compute_homophily(A, y)
    alpha_X, alpha_A, alpha_AX = bernstein_weights(h)

    # Step 4 (order of rng draws fixed: feature → sketch → multihop)
    Z_X  = feature_branch(X, gen, device)
    Z_A  = count_sketch_branch(A, gen, device)
    Z_AX = multihop_branch(A, X, gen, device)

    # Step 5
    Z = fuse(Z_X, Z_A, Z_AX, alpha_X, alpha_A, alpha_AX)

    # Step 6
    m_target = round(n * (1 - COARSENING_RATIO))
    q        = n - m_target

    # Steps 7/8
    if n <= LARGE_THRESHOLD:
        candidates = candidates_small(Z)
    else:
        candidates = candidates_large(Z, gen, device)

    # Step 9
    supernodes, assigned = greedy_matching(candidates, n, q)

    # Step 10
    if len(supernodes) < q:
        if n <= LARGE_THRESHOLD:
            supernodes, assigned = fallback_small(Z, assigned, supernodes, q)
        else:
            supernodes, assigned = fallback_large(Z, A, assigned, supernodes, q, gen, device)

    # Steps 11-13
    c, P, m_actual = build_assignment(supernodes, n)
    Ac = coarsen_adjacency(A, P)

    return A, Ac, c, P, m_actual, h, alpha_X, alpha_A, alpha_AX


# ── GCN training on the coarsened graph ────────────────────────────────────────

# GCN hyperparameters
GCN_HIDDEN  = 64
GCN_DROPOUT = 0.5
GCN_LR      = 0.01
GCN_WD      = 5e-4
GCN_EPOCHS  = 200


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=GCN_DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def random_split(n, train_ratio=0.6, val_ratio=0.2, seed=SEED):
    """Boolean train/val/test masks from a fresh seeded random split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_tr = int(train_ratio * n)
    n_va = int(val_ratio * n)
    train_mask = np.zeros(n, dtype=bool)
    val_mask   = np.zeros(n, dtype=bool)
    test_mask  = np.zeros(n, dtype=bool)
    train_mask[perm[:n_tr]]            = True
    val_mask[perm[n_tr:n_tr + n_va]]   = True
    test_mask[perm[n_tr + n_va:]]      = True
    return train_mask, val_mask, test_mask


def edges_from_scipy(A: sp.csr_matrix, device):
    """Return (edge_index, edge_weight) torch tensors from a SciPy matrix."""
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return ei, ew


def train_gcn(data, A, Ac, P, c, m, num_classes, device):
    """Train a GCN on the coarsened graph, evaluate on the original test set.

    Following the single-stage coarsening idea:
      - coarse features : X_c = P^T X  (normalized aggregate of member features)
      - coarse labels   : one-hot labels with NON-train nodes zeroed, aggregated
                          via P^T, then argmax (only train labels contribute)
      - coarse train set: supernodes that contain at least one train node
    The trained GCN is then applied to the ORIGINAL graph and accuracy is
    measured on the original test nodes.
    """
    X_np = data.x.cpu().numpy().astype(np.float32)
    y_np = data.y.cpu().numpy().astype(np.int64)
    n, f = X_np.shape

    train_mask, _, test_mask = random_split(n)

    P_t = P.T.tocsr()                                   # (m, n)

    # coarse features  X_c = P^T X
    Xc = torch.tensor(P_t @ X_np, dtype=torch.float32, device=device)

    # coarse labels: one-hot, zero non-train, aggregate, argmax
    Yhot = np.zeros((n, num_classes), dtype=np.float32)
    Yhot[np.arange(n), y_np] = 1.0
    Yhot[~train_mask] = 0.0
    yc = torch.tensor(np.argmax(P_t @ Yhot, axis=1), dtype=torch.long, device=device)

    # coarse train mask: supernodes that hold a train node
    coarse_train = np.zeros(m, dtype=bool)
    coarse_train[c[train_mask]] = True
    coarse_train_t = torch.tensor(coarse_train, device=device)

    ei_c, ew_c = edges_from_scipy(Ac, device)
    ei_o, ew_o = edges_from_scipy(A,  device)
    X_orig = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_orig = torch.tensor(y_np, dtype=torch.long, device=device)
    test_mask_t = torch.tensor(test_mask, device=device)

    model = GCN(f, GCN_HIDDEN, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=GCN_LR, weight_decay=GCN_WD)

    model.train()
    for _ in range(GCN_EPOCHS):
        opt.zero_grad()
        out = model(Xc, ei_c, ew_c)
        loss = F.cross_entropy(out[coarse_train_t], yc[coarse_train_t])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_orig, ei_o, ew_o).argmax(dim=1)
        acc = (pred[test_mask_t] == y_orig[test_mask_t]).float().mean().item()
    return acc


# ── Entry point ────────────────────────────────────────────────────────────────

DATASETS = [
    "cora", "citeseer", "pubmed",
    "texas", "wisconsin", "cornell",
    "computers", "cs", "dblp", "physics",
    "film", "squirrel", "chameleon",
    "flickr", "yelp", "reddit",
]


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false for --gcn")


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(arg)


def main():
    parser = argparse.ArgumentParser(description="Tri3GN-UGC Graph Coarsening")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Root directory for PyG datasets")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Datasets to run (default: all)")
    parser.add_argument("--gcn", type=str2bool, default=False,
                        help="Train a GCN on the coarsened graph and record "
                             "test accuracy (true/false, default: false)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    summary_dir = os.path.join(script_dir, "summary")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    fieldnames = ["dataset", "n_nodes", "n_edges", "m_supernodes",
                  "homophily", "alpha_X", "alpha_A", "alpha_AX",
                  "ree", "gcn_acc", "time_s"]

    for name in args.datasets:
        print(f"\n{'='*50}\nDataset: {name}")
        row = {k: "" for k in fieldnames}
        row["dataset"] = name
        try:
            data = load_dataset(name, args.data_root)
            n = data.num_nodes
            e = data.edge_index.shape[1]
            print(f"  Nodes: {n}  Edges: {e}")

            t0 = time.time()
            A, Ac, c, P, m, h, alpha_X, alpha_A, alpha_AX = coarsen(data, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            print("  Computing REE ...")
            ree = compute_ree(A, Ac)

            gcn_acc = ""
            if args.gcn:
                print("  Training GCN on coarsened graph ...")
                num_classes = int(data.y.max().item()) + 1
                gcn_acc = train_gcn(data, A, Ac, P, c, m, num_classes, device)
                print(f"  GCN test accuracy: {gcn_acc:.4f}")

            print(f"  Supernodes: {m}  Homophily: {h:.4f}  "
                  f"α_X={alpha_X:.4f}  α_A={alpha_A:.4f}  α_AX={alpha_AX:.4f}  "
                  f"REE: {ree:.4f}  Time: {elapsed:.2f}s")

            row.update({
                "n_nodes": n, "n_edges": e, "m_supernodes": m,
                "homophily": round(h, 4),
                "alpha_X":  round(alpha_X,  4),
                "alpha_A":  round(alpha_A,  4),
                "alpha_AX": round(alpha_AX, 4),
                "ree": round(ree, 4),
                "gcn_acc": round(gcn_acc, 4) if gcn_acc != "" else "",
                "time_s": round(elapsed, 2),
            })

        except Exception as ex:
            print(f"  ERROR: {ex}")
            row["time_s"] = f"ERROR: {ex}"

        csv_path = os.path.join(results_dir, f"{name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"  Saved → {csv_path}")

        # append to shared summary CSV
        summary_path   = os.path.join(summary_dir, "summary.csv")
        summary_fields = ["dataset", "coarsening_ratio",
                          "alpha_X", "alpha_A", "alpha_AX",
                          "ree", "gcn_acc", "time_s"]
        file_exists    = os.path.isfile(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "dataset":          name,
                "coarsening_ratio": COARSENING_RATIO,
                "alpha_X":          row["alpha_X"],
                "alpha_A":          row["alpha_A"],
                "alpha_AX":         row["alpha_AX"],
                "ree":              row["ree"],
                "gcn_acc":          row["gcn_acc"],
                "time_s":           row["time_s"],
            })
        print(f"  Appended → {summary_path}")


if __name__ == "__main__":
    main()
