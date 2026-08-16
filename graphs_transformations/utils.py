import json
import math
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.nn.functional import normalize
from torch_geometric.utils import get_laplacian, to_dense_adj


def get_percentile_radius(
    data: torch.Tensor,
    mask: torch.Tensor,
    percentile: float,
    cosine: bool = False,
) -> float:
    if torch.isnan(data).any():
        means = data.nanmean(dim=1, keepdim=True)
        data = torch.where(mask, data, means)

    if cosine:
        data = normalize(data, p=2, dim=1)

    dists = torch.cdist(data, data, p=2) if not cosine else 1 - torch.matmul(data, data.T)
    mask_self = ~torch.eye(data.shape[1], dtype=torch.bool)
    dists = dists[mask_self]  # remove self-distances

    min_dist = dists.min()
    max_dist = dists.max()

    # radius = torch.quantile(dists, percentile).item()
    radius = min_dist + percentile * (max_dist - min_dist)

    return radius.item()


def get_percentile_k(data: torch.Tensor, percentile: float, loop: bool = False) -> int:
    shape = data.shape
    max_k = shape[1] if loop else shape[1] - 1

    k = round(percentile * max_k)
    print(f"{percentile=} {data.shape[1]=} {max_k=} {k=}")

    return max(0, min(k, max_k))


def embed_time_series(x: torch.Tensor, dim: int, time_delay: int) -> torch.Tensor:
    N = x.size(0) - (dim - 1) * time_delay
    N = int(N)
    return torch.stack(([x[i : i + N] for i in range(0, dim * time_delay, time_delay)]), dim=1)


def get_radius_for_rec(
    x: torch.Tensor,
    alpha: float,
    dim: int,
    time_delay: int,
    low: float = 0.0,
    high: float = 100.0,
) -> float:
    X_emb = embed_time_series(x, dim, time_delay)

    dists = torch.cdist(X_emb, X_emb, p=2)
    dists = dists[dists > 0]

    r_min = torch.quantile(dists, low / 100.0)
    r_max = torch.quantile(dists, high / 100.0)

    r = r_min + alpha * (r_max - r_min)

    return r.item()


def compute_laplacian_smoothness(x, edge_index, edge_weight=None, mask=None, normalize=True, debug=False):
    batch_size, nodes, features = x.shape
    lap_edge_index, lap_edge_weight = get_laplacian(edge_index, edge_weight, normalization="sym")
    laplacian = to_dense_adj(lap_edge_index, edge_attr=lap_edge_weight, max_num_nodes=nodes).squeeze(0)

    # FIX: Force symmetry and positive semi-definiteness
    laplacian = 0.5 * (laplacian + laplacian.t())  # Ensure symmetry

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)  # Remove negative eigenvalues
    laplacian = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()

    if debug:
        # Check if Laplacian is symmetric
        is_symmetric = torch.allclose(laplacian, laplacian.t(), atol=1e-6)
        print(f"Laplacian is symmetric: {is_symmetric}")

        # Check eigenvalues to verify positive semi-definiteness
        try:
            eigenvalues = torch.linalg.eigvalsh(laplacian)
            min_eig = eigenvalues.min().item()
            max_eig = eigenvalues.max().item()
            print(f"Eigenvalue range: [{min_eig:.6f}, {max_eig:.6f}]")
            print(f"Any negative eigenvalues: {(eigenvalues < -1e-6).any().item()}")
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")

    if mask is not None:
        x = x.masked_fill(~mask, 0.0)

    # x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

    x_flat = x.permute(0, 2, 1).reshape(batch_size * features, nodes).unsqueeze(1)  # [B*F, 1, N]
    laplacian_expanded = laplacian.unsqueeze(0).expand(batch_size * features, -1, -1)  # [B*F, N, N]
    smoothness = torch.bmm(torch.bmm(x_flat, laplacian_expanded), x_flat.transpose(1, 2)).squeeze()

    smoothness_total = smoothness.sum()

    if normalize:
        energy = torch.sum(x**2) + 1e-8
        return (smoothness_total / energy).item()
    else:
        return smoothness_total.item()


def compute_edge_difference_smoothness(x, edge_index, edge_weight=None, mask=None, normalize=True):
    B, N, F = x.shape
    row, col = edge_index  # [E]

    x_i = x[:, row, :]  # [B, E, F]
    x_j = x[:, col, :]
    diff = x_i - x_j
    sq_diff = diff**2  # [B, E, F]

    if mask is not None:
        m_i = mask[:, row, :]
        m_j = mask[:, col, :]
        edge_mask = m_i & m_j  # [B, E, F]
        sq_diff = sq_diff * edge_mask.float()

    if edge_weight is not None:
        w = edge_weight.view(1, -1, 1)  # [1, E, 1]
        sq_diff = sq_diff * w  # weighted squared diff

    smoothness = sq_diff.sum()

    if normalize:
        energy = (x**2).sum() + 1e-8
        return (smoothness / energy).item()
    return smoothness.item()


def save_graph_characteristics(adjacency_matrix: torch.Tensor, binary_graph: bool, save_path: str) -> None:

    adj = adjacency_matrix.detach().cpu().numpy()
    is_weighted = not binary_graph

    G = nx.from_numpy_array(adj)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # ============================================================
    # Degree statistics
    # ============================================================

    binary_degrees = np.asarray(
        [d for _, d in G.degree(weight=None)],
        dtype=float,
    )

    binary_avg_degree = float(np.mean(binary_degrees)) if len(binary_degrees) else 0.0
    binary_median_degree = float(np.median(binary_degrees)) if len(binary_degrees) else 0.0
    binary_min_degree = float(np.min(binary_degrees)) if len(binary_degrees) else 0.0
    binary_max_degree = float(np.max(binary_degrees)) if len(binary_degrees) else 0.0
    binary_degree_std = float(np.std(binary_degrees)) if len(binary_degrees) else 0.0

    if is_weighted:
        degrees = np.asarray(
            [d for _, d in G.degree(weight="weight")],
            dtype=float,
        )

        avg_degree = float(np.mean(degrees)) if len(degrees) else 0.0
        median_degree = float(np.median(degrees)) if len(degrees) else 0.0
        min_degree = float(np.min(degrees)) if len(degrees) else 0.0
        max_degree = float(np.max(degrees)) if len(degrees) else 0.0
        degree_std = float(np.std(degrees)) if len(degrees) else 0.0
    else:
        degrees = binary_degrees
        avg_degree = binary_avg_degree
        median_degree = binary_median_degree
        min_degree = binary_min_degree
        max_degree = binary_max_degree
        degree_std = binary_degree_std

    # ============================================================
    # Density / clustering / triangles
    # ============================================================

    density = float(nx.density(G))

    if n_edges > 0:
        binary_clustering_coeff = float(nx.average_clustering(G, weight=None))

        clustering_coeff = float(
            nx.average_clustering(
                G,
                weight="weight" if is_weighted else None,
            )
        )

        triangles = float(sum(nx.triangles(G).values()) / 3)
    else:
        binary_clustering_coeff = 0.0
        clustering_coeff = 0.0
        triangles = 0.0

    # ============================================================
    # Connected components
    # ============================================================

    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]

    n_components = len(components)
    largest_component_size = max(component_sizes, default=0)

    connectivity_ratio = largest_component_size / n_nodes if n_nodes > 0 else 0.0

    if component_sizes:
        component_size_mean = float(np.mean(component_sizes))
        component_size_median = float(np.median(component_sizes))
        component_size_std = float(np.std(component_sizes))
        component_size_min = int(np.min(component_sizes))
        component_size_max = int(np.max(component_sizes))
    else:
        component_size_mean = 0.0
        component_size_median = 0.0
        component_size_std = 0.0
        component_size_min = 0
        component_size_max = 0

    # ============================================================
    # Edge weights / strengths
    # ============================================================

    if is_weighted and n_edges > 0:
        edge_weights = np.asarray(
            [d["weight"] for _, _, d in G.edges(data=True)],
            dtype=float,
        )

        avg_edge_weight = float(np.mean(edge_weights))
        median_edge_weight = float(np.median(edge_weights))
        min_edge_weight = float(np.min(edge_weights))
        max_edge_weight = float(np.max(edge_weights))
        edge_weight_std = float(np.std(edge_weights))

        strengths = np.asarray(
            [G.degree(weight="weight")[n] for n in G.nodes()],
            dtype=float,
        )

        avg_strength = float(np.mean(strengths))
        median_strength = float(np.median(strengths))
        min_strength = float(np.min(strengths))
        max_strength = float(np.max(strengths))
        strength_std = float(np.std(strengths))
    else:
        avg_edge_weight = np.nan
        median_edge_weight = np.nan
        min_edge_weight = np.nan
        max_edge_weight = np.nan
        edge_weight_std = np.nan

        avg_strength = np.nan
        median_strength = np.nan
        min_strength = np.nan
        max_strength = np.nan
        strength_std = np.nan

    # ============================================================
    # Cheap graph statistics
    # ============================================================

    results = {
        # Identity / basic
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": density,
        "is_weighted": is_weighted,
        # Binary degree
        "binary_avg_degree": binary_avg_degree,
        "binary_median_degree": binary_median_degree,
        "binary_min_degree": binary_min_degree,
        "binary_max_degree": binary_max_degree,
        "binary_degree_std": binary_degree_std,
        # Weighted degree / degree
        "avg_degree": avg_degree,
        "median_degree": median_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "degree_std": degree_std,
        # Clustering
        "binary_clustering_coefficient": binary_clustering_coeff,
        "clustering_coefficient": clustering_coeff,
        "triangle_count": triangles,
        # Edge weights
        "avg_edge_weight": avg_edge_weight,
        "median_edge_weight": median_edge_weight,
        "min_edge_weight": min_edge_weight,
        "max_edge_weight": max_edge_weight,
        "edge_weight_std": edge_weight_std,
        # Strength
        "avg_strength": avg_strength,
        "median_strength": median_strength,
        "min_strength": min_strength,
        "max_strength": max_strength,
        "strength_std": strength_std,
        # Components
        "component_count": n_components,
        "largest_component_size": largest_component_size,
        "connectivity_ratio": connectivity_ratio,
        "component_size_mean": component_size_mean,
        "component_size_median": component_size_median,
        "component_size_std": component_size_std,
        "component_size_min": component_size_min,
        "component_size_max": component_size_max,
        "component_sizes": component_sizes,
    }

    # ============================================================
    # Save exact adjacency matrix
    # ============================================================
    np.save(
        f"{save_path}_adjacency.npy",
        adj,
    )

    # ============================================================
    # Save explicit edge identities
    # ============================================================

    edges = []

    for u, v, data in G.edges(data=True):
        edge: dict[str, Any] = {
            "source": int(u),
            "target": int(v),
        }

        if is_weighted:
            edge["weight"] = float(data["weight"])

        edges.append(edge)

    if is_weighted:
        np.save(
            f"{save_path}_edges.npy",
            np.asarray(
                [(u, v, data["weight"]) for u, v, data in G.edges(data=True)],
                dtype=np.float64,
            ),
        )
    else:
        np.save(
            f"{save_path}_edges.npy",
            np.asarray([(u, v) for u, v in G.edges()], dtype=np.int64),
        )

    with open(f"{save_path}_edges.json", "w") as f:
        json.dump(edges, f, indent=4)

    # ============================================================
    # Save cheap statistics
    # ============================================================

    with open(f"{save_path}.json", "w") as f:
        json.dump(_sanitize_for_json(results), f, indent=4)

    print(f"Metrics saved to {save_path}.json")
    print(f"Adjacency saved to {save_path}_adjacency.npy")
    print(f"Edges saved to {save_path}_edges.json")


def _small_world_metrics_gnm(
    graph: nx.Graph,
    n_randomizations: int = 20,
) -> tuple[float, float, float, float]:
    """Small-world reference using G(n,m) random graphs."""

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if n <= 10 or m == 0:
        return np.nan, np.nan, np.nan, np.nan

    clustering_values = []
    path_values = []

    for _ in range(n_randomizations):
        random_graph = nx.gnm_random_graph(n, m)

        clustering_values.append(nx.average_clustering(random_graph))

        if nx.is_connected(random_graph):
            path_values.append(nx.average_shortest_path_length(random_graph))

    if not clustering_values or not path_values:
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)

    return (
        float(np.mean(clustering_values)),
        float(np.std(clustering_values)),
        float(np.mean(path_values)),
        float(np.std(path_values)),
    )


def _small_world_metrics_degree_preserving(
    graph: nx.Graph,
    n_randomizations: int = 20,
    swap_factor: int = 10,
    max_tries_factor: int = 100,
) -> tuple[float, float, float, float]:
    """Small-world reference using degree-preserving randomization."""

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if n <= 10 or m == 0:
        return np.nan, np.nan, np.nan, np.nan

    clustering_values = []
    path_values = []

    nswap = swap_factor * m
    max_tries = max_tries_factor * m

    for _ in range(n_randomizations):
        random_graph = graph.copy()

        try:
            nx.double_edge_swap(
                random_graph,
                nswap=nswap,
                max_tries=max_tries,
            )
        except nx.NetworkXAlgorithmError:
            continue

        clustering_values.append(nx.average_clustering(random_graph))

        if nx.is_connected(random_graph):
            path_values.append(nx.average_shortest_path_length(random_graph))

    if not clustering_values or not path_values:
        return np.nan, np.nan, np.nan, np.nan

    return (
        np.mean(clustering_values),
        np.std(clustering_values),
        np.mean(path_values),
        np.std(path_values),
    )


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        obj = float(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj

    return str(obj)


def debug_graph_characteristics(adjacency_matrix: torch.Tensor) -> None:
    adj = adjacency_matrix.detach().cpu().numpy()

    is_weighted = not np.array_equal(adj, adj.astype(bool).astype(float))

    G = nx.from_numpy_array(adj)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Basic stats, unweighted
    binary_degrees = [d for _, d in G.degree(weight=None)]
    binary_avg_degree = np.mean(binary_degrees) if binary_degrees != [] else 0
    binary_median_degree = np.median(binary_degrees) if binary_degrees != [] else 0
    binary_max_degree = max(binary_degrees) if binary_degrees != [] else 0
    binary_min_degree = min(binary_degrees) if binary_degrees != [] else 0
    binary_degree_std = np.std(binary_degrees) if binary_degrees != [] else 0

    if is_weighted:
        # If weighted graph, weighted basic stats
        # Basic stats, unweighted
        degrees = [d for _, d in G.degree(weight="weight")]
        avg_degree = np.mean(degrees) if degrees != [] else 0
        median_degree = np.median(degrees) if degrees != [] else 0
        max_degree = max(degrees) if degrees != [] else 0
        min_degree = min(degrees) if degrees != [] else 0
        degree_std = np.std(degrees) if degrees != [] else 0
    else:
        # Basic stats, unweighted
        degrees = binary_degrees
        avg_degree = binary_avg_degree
        median_degree = binary_median_degree
        max_degree = binary_max_degree
        min_degree = binary_min_degree
        degree_std = binary_degree_std

    print("DEBUG Graph characteristics:")
    print(f"Number of nodes: {n_nodes}; number of edges: {n_edges}\nAverage degree: {avg_degree}, min deg {min_degree}, max deg {max_degree}")
