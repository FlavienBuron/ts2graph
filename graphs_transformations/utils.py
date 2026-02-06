import json

import networkx as nx
import numpy as np
import torch
from community import community_louvain
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

    dists = (
        torch.cdist(data, data, p=2) if not cosine else 1 - torch.matmul(data, data.T)
    )
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
    return torch.stack(
        ([x[i : i + N] for i in range(0, dim * time_delay, time_delay)]), dim=1
    )


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


def compute_laplacian_smoothness(
    x, edge_index, edge_weight=None, mask=None, normalize=True, debug=False
):
    batch_size, nodes, features = x.shape
    lap_edge_index, lap_edge_weight = get_laplacian(
        edge_index, edge_weight, normalization="sym"
    )
    laplacian = to_dense_adj(
        lap_edge_index, edge_attr=lap_edge_weight, max_num_nodes=nodes
    ).squeeze(0)

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

    x_flat = (
        x.permute(0, 2, 1).reshape(batch_size * features, nodes).unsqueeze(1)
    )  # [B*F, 1, N]
    laplacian_expanded = laplacian.unsqueeze(0).expand(
        batch_size * features, -1, -1
    )  # [B*F, N, N]
    smoothness = torch.bmm(
        torch.bmm(x_flat, laplacian_expanded), x_flat.transpose(1, 2)
    ).squeeze()

    smoothness_total = smoothness.sum()

    if normalize:
        energy = torch.sum(x**2) + 1e-8
        return (smoothness_total / energy).item()
    else:
        return smoothness_total.item()


def compute_edge_difference_smoothness(
    x, edge_index, edge_weight=None, mask=None, normalize=True
):
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


def save_graph_characteristics(adjacency_matrix: torch.Tensor, save_path: str) -> None:
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

    # Density and Clustering
    density = nx.density(G)
    clustering_coeff = (
        nx.average_clustering(G, weight="weight" if is_weighted else None)
        if G.number_of_edges() != 0
        else 0
    )
    binary_clustering_coeff = (
        nx.average_clustering(G, weight=None) if G.number_of_edges() != 0 else 0
    )

    # Triangle counts
    try:
        triangles = (
            sum(nx.triangles(G).values()) / 3
        )  # divide by 3 as each triangle is counted 3 times (each node)
    except:
        triangles = 0.0

    # Component analysis
    components = list(nx.connected_components(G))
    n_components = len(components)
    component_sizes = [len(c) for c in components]

    components_sorted = sorted(components, key=len, reverse=True)

    if n_components > 1:
        component_size_mean = np.mean(component_sizes)
        component_size_std = np.std(component_sizes)
        component_size_min = np.min(component_sizes)
        component_size_max = np.max(component_sizes)
        component_size_median = np.median(component_sizes)
    else:
        component_size_mean = component_size_std = component_size_min = (
            component_size_max
        ) = component_size_median = n_nodes

        # Distance metrics within components
    avg_path_lengths = []
    diameters = []
    radiuses = []
    eccentricities = []

    component_metrics = []

    for i, component in enumerate(components_sorted):
        comp_size = len(component)

        if comp_size <= 1:
            continue

        subgraph = G.subgraph(component).copy()

        # Component specific stats
        comp_metrics = {
            "component_id": i,
            "size": comp_size,
            "size_ratio": comp_size / n_nodes,
            "edge_count": subgraph.number_of_edges(),
            "density": nx.density(subgraph),
        }

        comp_metrics["binary_clustering_coeff"] = nx.average_clustering(
            subgraph, weight=None
        )

        if is_weighted:
            comp_metrics["clustering_coeff"] = nx.average_clustering(
                subgraph, weight="weight"
            )
        else:
            comp_metrics["clustering_coeff"] = comp_metrics["binary_clustering_coeff"]

        # Try to compute path-based metrics
        try:
            comp_metrics["binary_avg_path_length"] = nx.average_shortest_path_length(
                subgraph
            )
            comp_metrics["binary_diameter"] = nx.diameter(subgraph)
            comp_metrics["binary_radius"] = nx.radius(subgraph)
            binary_eccentricities = list(nx.eccentricity(subgraph).values())
            comp_metrics["binary_avg_eccentricity"] = np.mean(binary_eccentricities)

            inv_weight_graph = nx.Graph()

            # For weighted graphs also calculate weighted path metrics
            if is_weighted:
                # Create copy with inverted weights for path calc.
                # (higher weights = stronger connection = shorter path)
                inv_weight_graph.add_nodes_from(subgraph.nodes())
                for u, v, d in subgraph.edges(data=True):
                    weight = d["weight"]
                    if weight > 0:
                        inv_weight = 1.0 / weight
                    else:
                        inv_weight = float("inf")
                    inv_weight_graph.add_edge(u, v, weight=inv_weight)

                comp_metrics["avg_path_length"] = nx.average_shortest_path_length(
                    inv_weight_graph, weight="weight"
                )
                comp_metrics["diameter"] = nx.diameter(
                    inv_weight_graph, weight="weight"
                )
                comp_metrics["radius"] = nx.radius(inv_weight_graph, weight="weight")
                weighted_eccentricities = list(
                    nx.eccentricity(inv_weight_graph, weight="weight").values()
                )
                comp_metrics["avg_eccentricity"] = np.mean(weighted_eccentricities)

                # Use weighted metrics for global stats
                avg_path_lengths.append(comp_metrics["avg_path_length"])
                diameters.append(comp_metrics["diameter"])
                radiuses.append(comp_metrics["radius"])
                eccentricities.extend(weighted_eccentricities)
            else:
                # For unweighted graphs, set weighted metrics to binary metrics
                comp_metrics["avg_path_length"] = comp_metrics["binary_avg_path_length"]
                comp_metrics["diameter"] = comp_metrics["binary_diameter"]
                comp_metrics["radius"] = comp_metrics["binary_radius"]
                comp_metrics["avg_eccentricity"] = comp_metrics[
                    "binary_avg_eccentricity"
                ]

                # Use unweighted metrics for global stats
                avg_path_lengths.append(comp_metrics["binary_avg_path_length"])
                diameters.append(comp_metrics["binary_diameter"])
                radiuses.append(comp_metrics["binary_radius"])
                eccentricities.extend(binary_eccentricities)

            try:
                # Always calc. unweighted Centrality measures
                comp_metrics["binary_degree_centrality"] = np.mean(
                    list(nx.degree_centrality(subgraph).values())
                )
                comp_metrics["binary_closeness_centrality"] = np.mean(
                    list(nx.closeness_centrality(subgraph).values())
                )

                # Only compute Betweeness for reasonably sized components
                if comp_size <= 1000:
                    comp_metrics["binary_betweeness_centrality"] = np.mean(
                        list(nx.betweenness_centrality(subgraph, weight=None).values())
                    )
                else:
                    comp_metrics["binary_betweeness_centrality"] = float("nan")

                if is_weighted:
                    comp_metrics["degree_centrality"] = comp_metrics[
                        "binary_degree_centrality"
                    ]  # Degree centrality ignore weights
                    comp_metrics["closeness_centrality"] = np.mean(
                        list(
                            nx.closeness_centrality(
                                inv_weight_graph, distance="weight"
                            ).values()
                        )
                    )

                    # Only compute Betweeness for reasonably sized components
                    if comp_size <= 1000:
                        comp_metrics["betweeness_centrality"] = np.mean(
                            list(
                                nx.betweenness_centrality(
                                    inv_weight_graph, weight="weight"
                                ).values()
                            )
                        )
                    else:
                        comp_metrics["betweeness_centrality"] = float("nan")
                else:
                    comp_metrics["degree_centrality"] = comp_metrics[
                        "binary_degree_centrality"
                    ]
                    comp_metrics["closeness_centrality"] = comp_metrics[
                        "binary_closeness_centrality"
                    ]
                    comp_metrics["betweeness_centrality"] = comp_metrics[
                        "binary_betweeness_centrality"
                    ]
            except:
                comp_metrics["binary_degree_centrality"] = float("nan")
                comp_metrics["binary_closeness_centrality"] = float("nan")
                comp_metrics["binary_betweeness_centrality"] = float("nan")
                comp_metrics["degree_centrality"] = float("nan")
                comp_metrics["closeness_centrality"] = float("nan")
                comp_metrics["betweeness_centrality"] = float("nan")

            # Community detection
            try:
                binary_partition = community_louvain.best_partition(
                    subgraph, weight=None
                )
                comp_metrics["binary_num_communities"] = len(
                    set(binary_partition.values())
                )
                comp_metrics["binary_modularity"] = community_louvain.modularity(
                    binary_partition, subgraph, weight=None
                )

                if is_weighted:
                    weighted_partition = community_louvain.best_partition(
                        subgraph, weight="weight"
                    )
                    comp_metrics["num_communities"] = len(
                        set(weighted_partition.values())
                    )
                    comp_metrics["modularity"] = community_louvain.modularity(
                        weighted_partition, subgraph, weight="weight"
                    )
                else:
                    comp_metrics["num_communities"] = comp_metrics[
                        "binary_num_communities"
                    ]
                    comp_metrics["modularity"] = comp_metrics["binary_modularity"]
            except:
                comp_metrics["binary_num_communities"] = float("nan")
                comp_metrics["binary_modularity"] = float("nan")
                comp_metrics["num_communities"] = float("nan")
                comp_metrics["modularity"] = float("nan")

            # Assortativity
            try:
                degrees = [d for _, d in subgraph.degree()]
                if len(degrees) < 2 or np.std(degrees) == 0:
                    raise ValueError("Degenerate degree distribution")

                comp_metrics["binary_degree_assortativity"] = (
                    nx.degree_assortativity_coefficient(subgraph, weight=None)
                )

                if is_weighted:
                    comp_metrics["degree_assortativity"] = (
                        nx.degree_assortativity_coefficient(subgraph, weight="weight")
                    )
                else:
                    comp_metrics["degree_assortativity"] = comp_metrics[
                        "binary_degree_assortativity"
                    ]
            except:
                comp_metrics["binary_degree_assortativity"] = float("nan")
                comp_metrics["degree_assortativity"] = float("nan")

            # Spectral properties
            try:
                binary_laplacian = nx.normalized_laplacian_matrix(
                    subgraph, weight=None
                ).todense()
                binary_eigenvalues = np.linalg.eigvalsh(binary_laplacian)
                comp_metrics["binary_spectral_gap"] = (
                    binary_eigenvalues[1] if len(binary_eigenvalues) > 1 else 0
                )
                comp_metrics["binary_algebraic_connectivity"] = comp_metrics[
                    "binary_spectral_gap"
                ]
                comp_metrics["binary_largest_eigenvalue"] = (
                    binary_eigenvalues[-1] if len(binary_eigenvalues) > 0 else 0
                )
                comp_metrics["binary_graph_energy"] = np.sum(np.abs(binary_eigenvalues))

                if is_weighted:
                    weighted_laplacian = nx.normalized_laplacian_matrix(
                        subgraph, weight="weight"
                    ).todense()
                    weighted_eigenvalues = np.linalg.eigvalsh(weighted_laplacian)
                    comp_metrics["spectral_gap"] = (
                        weighted_eigenvalues[1] if len(weighted_eigenvalues) > 1 else 0
                    )
                    comp_metrics["algebraic_connectivity"] = comp_metrics[
                        "spectral_gap"
                    ]
                    comp_metrics["largest_eigenvalue"] = (
                        weighted_eigenvalues[-1] if len(weighted_eigenvalues) > 0 else 0
                    )
                    comp_metrics["graph_energy"] = np.sum(np.abs(weighted_eigenvalues))
                else:
                    comp_metrics["spectral_gap"] = comp_metrics["binary_spectral_gap"]
                    comp_metrics["algebraic_connectivity"] = comp_metrics[
                        "binary_algebraic_connectivity"
                    ]
                    comp_metrics["largest_eigenvalue"] = comp_metrics[
                        "binary_largest_eigenvalue"
                    ]
                    comp_metrics["graph_energy"] = comp_metrics["binary_graph_energy"]
            except:
                comp_metrics["binary_spectral_gap"] = float("nan")
                comp_metrics["binary_algebraic_connectivity"] = float("nan")
                comp_metrics["binary_largest_eigenvalue"] = float("nan")
                comp_metrics["binary_graph_energy"] = float("nan")
                comp_metrics["spectral_gap"] = float("nan")
                comp_metrics["algebraic_connectivity"] = float("nan")
                comp_metrics["largest_eigenvalue"] = float("nan")
                comp_metrics["graph_energy"] = float("nan")

            # Small-world properties
            try:
                if comp_size > 10:
                    # Generate random graph with same number of nodes and edges
                    random_graph = nx.gnm_random_graph(
                        comp_size, subgraph.number_of_edges()
                    )

                    random_clustering = nx.average_clustering(random_graph)

                    if nx.is_connected(random_graph):
                        random_path_length = nx.average_shortest_path_length(
                            random_graph
                        )

                        # Calculate binary small-world coefficient
                        if random_clustering > 0 and random_path_length > 0:
                            binary_small_world_coef = (
                                comp_metrics["binary_clustering_coeff"]
                                / random_clustering
                            ) / (
                                comp_metrics["binary_avg_path_length"]
                                / random_path_length
                            )
                            comp_metrics["binary_small_world_coefficient"] = (
                                binary_small_world_coef
                            )
                        else:
                            comp_metrics["binary_small_world_coefficient"] = float(
                                "nan"
                            )

                        if is_weighted:
                            if random_clustering > 0 and random_path_length > 0:
                                small_world_coef = (
                                    comp_metrics["clustering_coeff"] / random_clustering
                                ) / (
                                    comp_metrics["avg_path_length"] / random_path_length
                                )
                                comp_metrics["small_world_coefficient"] = (
                                    small_world_coef
                                )
                            else:
                                comp_metrics["small_world_coefficient"] = float("nan")
                        else:
                            comp_metrics["small_world_coefficient"] = comp_metrics[
                                "binary_small_world_coefficient"
                            ]
                    else:
                        comp_metrics["binary_small_world_coefficient"] = float("nan")
                        comp_metrics["small_world_coefficient"] = float("nan")
                else:
                    comp_metrics["binary_small_world_coefficient"] = float("nan")
                    comp_metrics["small_world_coefficient"] = float("nan")
            except:
                comp_metrics["binary_small_world_coefficient"] = float("nan")
                comp_metrics["small_world_coefficient"] = float("nan")

            # For larger components, compute some additional metrics
            if comp_size >= 100:
                # Edge weight stats if weighted
                if is_weighted:
                    edge_weights = [
                        d["weight"] for _, _, d in subgraph.edges(data=True)
                    ]
                    comp_metrics["avg_edge_weight"] = np.mean(edge_weights)
                    comp_metrics["median_edge_weight"] = np.median(edge_weights)
                    comp_metrics["min_edge_weight"] = np.min(edge_weights)
                    comp_metrics["max_edge_weight"] = np.max(edge_weights)
                    comp_metrics["median_edge_weight"] = np.std(edge_weights)

                # Try to compute eigenvector centrality
                try:
                    comp_metrics["eigenvector_centrality"] = np.mean(
                        list(
                            nx.eigenvector_centrality(
                                subgraph, weight="weight" if is_weighted else None
                            ).values()
                        )
                    )
                except:
                    comp_metrics["eigenvector_centrality"] = float("nan")
        except nx.NetworkXError:
            # If metrics can't be computed, mark as NaN
            comp_metrics["avg_path_length"] = float("inf")
            comp_metrics["diameter"] = float("inf")
            comp_metrics["radius"] = float("inf")
            comp_metrics["avg_eccentricity"] = float("inf")

    avg_path_length = np.mean(avg_path_lengths) if avg_path_lengths else float("inf")
    avg_diameter = np.mean(diameters) if diameters else float("inf")
    avg_radius = np.mean(radiuses) if radiuses else float("inf")
    avg_eccentricity = np.mean(eccentricities) if eccentricities else float("inf")

    # Largest component metrics
    largest_component_size = component_sizes[0] if component_sizes else 0
    connectivity_ratio = largest_component_size / n_nodes if n_nodes > 0 else 0

    if is_weighted:
        edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
        avg_edge_weight = np.mean(edge_weights)
        median_edge_weight = np.median(edge_weights)
        min_edge_weight = np.min(edge_weights) if edge_weights else 0
        max_edge_weight = np.max(edge_weights) if edge_weights else 0
        edge_weight_std = np.std(edge_weights) if edge_weights else 0

        strengths = [G.degree(weight="weight")[n] for n in G.nodes()]
        avg_strength = np.mean(strengths)
        median_strength = np.median(strengths)
        min_strength = np.min(strengths)
        max_strength = np.max(strengths)
        strength_std = np.std(strengths)
    else:
        avg_edge_weight = median_edge_weight = min_edge_weight = max_edge_weight = (
            edge_weight_std
        ) = float("nan")
        avg_strength = median_strength = min_strength = max_strength = strength_std = (
            float("nan")
        )

    # Glabal binary path metrics
    if is_weighted:
        binary_avg_path_lengths = []
        binary_diameters = []
        binary_radiuses = []
        binary_eccentricities = []

        for component in components:
            if len(component) > 1:
                subgraph = G.subgraph(component).copy()
                try:
                    binary_avg_path_lengths.append(
                        nx.average_shortest_path_length(subgraph)
                    )
                    binary_diameters.append(nx.diameter(subgraph))
                    binary_radiuses.append(nx.radius(subgraph))
                    binary_eccentricities.extend(
                        list(nx.eccentricity(subgraph).values())
                    )
                except nx.NetworkXError:
                    pass

        binary_avg_path_length = (
            np.mean(binary_avg_path_lengths)
            if binary_avg_path_lengths
            else float("inf")
        )
        binary_avg_diameter = (
            np.mean(binary_diameters) if binary_diameters else float("inf")
        )
        binary_avg_radius = (
            np.mean(binary_radiuses) if binary_radiuses else float("inf")
        )
        binary_avg_eccentricity = (
            np.mean(binary_eccentricities) if binary_eccentricities else float("inf")
        )
    else:
        # For unweighted graphs, binary metrics are the same as normal metrics
        binary_avg_path_length = avg_path_length
        binary_avg_diameter = avg_diameter
        binary_avg_radius = avg_radius
        binary_avg_eccentricity = avg_eccentricity

    # Create results dictionary
    results = {
        # Basic statistics
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": density,
        "is_weighted": is_weighted,
        # Degree metrics (both weighted and unweighted)
        "avg_degree": avg_degree,
        "median_degree": median_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "degree_std": degree_std,
        "binary_avg_degree": binary_avg_degree,
        "binary_median_degree": binary_median_degree,
        "binary_min_degree": binary_min_degree,
        "binary_max_degree": binary_max_degree,
        "binary_degree_std": binary_degree_std,
        # Clustering metrics (both weighted and unweighted)
        "clustering_coefficient": clustering_coeff,
        "binary_clustering_coefficient": binary_clustering_coeff,
        "triangle_count": triangles,
        # Edge weight metrics for weighted graphs
        "avg_edge_weight": avg_edge_weight,
        "median_edge_weight": median_edge_weight,
        "min_edge_weight": min_edge_weight,
        "max_edge_weight": max_edge_weight,
        "edge_weight_std": edge_weight_std,
        # Strength metrics for weighted graphs
        "avg_strength": avg_strength,
        "median_strength": median_strength,
        "min_strength": min_strength,
        "max_strength": max_strength,
        "strength_std": strength_std,
        # Component analysis
        "component_count": n_components,
        "largest_component_size": largest_component_size,
        "connectivity_ratio": connectivity_ratio,
        "component_size_mean": component_size_mean,
        "component_size_median": component_size_median,
        "component_size_std": component_size_std,
        "component_size_min": component_size_min,
        "component_size_max": component_size_max,
        # Distance metrics (both weighted and unweighted)
        "avg_path_length": avg_path_length,
        "avg_diameter": avg_diameter,
        "avg_radius": avg_radius,
        "avg_eccentricity": avg_eccentricity,
        "binary_avg_path_length": binary_avg_path_length,
        "binary_avg_diameter": binary_avg_diameter,
        "binary_avg_radius": binary_avg_radius,
        "binary_avg_eccentricity": binary_avg_eccentricity,
        # Keep original component sizes for detailed analysis
        "component_sizes": component_sizes,
        # All component metrics
        "component_metrics": component_metrics,
    }

    with open(f"{save_path}.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Metrics saved to {save_path}.json")
