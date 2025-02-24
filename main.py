from argparse import ArgumentParser, Namespace

import networkx as nx
from numpy import mean

from datasets.dataloader import get_dataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="which dataset to use e.g. 'synthetic'",
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        help="which algorithm to use e.g. 'KNN'",
        default=None,
    )
    args = parser.parse_args()
    return args


def graph_characteristics(adj):
    print(adj.numpy().shape)
    G = nx.from_numpy_array(adj.numpy())
    print(type(G.degree()))
    degrees = [d for _, d in G.degree()]
    clustering_coeff = nx.average_clustering(G)
    n_component = nx.number_connected_components(G)
    largest_component = max(nx.connected_components(G), key=len)
    connectivity = len(largest_component) / G.number_of_nodes()
    print(
        f"Degrees: {mean(degrees)} | Clustering coefficient: {clustering_coeff} | Number Components: {n_component} | Connectivity: {connectivity}"
    )


def run(args: Namespace) -> None:
    dataset = get_dataset(args.dataset)
    dataset.corrupt()
    _ = dataset.get_dataloader(shuffle=False, batch_size=8)
    adj_matrix = dataset.get_adjacency()
    adj_matrix_knn = dataset.get_similarity_knn(k=5)
    # print(adj_matrix.shape)
    # print(adj_matrix_knn.shape)
    # print(dataset.shape())
    graph_characteristics(adj_matrix)
    graph_characteristics(adj_matrix_knn)


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN based method from MPIN
