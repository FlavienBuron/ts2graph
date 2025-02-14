from argparse import ArgumentParser, Namespace

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


def run(args: Namespace) -> None:
    dataset = get_dataset(args.dataset)
    dataloader = dataset.get_dataloader()
    adj_matrix = dataset.get_adjacency()
    adj_matrix_knn = dataset.get_similarity_knn(k=3)
    print(adj_matrix.shape)
    print(adj_matrix_knn.shape)
    print(dataset.shape())


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN based method from MPIN
