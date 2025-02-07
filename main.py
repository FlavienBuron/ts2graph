from argparse import ArgumentParser, Namespace

from torch.utils.data import DataLoader

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
    dataloader: DataLoader = dataset.get_dataloader()
    adj_matrix = dataset.get_similarity()
    print(adj_matrix.shape)
    _ = dataloader


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN based method from MPIN
