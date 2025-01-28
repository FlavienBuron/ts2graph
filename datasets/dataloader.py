from datasets.synthetic import SyntheticDataset


def get_dataset(dataset_name: str) -> SyntheticDataset:
    if dataset_name == "synthetic":
        return SyntheticDataset()
    else:
        return SyntheticDataset()
