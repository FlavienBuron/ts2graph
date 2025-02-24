from datasets.dataloaders.air_quality import AirQualityLoader
from datasets.dataloaders.graphloader import GraphLoader
from datasets.dataloaders.synthetic import SyntheticLoader


def get_dataset(dataset_name: str) -> GraphLoader:
    if dataset_name == "synthetic":
        return SyntheticLoader()
    if "air" in dataset_name:
        small = "small" in dataset_name
        return AirQualityLoader(small=small)
    else:
        return SyntheticLoader()
