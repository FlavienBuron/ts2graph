from datasets.dataloaders.air_quality import AirQualityLoader
from datasets.dataloaders.graphloader import GraphLoader
from datasets.dataloaders.synthetic import SyntheticLoader


def get_dataset(dataset_name: str, normalization_type: str = "min_max") -> GraphLoader:
    if dataset_name == "synthetic":
        return SyntheticLoader()
    if "air" in dataset_name:
        small = "small" in dataset_name
        return AirQualityLoader(small=small, normalization_type=normalization_type)
    else:
        return SyntheticLoader()
