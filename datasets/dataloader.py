from typing import Union

from datasets.air_quality import AirQualityDataset
from datasets.synthetic import SyntheticDataset


def get_dataset(dataset_name: str) -> Union[SyntheticDataset, AirQualityDataset]:
    if dataset_name == "synthetic":
        return SyntheticDataset()
    if "air" in dataset_name:
        small = False
        if "small" in dataset_name:
            small = True
        return AirQualityDataset(small=small)
    else:
        return SyntheticDataset()
