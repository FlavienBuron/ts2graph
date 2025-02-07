from typing import Union

from datasets.dataloaders.air_quality import AirQualityLoader
from datasets.dataloaders.synthetic import SyntheticLoader


def get_dataset(dataset_name: str) -> Union[SyntheticLoader, AirQualityLoader]:
    if dataset_name == "synthetic":
        return SyntheticLoader()
    if "air" in dataset_name:
        small = False
        if "small" in dataset_name:
            small = True
        return AirQualityLoader(small=small)
    else:
        return SyntheticLoader()
