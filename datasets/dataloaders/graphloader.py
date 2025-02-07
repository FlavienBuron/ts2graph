from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class GraphLoader(Dataset, ABC):
    def __init__(self) -> None:
        self.data = None
        self.distances = None

    @abstractmethod
    def get_similarity(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_dataloader(self, shuffle: bool, batch_size: int) -> Any:
        pass
