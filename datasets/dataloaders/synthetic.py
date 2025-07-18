import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataloaders.graphloader import GraphLoader


class SyntheticLoader(GraphLoader):
    def __init__(
        self,
        dataset_path: str = "./datasets/data/synthetic/",
        static_adj: bool = False,
        window: Optional[int] = None,
        use_exogenous: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.static_adj = static_adj

        if static_adj:
            dataset_path = os.path.join(self.dataset_path, "charged_static.npz")
        else:
            dataset_path = os.path.join(self.dataset_path, "charged_varying.npz")
        data = np.load(dataset_path)
        self.use_exogenous: bool = use_exogenous
        self.window: int = window if window is not None else data["loc"].shape[1]
        self.loc: torch.Tensor = torch.tensor(data["loc"][:, : self.window]).float()
        self.vel: torch.Tensor = torch.tensor(data["vel"][:, : self.window]).float()
        self.adj: torch.Tensor = torch.tensor(data["adj"]).float()

    def __len__(self) -> int:
        return self.loc.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        x: torch.Tensor = self.loc[index]
        result: Dict[str, torch.Tensor] = {"x": x}
        if self.use_exogenous:
            u: torch.Tensor = self.vel[index]
            result.update(u=u)
        return result

    # def load(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     filename = "charged_static.npz" if self.static_adj else "charged_varying.npz"
    #     data = np.load(os.path.join(self.dataset_path, filename))

    def get_adjacency(self) -> torch.Tensor:
        return self.adj

    def corrupt(self, missing_type="perc"):
        pass

    def get_geolocation_graph(
        self,
        threshold: float = 0.1,
        threshold_on_input: bool = False,
        include_self: bool = False,
        force_symmetric: bool = False,
    ):
        pass

    def get_knn_graph(
        self,
        k: int,
        use_corrupted_data: bool = False,
        loop: bool = False,
        cosine: bool = False,
    ):
        pass

    def get_radius_graph(
        self,
        radius: int,
        use_corrupted_data: bool = False,
        loop: bool = False,
        cosine: bool = False,
    ):
        pass

    @property
    def n_channels(self) -> int:
        return self.loc.size(-1)

    def n_nodes(self) -> int:
        return self.loc.size(-2)

    def n_exogenous(self) -> int:
        return self.vel.size(-1) if self.use_exogenous else 0

    def get_dataloader(
        self, use_corrupted_data: bool, shuffle: bool = False, batch_size: int = 8
    ) -> DataLoader:
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)
