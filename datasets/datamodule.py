from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, Subset

from datasets.dataloaders.graphloader import GraphLoader
from datasets.scalers.abstract_scaler import AbstractScaler
from datasets.scalers.grin_scaler import StandardScaler as GSS
from datasets.scalers.min_max_scaler import MinMaxScaler
from datasets.scalers.standard_scaler import StandardScaler


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: GraphLoader,
        scale: bool = True,
        scaling_axis: str = "global",
        scaling_type: str = "minmax",
        scale_exogenous: List = [],
        train_indices: List = [],
        val_indices: List = [],
        test_indices: List = [],
        batch_size: int = 32,
        workers=21,
        samples_per_epoch: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self._has_setup_fit = True

        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)

        # preprocessing
        self.scale = scale
        self.scaling_type = scaling_type
        self.scaling_axis = scaling_axis
        self.scale_exogenous = scale_exogenous
        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.samples_per_epoch = samples_per_epoch

        if self.scale:
            scaling_axes = self.get_scaling_axes(self.scaling_axis)
            train = self.dataset.data[self.train_slice]
            train_mask = self.dataset.mask[self.train_slice]
            scaler = self.get_scaler(axis=scaling_axes)
            scaler.fit(x=train, mask=train_mask, keepdims=True)
            self.dataset.scaler = scaler.to_torch()

            if len(self.scale_exogenous) > 0:
                for label in self.scale_exogenous:
                    exo = getattr(self, label)
                    scaler = self.get_scaler()(axis=scaling_axes)
                    scaler.fit(exo[self.train_slice], keepdims=True)
                    scaler = scaler.to_torch()
                    setattr(self, label, scaler.transform(exo))

    @property
    def n_nodes(self):
        return self.dataset.n_nodes

    @property
    def d_in(self):  # changed from n_channels
        return self.dataset.n_channels

    @property
    def d_out(self):
        return self.dataset.horizon

    @property
    def train_slice(self):
        return self.dataset.expand_and_merge_indices(self.train_set.indices)

    @property
    def val_slice(self):
        return self.dataset.expand_and_merge_indices(self.val_set.indices)

    @property
    def test_slice(self):
        return self.dataset.expand_and_merge_indices(self.test_set.indices)

    def train_dataloader(self):
        rnd_sampler = None
        shuffle = True
        if self.samples_per_epoch > 0:
            shuffle = False

            rnd_sampler = RandomSampler(
                self.train_set,
                replacement=True,
                num_samples=self.samples_per_epoch,
            )

        return DataLoader(
            self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            sampler=rnd_sampler,
            drop_last=True,
            num_workers=self.workers,
        )

    def val_dataloader(self, shuffle: bool = False):
        return DataLoader(
            dataset=self.val_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def test_dataloader(self, shuffle: bool = False):
        return DataLoader(
            dataset=self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def get_scaling_axes(self, dim: str = "global"):
        scaling_axis = tuple()
        if dim == "global":
            scaling_axis = (0, 1, 2)
        elif dim == "channels":
            scaling_axis = (0, 1)
        elif dim == "nodes":
            scaling_axis = (0,)
        else:
            raise ValueError(f"Scaling axis '{dim}' is not valid")

        return scaling_axis

    def get_scaler(self, axis) -> AbstractScaler:
        if self.scaling_type == "std":
            return GSS(axis=axis)
        elif self.scaling_type == "minmax":
            return MinMaxScaler(axis=axis)
        else:
            StandardScaler()
            raise NotImplementedError
