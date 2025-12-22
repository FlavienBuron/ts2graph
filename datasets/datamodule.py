from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, Subset

from datasets.dataloaders.graphloader import GraphLoader
from datasets.scalers.abstract_scaler import AbstractScaler
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
        print(
            f"DEBUG: Is LightningDataModule? {isinstance(self, pl.LightningDataModule)}"
        )
        print(f"DEBUG: MRO: {self.__class__.__mro__}")
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
            print(
                f"{scaling_axes=} {len(self.train_slice)=} {train.shape=} {train_mask.shape=} {self.dataset._mask.shape=} {self.dataset.mask.shape=}"
            )
            scaler = self.get_scaler()(axis=scaling_axes)
            print(
                f"{type(train)=} {type(train_mask)=} {type(self.dataset.mask)=} {type(self.dataset.data)=}"
            )
            scaler.fit(x=train, mask=train_mask, keepdims=True)
            self.scaler = scaler.to_torch()

            if len(self.scale_exogenous) > 0:
                for label in self.scale_exogenous:
                    exo = getattr(self, label)
                    scaler = self.get_scaler()(axis=scaling_axes)
                    scaler.fit(exo[self.train_slice], keepdims=True)
                    scaler = scaler.to_torch()
                    setattr(self, label, scaler.transform(exo))
        print("DEBUG: DataModule successfully initialized")

    @property
    def n_nodes(self):
        print(f"DEBUG: DM n_nodes: {self.dataset.shape[1]} shape={self.dataset.shape}")
        return self.dataset.shape[1]

    @property
    def d_in(self):  # changed from n_channels
        print(f"DEBUG: DM d_in: {self.dataset.shape[-1]} shape={self.dataset.shape}")
        return self.dataset.shape[-1]

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

    def train_dataloader(self, *args, **kwargs):
        print("DEBUG: train_dataloader method is being called!")
        rnd_sampler = None
        shuffle = True
        print(f"train_loader: {len(self.train_set)=} {shuffle=} {self.batch_size=}")
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
        )

    def val_dataloader(self, shuffle: bool = False):
        print(f"val_dataloader: {shuffle=} {self.batch_size=} {len(self.val_set)=}")
        return DataLoader(
            dataset=self.val_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
        )

    def test_dataloader(self, shuffle: bool = False):
        print(f"test_dataloader: {shuffle=} {self.batch_size=} {len(self.test_set)=}")
        return DataLoader(
            dataset=self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
        )

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

    def get_scaler(self) -> type[AbstractScaler]:
        if self.scaling_type == "std":
            return StandardScaler
        elif self.scaling_type == "minmax":
            return MinMaxScaler
        else:
            raise NotImplementedError
