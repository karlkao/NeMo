# Copyright (c) 2019 NVIDIA Corporation
from typing import Optional, Dict

import torch
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import *


class FastSpeechDataset(torch.utils.data.Dataset):
    """

    This dataset should assumed particular file structure. Take a look at:
    https://ngc.nvidia.com/datasets/bSOCUeD5QHO0uyIGDtTphA

    """

    def __init__(self, mapping_json_file):
        """"""
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class FastSpeechDataLayer(DataLayerNM):
    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(
            audio=NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
            }),
            audio_length=NeuralType({
                0: AxisType(BatchTag),
            }),
            text=NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
            }),
            text_length=NeuralType({
                0: AxisType(BatchTag),
            }),
            alignment=NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(TimeTag),
            }),
        )

    def __init__(
            self,
            mapping_json,
            labels,
            bos_id=None,
            eos_id=None,
            pad_id=None,
            is_train=True,
            batch_size=32,
            num_workers=0,
    ):
        super().__init__()

        bos_id, eos_id, pad_id = len(labels), len(labels) + 1, len(labels) + 2
        self._dataset = FastSpeechDataset()

        # Set up data multi-GPU sampler.
        sampler = None
        if self._placement == DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=self._dataset,
            )

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            shuffle=is_train if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate,
            drop_last=is_train,
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader

    @staticmethod
    def _collate(batch):
        # TODO: ...
        pass
