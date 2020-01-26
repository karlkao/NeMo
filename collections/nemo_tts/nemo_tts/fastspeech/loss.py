# Copyright (c) 2019 NVIDIA Corporation
import torch
import torch.nn as nn

from nemo.backends.pytorch.nm import LossNM
# noinspection PyPep8Naming
from torch.nn import functional as F
from nemo.core.neural_types import (
    NeuralType, AxisType, BatchTag, TimeTag, ChannelTag,
)


class FastSpeechLoss(LossNM):
    """
    Neural Module wrapper for pytorch's ctcloss

    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
    """

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(
            true_mel=NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag),
            }),
            pred_mel=NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag),
            }),
        )

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(loss=NeuralType(None))

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    def _loss(self, true_mel, pred_mel, true_dur, pred_dur):
        mel_loss = F.mse_loss(pred_mel, true_mel, reduction='none')
        mel_loss *= true_mel.ne(0).float()  # Multiply by true mel mask.
        mel_loss = mel_loss.mean()

        dur_loss = F.mse_loss()

        loss = mel_loss + dur_loss

        return loss
