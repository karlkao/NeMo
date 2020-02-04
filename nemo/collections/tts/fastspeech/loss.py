# Copyright (c) 2019 NVIDIA Corporation
from typing import Dict, Optional

import torch
from torch import nn

# noinspection PyPep8Naming
from torch.nn import functional as F

from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import (
    AxisType,
    BatchTag,
    ChannelTag,
    MelSpectrogramSignalTag,
    NeuralType,
    ProcessedTimeTag,
    TimeTag,
)


class FastSpeechLoss(LossNM):
    """Neural Module wrapper for FastSpeech loss."""

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(
            mel_true=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            mel_pred=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            dur_true=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            dur_pred=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            text_pos=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        )

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(loss=NeuralType(None))

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

    @staticmethod
    def _loss(
        mel_true, mel_pred, dur_true, dur_pred, text_pos,
    ):
        """Do the actual math in FastSpeech loss calculation.

        Args:
            mel_true: Ground truth mel spectrogram features (BTC, float).
            mel_pred: Predicted mel spectrogram features (BTC, float).
            dur_true: Ground truth durations (BQ, float).
            dur_pred: Predicted log-normalized durations (BQ, float).

        Returns:
            Single 0-dim loss tensor.

        """

        mel_loss = F.mse_loss(mel_pred, mel_true, reduction='none')
        mel_loss *= mel_true.ne(0).float()
        mel_loss = mel_loss.mean()

        dur_loss = F.mse_loss(dur_pred, (dur_true + 1).log(), reduction='none')
        dur_loss *= text_pos.ne(0).float()
        dur_loss = dur_loss.mean()

        loss = mel_loss + dur_loss

        return loss
