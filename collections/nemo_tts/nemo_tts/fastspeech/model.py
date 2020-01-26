from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pad_sequence

from fastspeech.model.module import Conv
from fastspeech.model.transformer.Models import Encoder, Decoder
from fastspeech.utils.pytorch import to_device_async


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_output_size, duration_predictor_filter_size,
                 duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor(
            input_size=encoder_output_size,
            filter_size=duration_predictor_filter_size,
            kernel=duration_predictor_kernel_size,
            conv_output_size=duration_predictor_filter_size,
            dropout=dropout
        )

    def forward(self, encoder_output, encoder_output_mask, target=None,
                alpha=1.0, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(
            encoder_output, encoder_output_mask)
        # print(duration_predictor_output)

        if self.training:
            output, dec_pos = self.get_output(
                encoder_output, target, alpha, mel_max_length)
        else:
            duration_predictor_output = torch.clamp_min(
                torch.exp(duration_predictor_output) - 1, 0)

            output, dec_pos = self.get_output(
                encoder_output, duration_predictor_output, alpha)

        return output, dec_pos, duration_predictor_output

    @staticmethod
    def get_output(encoder_output, duration_predictor_output, alpha,
                   mel_max_length=None):
        output = list()
        dec_pos = list()

        for i in range(encoder_output.size(0)):
            repeats = duration_predictor_output[i].float() * alpha
            repeats = torch.round(repeats).long()
            output.append(
                torch.repeat_interleave(encoder_output[i], repeats, dim=0))
            dec_pos.append(
                torch.from_numpy(np.indices((output[i].shape[0],))[0] + 1))

        output = pad_sequence(output, batch_first=True)
        dec_pos = pad_sequence(dec_pos, batch_first=True)

        dec_pos = to_device_async(dec_pos, device=output.device)

        if mel_max_length:
            output = output[:, :mel_max_length]
            dec_pos = dec_pos[:, :mel_max_length]

        return output, dec_pos


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, input_size, filter_size, kernel, conv_output_size,
                 dropout):
        super(DurationPredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.conv_output_size = conv_output_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1, bias=True)

    def forward(self, encoder_output, encoder_output_mask):
        encoder_output = encoder_output * encoder_output_mask

        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out * encoder_output_mask
        out = out.squeeze(-1)

        return out


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, d_dec_out, n_mels,
                 max_seq_len, word_vec_dim,
                 encoder_n_layer, encoder_head, encoder_conv1d_filter_size,
                 decoder_n_layer, decoder_head, decoder_conv1d_filter_size,
                 fft_conv1d_kernel, fft_conv1d_padding,
                 encoder_output_size, duration_predictor_filter_size,
                 duration_predictor_kernel_size, dropout):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(
            len_max_seq=max_seq_len,
            d_word_vec=word_vec_dim,
            n_layers=encoder_n_layer,
            n_head=encoder_head,
            d_k=64,
            d_v=64,
            d_model=word_vec_dim,
            d_inner=encoder_conv1d_filter_size,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout
        )
        self.length_regulator = LengthRegulator(
            encoder_output_size,
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            dropout
        )

        self.decoder = Decoder(
            len_max_seq=max_seq_len,
            d_word_vec=word_vec_dim,
            n_layers=decoder_n_layer,
            n_head=decoder_head,
            d_k=64,
            d_v=64,
            d_model=word_vec_dim,
            d_inner=decoder_conv1d_filter_size,
            fft_conv1d_kernel=fft_conv1d_kernel,
            fft_conv1d_padding=fft_conv1d_padding,
            dropout=dropout
        )

        self.mel_linear = nn.Linear(d_dec_out, n_mels, bias=True)

    def forward(self, src_seq, src_pos, mel_max_length=None,
                length_target=None, alpha=1.0):
        encoder_output, encoder_mask = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, decoder_pos, duration_predictor_output = (
                self.length_regulator(
                    encoder_output,
                    encoder_mask,
                    length_target,
                    alpha,
                    mel_max_length)
            )

            assert length_regulator_output.shape[1] <= mel_max_length

        else:
            length_regulator_output, decoder_pos, duration_predictor_output = (
                self.length_regulator(
                    encoder_output, encoder_mask, alpha=alpha
                )
            )

        decoder_output, decoder_mask = self.decoder(length_regulator_output,
                                                    decoder_pos)
        mel_output = self.mel_linear(decoder_output)

        return mel_output, decoder_mask, duration_predictor_output
