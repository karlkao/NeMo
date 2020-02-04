# Copyright (c) 2019 NVIDIA Corporation
import json
import os
import pathlib
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch

# noinspection PyPep8Naming
import torch.nn.functional as F

import nemo
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.collections.asr.parts import collections
from nemo.collections.asr.parts import features as asr_parts_features
from nemo.collections.asr.parts import parsers
from nemo.collections.tts.fastspeech import text_norm
from nemo.core.neural_types import AxisType, BatchTag, ChannelTag, NeuralType, TimeTag


class AdHoc:
    @staticmethod
    def __dynamic_range_compression(x, C=1, clip_val=1e-5):
        return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

    @staticmethod
    def wav_file_to_features(wav_file):
        wav, sr = librosa.load(wav_file, sr=22050)
        wav, _ = librosa.effects.trim(wav, frame_length=1024, hop_length=256)
        mel = librosa.feature.melspectrogram(
            wav, sr=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0,
        )
        mel = AdHoc.__dynamic_range_compression(mel)
        return mel


class FastSpeechDataset(torch.utils.data.Dataset):
    """Merges audio, text and alignments examples into one dataset.

    This dataset should assumed particular file structure. Take a look at:
    https://ngc.nvidia.com/datasets/bSOCUeD5QHO0uyIGDtTphA.

    """

    def __init__(
        self,
        data_dir,
        split_name,
        labels,
        normalize,
        min_duration,
        max_duration,
        sample_rate,
        int_values,
        trim,
        bos_id,
        eos_id,
    ):

        data_dir = pathlib.Path(data_dir)
        wavs_dir, alignments_dir = data_dir / 'wavs', data_dir / 'alignments'
        audio_files, durations, texts, alignments_files = [], [], [], []
        with open(data_dir / f'{split_name}.json', 'r') as f:
            for example_dict in json.load(f):
                name = example_dict['name']
                audio_files.append(str(wavs_dir / f'{name}.wav'))
                durations.append(example_dict['duration'])
                texts.append(example_dict['text'])
                alignments_files.append(str(alignments_dir / f'{name}.npy'))

        # parser = parsers.ENCharParser(labels=labels, do_normalize=normalize)
        del labels
        del normalize

        def parser(text):
            return text_norm.text_to_sequence(text, cleaner_names=['english_cleaners'])

        self._audio_text = collections.AudioText(
            audio_files=audio_files,
            durations=durations,
            texts=texts,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self._featurizer = asr_parts_features.WaveformFeaturizer(
            sample_rate=sample_rate, int_values=int_values, augmentor=None
        )
        self._trim = trim
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._alignments_files = alignments_files

    def __getitem__(self, index):
        audio_text, alignments_file = self._audio_text[index], self._alignments_files[index]
        # audio_features = self._featurizer.process(
        #     audio_text.audio_file, offset=0, duration=audio_text.duration, trim=self._trim
        # )
        audio_features = AdHoc.wav_file_to_features(audio_text.audio_file)
        text = audio_text.text_tokens
        alignments = list(np.load(alignments_file))
        if self._bos_id is not None:
            text = [self._bos_id] + text
            alignments = [0.0] + alignments
        if self._eos_id is not None:
            text = text + [self._eos_id]
            alignments = alignments + [0.0]

        return {
            'text': torch.tensor(text, dtype=torch.float),
            'text_length': torch.tensor(len(text), dtype=torch.long),
            'mel_true': torch.tensor(audio_features.T, dtype=torch.float),
            'dur_true': torch.tensor(alignments, dtype=torch.float),
        }

    def __len__(self):
        return len(self._audio_text)


class FastSpeechDataLayer(DataLayerNM):
    # noinspection DuplicatedCode
    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return dict(
            text=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            text_pos=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            mel_true=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag)}),
            dur_true=NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        )

    def __init__(
        self,
        data_dir: str,
        split_name: str,
        labels: List[str],
        normalize=False,
        min_duration=None,
        max_duration=None,
        sample_rate=16000,
        int_values=False,
        trim=False,
        bos_id=None,
        eos_id=None,
        pad_id=None,
        batch_size=32,
        num_workers=0,
    ):
        super().__init__()

        self._dataset = FastSpeechDataset(
            data_dir=data_dir,
            split_name=split_name,
            labels=labels,
            normalize=normalize,
            min_duration=min_duration,
            max_duration=max_duration,
            sample_rate=sample_rate,
            int_values=int_values,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        self._pad_id = pad_id

        sampler = None
        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        is_train = split_name == 'train'
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            shuffle=is_train if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate,
            drop_last=is_train,
        )

    def _collate(self, batch):
        def merge(tensors, value=0.0, dtype=torch.float):
            max_len = max(tensor.shape[0] for tensor in tensors)
            new_tensors = []
            for tensor in tensors:
                pad = (2 * len(tensor.shape)) * [0]
                pad[-1] = max_len - tensor.shape[0]
                new_tensors.append(F.pad(tensor, pad=pad, value=value))
            return torch.stack(new_tensors).to(dtype=dtype)

        def make_pos(lengths):
            return merge([torch.arange(length) + 1 for length in lengths], value=0, dtype=torch.int64)

        batch = {key: [example[key] for example in batch] for key in batch[0]}

        text = merge(batch['text'], value=self._pad_id or 0, dtype=torch.long)
        text_pos = make_pos(batch.pop('text_length'))
        mel_true = merge(batch['mel_true'])
        dur_true = merge(batch['dur_true'])

        return text, text_pos, mel_true, dur_true

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> Optional[torch.utils.data.Dataset]:
        return None

    @property
    def data_iterator(self) -> Optional[torch.utils.data.DataLoader]:
        return self._dataloader
