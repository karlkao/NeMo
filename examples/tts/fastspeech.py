# Copyright (c) 2019 NVIDIA Corporation
import argparse
import math
import os
from pathlib import Path

import attrdict
from ruamel import yaml

import nemo
from nemo.collections import asr as nemo_asr
from nemo.collections import tts as nemo_tts
from nemo.collections.tts.fastspeech import text_norm
from nemo.utils import argparse as nm_argparse
from nemo.utils import lr_policies


def parse_args():
    parser = argparse.ArgumentParser(
        description='FastSpeech training pipeline.',
        parents=[nm_argparse.NemoArgParser()],
        conflict_handler='resolve',  # For parents common flags.
    )
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer='adam',
        batch_size=32,
        eval_batch_size=32,
        lr=0.001,
        amp_opt_level='O0',
        create_tb_writer=True,
        lr_policy=None,
        weight_decay=1e-6,
    )

    parser.add_argument('--id', type=str, default='default', help="Experiment identificator for clarity.")
    parser.add_argument('--data_dir', type=str, help="Dataset root directory path.")
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument('--min_lr', type=float, default=1e-5, help="Minimum learning rate to decay to.")

    args = parser.parse_args()

    return args


class FastSpeechGraph:
    def __init__(self, args, config, num_workers):
        n_labels = len(text_norm.symbols)
        # bos_id, eos_id, pad_id, n_labels = None, None, n_labels, n_labels + 1
        bos_id, eos_id, pad_id, n_labels = None, None, 0, n_labels
        self.data_layer = nemo_tts.FastSpeechDataLayer(
            data_dir=args.data_dir,
            split_name='train',
            labels=text_norm.symbols,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            batch_size=args.batch_size,
            num_workers=num_workers,
            **config.FastSpeechDataLayer,
        )
        self.fastspeech = nemo_tts.FastSpeech(**config.FastSpeech, n_src_vocab=n_labels, pad_id=pad_id)
        self.losser = nemo_tts.FastSpeechLoss()

    def build_loss(self):
        callbacks = []
        data = self.data_layer()
        mel_pred, dur_pred = self.fastspeech(
            text=data.text, text_pos=data.text_pos, mel_true=data.mel_true, dur_true=data.dur_true,
        )
        loss = self.losser(
            mel_true=data.mel_true,
            mel_pred=mel_pred,
            dur_true=data.dur_true,
            dur_pred=dur_pred,
            text_pos=data.text_pos,
        )

        callbacks.append(
            nemo.core.SimpleLossLoggerCallback([loss], print_func=lambda x: nemo.logging.info(f'Loss: {x[0].data}'))
        )

        return loss, callbacks


def main():
    args = parse_args()
    work_dir = Path(args.work_dir) / args.id
    engine = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        cudnn_benchmark=args.cudnn_benchmark,
        log_dir=work_dir / 'log',
        checkpoint_dir=work_dir / 'checkpoints',
        tensorboard_dir=work_dir / 'tensorboard',
        files_to_copy=[args.model_config],
    )

    yaml_loader = yaml.YAML(typ="safe")
    with open(args.model_config) as f:
        config = attrdict.AttrDict(yaml_loader.load(f))
    nemo.logging.info(f'Config: {config}')
    graph = FastSpeechGraph(args, config, num_workers=max(int(os.cpu_count() / engine.world_size), 1))

    steps_per_epoch = math.ceil(len(graph.data_layer) / (args.batch_size * engine.world_size))
    total_steps = args.max_steps if args.max_steps is not None else args.num_epochs * steps_per_epoch
    loss, callbacks = graph.build_loss()
    engine.train(
        tensors_to_optimize=[loss],
        optimizer=args.optimizer,
        optimization_params=dict(
            num_epochs=args.num_epochs,
            max_steps=total_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_norm_clip=args.grad_norm_clip,
        ),
        callbacks=callbacks,
        lr_policy=lr_policies.CosineAnnealing(total_steps, min_lr=args.min_lr),
    )


if __name__ == '__main__':
    main()
