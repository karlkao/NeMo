import argparse
import nemo
from pathlib import Path
import nemo_asr

from ruamel import yaml
from nemo.utils import argparse as nm_argparse
import attrdict

import nemo_tts

def parse_args():
    parser = argparse.ArgumentParser(
        description='FastSpeech training pipeline.',
        parents=[nm_argparse.NemoArgParser()],
        conflict_handler='resolve',  # For parents common flags.
    )

    parser.add_argument(
        '--id',
        type=str,
        default='default',
        help='Experiment identificator for clarity.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed value for reproducibility.',
    )

    args = parser.parse_args()

    return args


def define_dag(trainer, config):
    preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
        **config.AudioToMelSpectrogramPreprocessor
    )

    return None


class FastSpeechGraph:
    def __init__(self, config):
        self.processor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            **config.preprocessor
        )

        self.text_embedding = nemo_tts.TextEmbedding()

    def loss(self):
        pass


def main():
    args = parse_args()

    # ...

    work_dir = Path(args.work_dir) / args.id
    trainer = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        cudnn_benchmark=args.cudnn_benchmark,
        random_seed=args.seed,
        log_dir=work_dir / 'log',
        checkpoint_dir=work_dir / 'checkpoints',
        tensorboard_dir=work_dir / 'tensorboard',
        files_to_copy=[args.model_config, __file__],
    )

    yaml_loader = yaml.YAML(typ="safe")
    with open(args.model_config) as f:
        config = attrdict.AttrDict(yaml_loader.load(f))

    print(config)
    exit(0)

    graph = FastSpeechGraph(config)
    loss = graph.loss()

    print(loss)


if __name__ == '__main__':
    main()
