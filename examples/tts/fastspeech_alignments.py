# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import os
import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from ruamel.yaml import YAML
from scipy.io.wavfile import write
from tacotron2 import create_NMs

import nemo
import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts

logging = nemo.logging


def parse_args():
    parser = argparse.ArgumentParser(description='TTS')
    parser.add_argument("--local_rank", default=None, type=int)
    parser.add_argument(
        "--spec_model",
        type=str,
        required=True,
        choices=["tacotron2"],
        help="Model generated to generate spectrograms",
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        required=True,
        choices=["griffin-lim", "waveglow"],
        help="Vocoder used to convert from spectrograms to audio",
    )
    parser.add_argument(
        "--spec_model_config", type=str, required=True, help="spec model configuration file: model.yaml",
    )
    parser.add_argument(
        "--vocoder_model_config",
        type=str,
        help=("vocoder model configuration file: model.yaml. Not required for " "griffin-lim."),
    )
    parser.add_argument(
        "--spec_model_load_dir", type=str, required=True, help="directory containing checkpoints for spec model",
    )
    parser.add_argument(
        "--vocoder_model_load_dir",
        type=str,
        help=("directory containing checkpoints for vocoder model. Not " "required for griffin-lim"),
    )
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--save_dir", type=str, help="directory to save audio files to")

    # Grifflin-Lim parameters
    parser.add_argument(
        "--griffin_lim_mag_scale",
        type=float,
        default=2048,
        help=(
            "This is multiplied with the linear spectrogram. This is "
            "to avoid audio sounding muted due to mel filter normalization"
        ),
    )
    parser.add_argument(
        "--griffin_lim_power",
        type=float,
        default=1.2,
        help=(
            "The linear spectrogram is raised to this power prior to running"
            "the Griffin Lim algorithm. A power of greater than 1 has been "
            "shown to improve audio quality."
        ),
    )
    parser.add_argument(
        '--durations_dir', type=str, default='durs',
    )

    # Waveglow parameters
    parser.add_argument(
        "--waveglow_denoiser_strength",
        type=float,
        default=0.0,
        help=("denoiser strength for waveglow. Start with 0 and slowly " "increment"),
    )
    parser.add_argument("--waveglow_sigma", type=float, default=0.6)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--amp_opt_level", default="O1")

    args = parser.parse_args()
    if args.vocoder == "griffin-lim" and (args.vocoder_model_config or args.vocoder_model_load_dir):
        raise ValueError(
            "Griffin-Lim was specified as the vocoder but the a value for "
            "vocoder_model_config or vocoder_model_load_dir was passed."
        )
    return args


def create_infer_dags(
    neural_factory, neural_modules, tacotron2_params, infer_dataset, infer_batch_size, cpu_per_dl=1,
):
    (data_preprocessor, text_embedding, t2_enc, t2_dec, t2_postnet, _, _) = neural_modules

    eval_dl_params = copy.deepcopy(tacotron2_params["AudioToTextDataLayer"])
    eval_dl_params.update(tacotron2_params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=infer_dataset,
        labels=tacotron2_params['labels'],
        bos_id=len(tacotron2_params['labels']),
        eos_id=len(tacotron2_params['labels']) + 1,
        pad_id=len(tacotron2_params['labels']) + 2,
        batch_size=infer_batch_size,
        num_workers=cpu_per_dl,
        **eval_dl_params,
    )

    audio, audio_len, transcript, transcript_len = data_layer()
    spec_target, spec_target_len = data_preprocessor(input_signal=audio, length=audio_len)

    transcript_embedded = text_embedding(char_phone=transcript)
    transcript_encoded = t2_enc(char_phone_embeddings=transcript_embedded, embedding_length=transcript_len,)
    if isinstance(t2_dec, nemo_tts.Tacotron2Decoder):
        t2_dec.eval()
        t2_dec.training = True
        mel_decoder, gate, alignments = t2_dec(
            char_phone_encoded=transcript_encoded, encoded_length=transcript_len, mel_target=spec_target,
        )
    else:
        raise ValueError("The Neural Module for tacotron2 decoder was not understood")
    mel_postnet = t2_postnet(mel_input=mel_decoder)

    return [mel_postnet, gate, alignments, spec_target_len, transcript_len]


def main():
    args = parse_args()
    neural_factory = nemo.core.NeuralModuleFactory(
        optimization_level=args.amp_opt_level, backend=nemo.core.Backend.PyTorch, local_rank=args.local_rank,
    )

    use_cache = True
    if args.local_rank is not None:
        logging.info("Doing ALL GPU")
        use_cache = False

    # Create text to spectrogram model
    if args.spec_model == "tacotron2":
        yaml = YAML(typ="safe")
        with open(args.spec_model_config) as file:
            tacotron2_params = yaml.load(file)
        spec_neural_modules = create_NMs(tacotron2_params, decoder_infer=False)
        infer_tensors = create_infer_dags(
            neural_factory=neural_factory,
            neural_modules=spec_neural_modules,
            tacotron2_params=tacotron2_params,
            infer_dataset=args.eval_dataset,
            infer_batch_size=args.batch_size,
        )

    logging.info("Running Tacotron 2")
    # Run tacotron 2
    evaluated_tensors = neural_factory.infer(
        tensors=infer_tensors, checkpoint_dir=args.spec_model_load_dir, cache=False, offload_to_cpu=True,
    )

    def get_D(alignment):
        D = np.array([0 for _ in range(np.shape(alignment)[1])])

        for i in range(np.shape(alignment)[0]):
            max_index = alignment[i].tolist().index(alignment[i].max())
            D[max_index] = D[max_index] + 1

        assert D.sum() == alignment.shape[0]

        return D

    # Save durations.
    alignments_dir = pathlib.Path(args.durations_dir)
    alignments_dir.mkdir(exist_ok=True)
    k = -1
    for alignments, mel_lens, text_lens in zip(
        tqdm.tqdm(evaluated_tensors[2]), evaluated_tensors[3], evaluated_tensors[4],
    ):
        for alignment, mel_len, text_len in zip(alignments, mel_lens, text_lens):
            alignment = alignment.cpu().numpy()
            mel_len = mel_len.cpu().numpy().item()
            text_len = text_len.cpu().numpy().item()
            dur = get_D(alignment[:mel_len, :text_len])
            k += 1
            np.save(alignments_dir / f'{k}.npy', dur, allow_pickle=False)


if __name__ == '__main__':
    main()
