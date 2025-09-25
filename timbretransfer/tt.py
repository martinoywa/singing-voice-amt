# Ignore a bunch of deprecation warnings
import argparse
import warnings

import pandas as pd
from scipy.io import wavfile

warnings.filterwarnings("ignore")

import copy
import os
import time

import crepe
import ddsp
import ddsp.training
from ddsp.colab.colab_utils import (
    auto_tune, get_tuning_factor,
    specplot, audio_bytes_to_np,
    DEFAULT_SAMPLE_RATE)
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)
import gin
# from google.colab import files
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# Helper Functions
sample_rate = DEFAULT_SAMPLE_RATE  # 16000

## Helper functions.
def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features['loudness_db'] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
    audio_features['f0_hz'] = np.clip(audio_features['f0_hz'],
                                      0.0,
                                      librosa.midi_to_hz(110.0))
    return audio_features

def load_audio_files(metadata_file_path):
    metadata = pd.read_csv(metadata_file_path)
    fnames, audios = [], []

    for _, row in metadata.iterrows():
        fname = "../"+row['wav']
        with open(fname, "rb") as f:
            audio_bytes = f.read()
        file_audio = audio_bytes_to_np(audio_bytes,
                                       sample_rate=sample_rate,
                                       normalize_db=None)
        audios.append(file_audio)
        fnames.append(fname)

    return audios, fnames

def save_wav(array_of_floats,
             filename,
             sample_rate=DEFAULT_SAMPLE_RATE):
    """Save an array of floats as a WAV file."""

    # If batched, take first element
    if len(array_of_floats.shape) == 2:
        array_of_floats = array_of_floats[0]

    # Clip values to [-1, 1] just in case
    array_of_floats = np.clip(array_of_floats, -1.0, 1.0)

    # Convert to 16-bit PCM
    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(
        np.asarray(array_of_floats) * normalizer, dtype=np.int16)

    # Save as WAV
    wavfile.write(filename, sample_rate, array_of_ints)
    print(f"\nSaved WAV file: {filename}")


def process_audio(audio, ckpt, DATASET_STATS):
    # Setup the session.
    ddsp.spectral_ops.reset_crepe()

    if len(audio.shape) == 1:
        audio = audio[np.newaxis, :]

    # Compute features.
    start_time = time.time()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    audio_features_mod = None
    print('Audio features took %.1f seconds' % (time.time() - start_time))

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    # print("\n===Resynthesis===")
    # print("Time Steps", time_steps)
    # print("Samples", n_samples)
    # print('')

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]

    # Set up the model just to predict audio given new conditioning
    # TODO figure out how to use restore model outside this function. Temporary fix for clipped outputs.
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)
    print('Restoring model took %.1f seconds' % (time.time() - start_time))

    #@markdown ## Note Detection

    #@markdown You can leave this at 1.0 for most cases
    threshold = 0 #@param {type:"slider", min: 0.0, max:2.0, step:0.01}


    #@markdown ## Automatic

    ADJUST = True #@param{type:"boolean"}

    #@markdown Quiet parts without notes detected (dB)
    quiet = 60 #@param {type:"slider", min: 0, max:60, step:1}

    #@markdown Force pitch to nearest note (amount)
    autotune = 1 #@param {type:"slider", min: 0.0, max:1.0, step:0.1}

    #@markdown ## Manual


    #@markdown Shift the pitch (octaves)
    pitch_shift =  0 #@param {type:"slider", min:-2, max:2, step:1}

    #@markdown Adjust the overall loudness (dB)
    loudness_shift = 0 #@param {type:"slider", min:-20, max:20, step:1}


    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}

    mask_on = None

    if ADJUST and DATASET_STATS is not None:
        # Detect sections that are "on".
        mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                              audio_features['f0_confidence'],
                                              threshold)

        if np.any(mask_on):
            # Shift the pitch register.
            target_mean_pitch = DATASET_STATS['mean_pitch']
            pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
            mean_pitch = np.mean(pitch[mask_on])
            p_diff = target_mean_pitch - mean_pitch
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)


            # Quantile shift the note_on parts.
            _, loudness_norm = fit_quantile_transform(
                audio_features['loudness_db'],
                mask_on,
                inv_quantile=DATASET_STATS['quantile_transform'])

            # Turn down the note_off parts.
            mask_off = np.logical_not(mask_on)
            loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
            loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)

            audio_features_mod['loudness_db'] = loudness_norm

            # Auto-tune.
            if autotune:
                f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
                tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
                f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
                audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)

        else:
            print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')

    else:
        print('\nSkipping auto-adujst (box not checked or no dataset statistics found).')

    # Manual Shifts.
    audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
    audio_features_mod = shift_f0(audio_features_mod, pitch_shift)


    #@title #Resynthesize Audio
    af = audio_features if audio_features_mod is None else audio_features_mod

    # Run a batch of predictions.
    start_time = time.time()
    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print('Prediction took %.1f seconds' % (time.time() - start_time))

    return audio_gen



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model name: Violin/Flute/Trumpet/Tenor_Saxophone")
    args = parser.parse_args()


    model_name = args.model_name #@param ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone', 'Upload your own (checkpoint folder as .zip)']
    MODEL = model_name

    if model_name in ('Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone'):
        # Pretrained models.
        PRETRAINED_DIR = '../pretrained/' + model_name.lower()
        model_dir = PRETRAINED_DIR
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')

    # Load the dataset statistics.
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print('Loading dataset statistics from pickle failed: {}.'.format(err))

    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    # print("===Trained model===")
    # print("Time Steps", time_steps_train)
    # print("Samples", n_samples_train)
    # print("Hop Size", hop_size)

    # TODO Process
    audios, fnames = load_audio_files("../dataset/raw/metadata.csv")
    for audio, fname in zip(audios, fnames):
        audio_gen = process_audio(audio, ckpt, DATASET_STATS=DATASET_STATS)
        save_path = "../tt_dataset/"+model_name.lower()+"/"+fname.split("/")[-1]
        save_wav(audio_gen, save_path)




