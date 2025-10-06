"""
Loads metadata data, and runs automatic music transcription
using basic-pitch and MT3 of each audio file.
This generates these models baseline transcriptions.
"""
# from basic_pitch.inference import predict, Model
# from basic_pitch import ICASSP_2022_MODEL_PATH
import pandas as pd
import time

import os

os.environ["JAX_PLATFORMS"] = "cpu" # Fixes last_device.core_on_chip + 1 AttributeError

from utils import InferenceModel, upload_audio
import note_seq


SAMPLE_RATE = 16000
# basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
MODEL = "mt3"
checkpoint_path = f'checkpoints/{MODEL}/'
inference_model = InferenceModel(checkpoint_path, MODEL)


def load_audio_files(metadata_file_path):
    metadata = pd.read_csv(metadata_file_path)
    fnames = []

    for _, row in metadata.iterrows():
        fname = "../"+row['wav']
        fnames.append(fname)

    return fnames

def basic_pitch_transcription(fname, output_dir="../amt_dataset/basic_pitch/"):
    model_output, midi_data, note_events = predict(fname, basic_pitch_model)
    # return model_output, midi_data, note_events
    save_path = output_dir+fname.split("/")[-1].split(".")[0]+".mid"
    midi_data.write(save_path)


def MT3_transcription(fname, output_dir="amt_dataset/MT3/", sample_rate=SAMPLE_RATE):
    audio = upload_audio(fname, sample_rate=SAMPLE_RATE)
    est_ns = inference_model(audio)
    save_path = output_dir+fname.split("/")[-1].split(".")[0]+".mid"
    note_seq.sequence_proto_to_midi_file(est_ns, save_path)


if __name__ == "__main__":
    start_time = time.time()
    fnames = load_audio_files("dataset/raw/metadata.csv")
    for fname in fnames:
        # basic_pitch_transcription(fname)
        MT3_transcription(fname[3:])
        print("Processed audio file ", fname)
    end_time = time.time()
    print(f"Done! Total time: {end_time-start_time:.2f} seconds.")