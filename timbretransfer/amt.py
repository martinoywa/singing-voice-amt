"""
Loads metadata data, and runs automatic music transcription
using basic-pitch and MT3 of each audio file.
This generates these models baseline transcriptions.
"""
import argparse

from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
import pandas as pd
import time

import os
import glob

os.environ["JAX_PLATFORMS"] = "cpu" # Fixes last_device.core_on_chip + 1 AttributeError

from utils import InferenceModel, upload_audio
import note_seq


SAMPLE_RATE = 16000
basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)
MODEL = "mt3"
checkpoint_path = f'checkpoints/{MODEL}/'
inference_model = InferenceModel(checkpoint_path, MODEL)


def load_audio_files(metadata_file_path, filetype="path"):
    fnames = []
    if filetype == "csv":
        metadata = pd.read_csv(metadata_file_path)
        for _, row in metadata.iterrows():
            fname = "../"+row['wav']
            fnames.append(fname)
    else:
        fnames = [_  for _ in glob.glob(metadata_file_path+"*.*")]
    return fnames

def basic_pitch_transcription(fname, output_dir="../amt_dataset/basic_pitch/"): # TODO create argparse variable
    model_output, midi_data, note_events = predict(fname, basic_pitch_model)
    # return model_output, midi_data, note_events
    save_path = output_dir+fname.split("/")[-1].split(".")[0]+".mid"
    midi_data.write(save_path)


def MT3_transcription(fname, output_dir="amt_dataset/MT3/", sample_rate=SAMPLE_RATE): # TODO create argparse variable
    audio = upload_audio(fname, sample_rate=SAMPLE_RATE)
    est_ns = inference_model(audio)
    save_path = output_dir+fname.split("/")[-1].split(".")[0]+".mid"
    note_seq.sequence_proto_to_midi_file(est_ns, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load metadata and run transcription")
    parser.add_argument("--metadata-file-path", type=str, required=True, default="../dataset/raw/metadata.csv")
    parser.add_argument("--file-type", type=str, required=True, default="csv")
    parser.add_argument("--output-dir", type=str, default="../amt_dataset/basic_pitch/")
    args = parser.parse_args()

    metadata_file_path = args.metadata_file_path
    file_type = args.file_type
    output_dir = args.output_dir

    start_time = time.time()
    fnames = load_audio_files(metadata_file_path, filetype=file_type) # TODO add option to load files from path
    for fname in fnames:
        basic_pitch_transcription(fname, output_dir=output_dir)
        # MT3_transcription(fname[3:], output_dir=output_dir)
        print("Processed audio file ", fname)
    end_time = time.time()
    print(f"Done! Total time: {end_time-start_time:.2f} seconds.")