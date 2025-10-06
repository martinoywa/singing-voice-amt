"""
Loads metadata data, and runs automatic music transcription
using basic-pitch and MT3 concurrently of each audio file.
This generates these models baseline transcriptions.
"""
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
import pandas as pd
import time


basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

def load_audio_files(metadata_file_path):
    metadata = pd.read_csv(metadata_file_path)
    fnames = []

    for _, row in metadata.iterrows():
        fname = "../"+row['wav']
        fnames.append(fname)

    return fnames

def basic_pitch_transcription(fname, audio_output_dir="../amt_dataset/basic_pitch/"):
    model_output, midi_data, note_events = predict(fname, basic_pitch_model)
    # return model_output, midi_data, note_events
    save_path = audio_output_dir+fname.split("/")[-1].split(".")[0]+".mid"
    midi_data.write(save_path)
    print("Processed audio file ", fname)


if __name__ == "__main__":
    start_time = time.time()
    fnames = load_audio_files("../dataset/raw/metadata.csv")
    for fname in fnames:
        basic_pitch_transcription(fname)
    end_time = time.time()
    print(f"Done! Total time: {end_time-start_time:.2f} seconds.")