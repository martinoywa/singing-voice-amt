import glob
import os
import random
import pandas as pd


def select_audio_trios(audio_dir, transcribed=["MT3", "Basic Pitch"], instrument=["flute", "trumpet", "tenor_saxophone", "violin"], n=5):
    fnames = [f for f in glob.glob(os.path.join(audio_dir, "*.*")) if os.path.isfile(f)]
    selected_files = random.sample(fnames, min(n, len(fnames)))
    trios = [(f, random.choice(transcribed), random.choice(instrument)) for f in selected_files]
    return trios


def generate_listening_test_csv(trios):
    data = {"Singing Voice": [],
            "Transcribed": [],
            "Instrument": [],
            "Similar? (Yes/No)": [],
            "Notes": []
            }

    for trio in trios:
        data["Singing Voice"].append(trio[0])
        data["Transcribed"].append(trio[1])
        data["Instrument"].append(trio[2])
        data["Similar? (Yes/No)"].append(None)
        data["Notes"].append(None)

    df = pd.DataFrame(data)
    df.to_csv("listening_test.csv", index=False)


if __name__ == '__main__':
    data_path, output_file = "../dataset/wav", "listening_test_trios.txt"
    trios = select_audio_trios(data_path)
    # print("Selected audio trios:", trios)
    with open(output_file, "w") as f:
        for trio in trios:
            f.write(", ".join(trio) + "\n")

    generate_listening_test_csv(trios)
