import librosa
import numpy as np
from scipy.spatial.distance import cosine
import openl3
import soundfile as sf
import os
import argparse
import pandas as pd


def mfcc_similarity(original_wav, converted_wav):
    y1, sr1 = librosa.load(original_wav, sr=16000)
    y2, sr2 = librosa.load(converted_wav, sr=16000)

    # Extract MFCC features (mean across time)
    mfcc1 = np.mean(librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13), axis=1)
    mfcc2 = np.mean(librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13), axis=1)

    # Compute similarity (1 - cosine distance)
    similarity = 1 - cosine(mfcc1, mfcc2)
    return similarity

def embedding_similarity(file1, file2):
    x1, sr1 = sf.read(file1)
    x2, sr2 = sf.read(file2)
    emb1, _ = openl3.get_audio_embedding(x1, sr1, embedding_size=512, content_type="music")
    emb2, _ = openl3.get_audio_embedding(x2, sr2, embedding_size=512, content_type="music")
    return 1 - cosine(np.mean(emb1, axis=0), np.mean(emb2, axis=0))

def batch_evaluate(ground_truth_dir, predicted_dir, save_csv=True, output_csv="timbre_fidelity_results.csv"):
    results = []

    gt_files = {os.path.splitext(f)[0]: os.path.join(ground_truth_dir, f)
                for f in os.listdir(ground_truth_dir) if f.endswith(".wav")}
    pred_files = {os.path.splitext(f)[0]: os.path.join(predicted_dir, f)
                  for f in os.listdir(predicted_dir) if f.endswith(".wav")}

    common_files = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    print(f"Found {len(common_files)} matching WAV pairs")

    for file in common_files:
        print(f"Evaluating {file}")
        gt_path = gt_files[file]
        pred_path = pred_files[file]

        score = {}
        score["MFCC Similarity"] = mfcc_similarity(gt_path, pred_path)
        score["Embedding Similarity"] = embedding_similarity(gt_path, pred_path)
        score["Filename"] = file
        results.append(score)

    df = pd.DataFrame(results)
    avg = df.mean(numeric_only=True)

    print("\nAverage Results:")
    for k, v in avg.items():
        print(f"{k}: {v:.3f}")

    if save_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved detailed results to {output_csv}")

    return df, avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", type=str, required=True, default="../dataset/wav/")
    parser.add_argument("--predicted-dir", type=str, required=True, help="converted wav files")
    parser.add_argument("--output-csv", type=str, default="timbre_fidelity_results.csv")
    args = parser.parse_args()

    ground_truth_dir = args.ground_truth_dir
    predicted_dir = args.predicted_dir # "../tt_dataset/flute/"
    output_csv = args.output_csv

    df, avg = batch_evaluate(ground_truth_dir, predicted_dir, output_csv=output_csv)
