import argparse
import os
import numpy as np
import pandas as pd
import pretty_midi
import mir_eval

# -----------------------------------------------------------
# 1. Load MIDI notes from file
# -----------------------------------------------------------
def load_midi_as_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch))
    if len(notes) == 0:
        return np.zeros((0, 3))
    return np.array(notes)


# -----------------------------------------------------------
# 2. Evaluate a single MIDI pair
# -----------------------------------------------------------
def evaluate_midi_pair(ref_midi, est_midi, onset_tolerance=0.05, offset_ratio=0.2):
    ref_notes = load_midi_as_notes(ref_midi)
    est_notes = load_midi_as_notes(est_midi)

    if len(ref_notes) == 0 or len(est_notes) == 0:
        return {"Precision": np.nan, "Recall": np.nan, "F1": np.nan,
                "Onset Precision": np.nan, "Onset Recall": np.nan, "Onset F1": np.nan}

    # ✅ Sort notes by start time to ensure increasing order
    ref_notes = ref_notes[np.argsort(ref_notes[:, 0])]
    est_notes = est_notes[np.argsort(est_notes[:, 0])]

    # ✅ Remove duplicates in onset times if they exist
    _, unique_ref_idx = np.unique(ref_notes[:, 0], return_index=True)
    _, unique_est_idx = np.unique(est_notes[:, 0], return_index=True)
    ref_notes = ref_notes[unique_ref_idx]
    est_notes = est_notes[unique_est_idx]

    ref_intervals = ref_notes[:, :2]
    ref_pitches = ref_notes[:, 2].astype(int)
    est_intervals = est_notes[:, :2]
    est_pitches = est_notes[:, 2].astype(int)

    # Note-based metrics
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=onset_tolerance, offset_ratio=None
    )

    # Onset + offset metrics (more strict)
    _, _, f1_onoff, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=onset_tolerance, offset_ratio=offset_ratio
    )

    # Onset-only metrics
    onset_precision, onset_recall, onset_f1 = mir_eval.onset.f_measure(
        ref_intervals[:, 0], est_intervals[:, 0], window=onset_tolerance
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Onset Precision": onset_precision,
        "Onset Recall": onset_recall,
        "Onset F1": onset_f1,
        "Onset-Offset F1": f1_onoff
    }


# -----------------------------------------------------------
# 3. Batch evaluation for all MIDI pairs
# -----------------------------------------------------------
def batch_evaluate(ground_truth_dir, predicted_dir, save_csv=True, output_csv="evaluation_results_basic_pitch.csv"):
    results = []

    gt_files = {os.path.splitext(f)[0]: os.path.join(ground_truth_dir, f)
                for f in os.listdir(ground_truth_dir) if f.endswith(".mid")}
    pred_files = {os.path.splitext(f)[0]: os.path.join(predicted_dir, f)
                  for f in os.listdir(predicted_dir) if f.endswith(".mid")}

    common_files = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    print(f"Found {len(common_files)} matching MIDI pairs")

    for file in common_files:
        gt_path = gt_files[file]
        pred_path = pred_files[file]

        metrics = evaluate_midi_pair(gt_path, pred_path)
        metrics["Filename"] = file
        results.append(metrics)

    df = pd.DataFrame(results)
    avg = df.mean(numeric_only=True)

    print("\nAverage Results:")
    for k, v in avg.items():
        print(f"{k}: {v:.3f}")

    if save_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved detailed results to {output_csv}")

    return df, avg


# -----------------------------------------------------------
# 4. Example Usage
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", type=str, required=True, default="../dataset/midi/")
    parser.add_argument("--predicted-dir", type=str, required=True, help="predicted midi files")
    parser.add_argument("--output-csv", type=str, default="evaluation_results_basic_pitch.csv")
    args = parser.parse_args()

    ground_truth_dir = args.ground_truth_dir
    predicted_dir = args.predicted_dir # "../amt_dataset/basic_pitch/"
    output_csv = args.output_csv

    df, avg = batch_evaluate(ground_truth_dir, predicted_dir, output_csv=output_csv)