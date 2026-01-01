# Singing Voice AMT with Timbre Transfer

This repository contains the code and notebooks for the MSc dissertation **“Timbre Transfer of the Human Singing Voice for Effective Automatic Music Transcription”** by **Bonface Martin Oywa** (University of East London / UNICAF, 2025). [Timbre Transfer of Human Singing Voice for Effective Automatic Music Transcription Dissertation.pdf](https://github.com/martinoywa/singing-voice-amt/blob/master/Timbre%20Transfer%20of%20Human%20Singing%20Voice%20for%20Effective%20Automatic%20Music%20Transcription%20Dissertation.pdf)

The project investigates whether converting human singing voice into instrument-like timbres (via DDSP) can improve the performance of state-of-the-art Automatic Music Transcription (AMT) systems such as **Basic Pitch** and **MT3**.

> **Short answer:** timbre transfer substantially changes the audio, but does **not** reliably improve overall vocal AMT accuracy compared to transcribing the raw singing voice.

---

## 1. Project Overview

### 1.1 Research Question

> *Does converting human singing voice into instrument-like timbres via timbre transfer improve downstream AMT performance compared to direct vocal transcription, and under what conditions?* 

To answer this, the repository implements a **unified pipeline**:

1. **Data Engineering / Ground Truth**
    - Start from **Annotated-VocalSet** extended CSV files (note onsets, offsets, MIDI pitches, etc.).
    - Generate ground-truth **MIDI** files and a `metadata.csv` linking each audio excerpt to its MIDI.

2. **Timbre Transfer (DDSP)**
    - Convert monophonic singing into instrument-like audio (Violin, Flute, Trumpet, Tenor Saxophone) using pre-trained **DDSP** autoencoders.

3. **Automatic Music Transcription (AMT)**
    - Transcribe both **raw vocals** and **timbre-transferred** audio using:
        - **Basic Pitch** (Spotify)
        - **MT3** (Multi-Task Multitrack Music Transcription, Google) 

4. **Evaluation**
    - Objective AMT evaluation with **MIR-Eval**: Precision, Recall, F1, Onset F1, Onset-Offset F1, and Transcription Error Rate (TER).
    - Timbre fidelity evaluation with **MFCC similarity** and **OpenL3 embedding similarity** between original and converted audio.
    - Small subjective listening test for “note-flow” similarity.

---

## 2. Repository Structure

At a high level:

```text
singing-voice-amt/
├── dataset/                # Local data organisation (WAV, MIDI, metadata)
│   └── raw/                # Entry point for downloaded datasets (see below)
├── timbretransfer/         # DDSP timbre transfer utilities & helpers
├── evaluation/             # AMT + timbre fidelity evaluation scripts (MIR-Eval, OpenL3, etc.)
├── data_engineering.ipynb  # Build metadata.csv + generate MIDI from Annotated-VocalSet
├── timbre_transfer.ipynb   # End-to-end DDSP timbre transfer for all excerpts
├── Music_Transcription_with_Transformers.ipynb  # MT3-based AMT experiments
├── Onsets_and_Frames.ipynb # Additional AMT baseline experiments (piano-style model)
├── requirements-MT3.txt    # Dependencies for MT3-based experiments
├── requirements-TT.txt     # Dependencies for DDSP timbre transfer
├── README.md               # (You are here)
└── LICENSE                 # CC0-1.0

Note: Paths like dataset/wav, dataset/midi, tt_dataset/<instrument>/, and amt_dataset/... are created as you run the notebooks and scripts.  ￼
```

## 3. Datasets

The repository works with publicly-available singing datasets; it does not redistribute any audio.

### 3.1 Annotated-VocalSet (Core Dataset)

Experiments in the dissertation primarily use Annotated-VocalSet (extension of VocalSet) which provides:  ￼
1. Monophonic a cappella singing by 20 professional singers (9 male, 11 female).
2.  Extended CSV annotations with:
    - Onset / offset times
    - MIDI pitch (estimated and ground truth)
    - Note durations
    - Vowel and lyric information

3. You’ll need:
	- Extended CSVs (extended_2 variant recommended in the project)
	- Corresponding .wav audio files

4. Download from Zenodo (see links already listed in [Timbre Transfer of Human Singing Voice for Effective Automatic Music Transcription Dissertation.pdf](https://github.com/martinoywa/singing-voice-amt/blob/master/Timbre%20Transfer%20of%20Human%20Singing%20Voice%20for%20Effective%20Automatic%20Music%20Transcription%20Dissertation.pdf)).

### 3.2 Other Mentioned Datasets

The dissertation also discusses, but does not necessarily use in the main experiments:
1. Choral Singing Dataset – choir recordings with MIDI, f0, and note annotations (rejected due to vibrato-only style).  ￼
2. Other vocal datasets (MIR-1K, NUS-48E, etc.) were considered but not adopted due to lack of aligned MIDI.  ￼



## 4. Installation & Environments

The original experiments used two machines because of library constraints:  ￼
1. macOS (M1 Pro, 16 GB RAM):
	- Python 3.8 / 3.11 via Anaconda
 	- Used for DDSP timbre transfer and some AMT runs.
2. Ubuntu (RTX 3060, 16 GB RAM):
	- Used for MT3 (transformer) experiments due to TensorFlow / JAX / GPU setup.

You can adapt this to a single machine if your environment supports all dependencies.

### 4.1 Clone the Repository

```
git clone https://github.com/martinoywa/singing-voice-amt.git
cd singing-voice-amt
```

### 4.2 Set Up Python Environments

Timbre Transfer (DDSP)
```
python3.11 -m venv .venv-tt
source .venv-tt/bin/activate
pip install --upgrade pip
pip install -r requirements-macOS.txt
```

MT3 / AMT Experiments
```
python3.8 -m venv .venv-mt3
source .venv-mt3/bin/activate
pip install --upgrade pip
pip install -r requirements-Ubuntu.txt
```
The requirements files include core libraries such as DDSP, TensorFlow/JAX, librosa, pretty_midi, MIR-Eval, OpenL3, and the relevant AMT libraries (e.g. MT3, Basic Pitch, etc.). See the requirements-*.txt files for the precise versions.


## 5. End-to-End Pipeline

This section summarises how to go from raw Annotated-VocalSet files to evaluation results.

### 5.1 Step 1 – Data Engineering

Notebook: data_engineering.ipynb

Tasks:  ￼
1. Filter excerpts:
	- Select only straight singing styles (e.g. “caro”, “dona”, “row”) with natural voice and minimal stylistic variation.
2. Generate MIDI:
	- Parse extended_2 CSV annotations.
	- Create MIDI notes using:
	- Onset time
	- Offset time
	- MIDI pitch
	- Ignore rows labelled as non-sound events (rests, transitions).
	- Apply a fixed velocity for all notes (e.g. 100) for simplicity.
3.	Align audio & MIDI:
	- Copy the corresponding .wav files into a local folder (e.g. dataset/wav/).
	- Save generated .mid files into dataset/midi/.
4.	Build metadata. Create dataset/metadata.csv with at least:
	- excerpt_name (e.g. m6_row_straight)
	- wav_path (e.g. dataset/wav/m6_row_straight.wav)
	- midi_path (e.g. dataset/midi/m6_row_straight.mid)

### 5.2 Step 2 – Timbre Transfer (DDSP)

Notebook: timbre_transfer.ipynb
Helpers: timbretransfer/  ￼
1.	Download pre-trained DDSP models:
	- Violin, Flute, Trumpet, Tenor Saxophone
	- Place checkpoints under pretrained/<instrument_name>/:
		1. operative_config-0.gin
		2. ckpt-*
2. Run the notebook:
	- For each entry in metadata.csv:
		1.	Load 16 kHz mono audio (dataset/wav/<name>.wav).
		2.	Extract DDSP features: f0, loudness, confidence, etc.
		3.	Normalise/auto-tune f0 and loudness using dataset statistics.
		4.	Synthesize with the chosen instrument model.
		3.	Output layout

```
tt_dataset/
├── violin/
│   ├── m6_row_straight.wav
│   └── ...
├── flute/
├── trumpet/
└── tenor_saxophone/
```

On the original hardware, each 33-second excerpt took ~1.5 minutes to convert per instrument.  ￼

### 5.3 Step 3 – Automatic Music Transcription (AMT)

Notebook examples:
1. Music_Transcription_with_Transformers.ipynb – MT3 experiments

The dissertation’s main AMT systems are:  ￼
1. Basic Pitch (instrument-agnostic, lightweight AMT)
2. MT3 (transformer-based, multi-instrument AMT)

You will typically:
1.	Transcribe baseline vocal audio
	- For each dataset/wav/<name>.wav:
		1. Basic Pitch → amt_dataset/baselines/basic_pitch/<name>.mid
		2. Run MT3 → amt_dataset/baselines/MT3/<name>.mid
2. Transcribe timbre-transferred audio
	- For each instrument and each file in tt_dataset/<instrument>/<name>.wav:
		1. Run Basic Pitch → amt_dataset/converted/basic_pitch/<instrument>/<name>.mid
		2. Run MT3 → amt_dataset/converted/MT3/<instrument>/<name>.mid
3. Keep sample rate consistent (16 kHz is used across DDSP and AMT in the project).  ￼

### 5.4 Step 4 – Evaluation

Scripts in evaluation/ compute:
1.	Note-level AMT metrics (using MIR-Eval):
	- Precision, Recall, F1
	- Onset F1
	- Onset-Offset F1
	- Transcription Error Rate (TER)
	- Typical onset tolerance: 50 ms
	- Offset tolerance: 20% of note duration
2.	Timbre fidelity metrics:
	- MFCC similarity (cosine similarity of time-aggregated MFCCs)
	- Embedding similarity using OpenL3 (cosine similarity of music embeddings)  ￼
3.	Listening test utilities:
	- Simple randomised selection of original vs converted excerpts.
	- Allows a binary “musically similar note flow? (Yes/No)” rating.

⸻

## 6. Key Results (Summary)

High-level findings from the dissertation:  ￼
1. Baseline vocal AMT:
	- Both Basic Pitch and MT3 underperform their reported piano/instrument benchmarks on singing voice.
	- Average note-level F1 ≈ 0.33–0.34, Onset-Offset F1 ≈ 0.18–0.20.
	- MT3 generally has more stable Onset F1 and lower TER; Basic Pitch sometimes has slightly higher note-level F1.
2. Converted (timbre-transferred) AMT:
	- Tenor Saxophone is the most AMT-friendly timbre:
		1. Best Basic Pitch F1 ≈ 0.20
		2. Best MT3 F1 ≈ 0.14
		3. Onset F1 > 0.5 for both models.
	- Flute and Violin timbres yield very low F1 (near zero for some settings).
	- Overall, timbre transfer does not outperform baseline vocal transcription when averaged over all excerpts and metrics; F1 drops and TER rises compared to raw vocals.
3. Timbre fidelity vs transcription accuracy:
	- MFCC similarity is a poor predictor of AMT performance (weak/non-significant correlations).
	- Embedding similarity (OpenL3) correlates strongly and positively with AMT metrics (r ≳ 0.9).
	- Increasing timbre fidelity beyond a certain threshold (~0.96–0.97 MFCC similarity) shows diminishing or even negative returns on F1.
4. Listening test:
	- Only 2/5 randomly selected converted excerpts were judged as clearly similar in “note flow” to the original vocals, indicating residual artefacts and noisy sustains in some conversions.

⸻

## 7. How to Reproduce the Dissertation Experiments
1.	Obtain datasets
	- Download Annotated-VocalSet extended CSVs and corresponding WAVs from Zenodo.
3.	Generate MIDI + metadata
	- Run data_engineering.ipynb to create:
		1. dataset/midi/
  		2. dataset/wav/
    	3. dataset/metadata.csv
4.	Run timbre transfer
	- Set up DDSP environment (e.g. macOS).
 	- Download pre-trained DDSP models and place them under pretrained/<instrument>/.
  	- Run timbre_transfer.ipynb → tt_dataset/<instrument>/.
6.	Run AMT models
	- Set up an environment for MT3 (e.g. Ubuntu + GPU).
 	- Use Basic Pitch + MT3 to transcribe:
  	- dataset/wav/ → amt_dataset/baselines/...
   	- tt_dataset/... → amt_dataset/converted/...
7.	Evaluate
	- Use scripts in evaluation/ to compute:
 		1. Per-file metrics, aggregated results tables & plots to compare:
   			- Baseline vs converted
      		- Different timbres (Violin, Flute, Trumpet, Tenor Sax)
8.	Optional
	- Run the listening test utilities to replicate the small subjective evaluation.

⸻

## 8. Citation

If you use this code or reproduce parts of the pipeline, please cite the dissertation:

```
@mastersthesis{oywa2025timbretransfer,
  author       = {Bonface Martin Oywa},
  title        = {Timbre Transfer of the Human Singing Voice for Effective Automatic Music Transcription},
  school       = {University of East London / UNICAF},
  year         = {2025},
  month        = {December}
}
```


⸻

## 9. License

This repository is released under the CC0-1.0 license (public domain dedication). See LICENSE for details.

