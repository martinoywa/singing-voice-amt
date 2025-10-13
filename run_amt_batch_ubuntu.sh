#!/bin/bash
# This script runs the AMT transcription for multiple instruments using a Python script.
# Ensure the script is executable: chmod +x run_amt_batch_ubuntu.sh
# Ensure you have the necessary Python environment activated before running this script.
# Usage: ./run_amt_batch_ubuntu.sh
# This script was created to process multiple instruments in a batch for Ubuntu since it takes a long time to run. ~13 minutes per instrument.

# Base paths
PYTHON_EXEC="/home/martinoywa/miniconda3/envs/svamt/bin/python"
SCRIPT_PATH="/home/martinoywa/Developer/singing-voice-amt/timbretransfer/amt.py"
INPUT_BASE="tt_dataset"
OUTPUT_BASE="amt_dataset/converted/MT3"

# Instruments list
INSTRUMENTS=("flute" "tenor_saxophone" "trumpet" "violin")

# Loop through instruments and run the command
for instrument in "${INSTRUMENTS[@]}"; do
    echo "Processing $instrument..."
    INPUT_PATH="$INPUT_BASE/$instrument"
    OUTPUT_PATH="$OUTPUT_BASE/$instrument"

    $PYTHON_EXEC "$SCRIPT_PATH" \
        --metadata-file-path "$INPUT_PATH/" \
        --file-type path \
        --output-dir "$OUTPUT_PATH/"

    echo "✅ Finished $instrument"
    echo "-----------------------------------"
done

echo "🎵 All instruments processed successfully."
