#!/bin/bash
# This script runs the Timbre Fidelity for multiple instruments using a Python script.
# Ensure the script is executable: chmod +x run_amt_batch_ubuntu.sh
# Ensure you have the necessary Python environment activated before running this script.
# Usage: ./run_amt_batch_ubuntu.sh
# This script was created to process multiple instruments in a batch since it takes a long time to run. ~50 minutes per instrument.

# Base paths
PYTHON_EXEC="/opt/anaconda3/envs/timbre-fidelity/bin/python"
SCRIPT_PATH="/Users/martinoywa/Developer/singing-voice-amt/evaluation/timbre_fidelity.py"
INPUT_BASE="../dataset/wav/"
PREDICTED_BASE="../tt_dataset"

# Instruments list
INSTRUMENTS=("flute" "tenor_saxophone" "trumpet" "violin")

# Loop through instruments and run the command
for instrument in "${INSTRUMENTS[@]}"; do
    echo "Processing $instrument..."
    PREDICTED_PATH="$PREDICTED_BASE/$instrument/"
    OUTPUT_CSV_PATH="timbre_fidelity_results_$instrument.csv"

#    echo "... $PREDICTED_PATH... $OUTPUT_CSV_PATH"

    $PYTHON_EXEC "$SCRIPT_PATH" \
        --ground-truth-dir "$INPUT_BASE/" \
        --predicted-dir "$PREDICTED_PATH/" \
        --output-csv "$OUTPUT_CSV_PATH"

    echo "✅ Finished $instrument"
    echo "-----------------------------------"
done

echo "🎵 All instruments processed successfully."
