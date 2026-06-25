#!/bin/bash

# Usage: ./run_generate_training_data.sh [CASE_NAME] [INPUT_DIM] [NUM_SAMPLES]
# Defaults match the previous hardcoded configuration (case14, 3D, 5000 samples)

export CASE_NAME="${1:-case14}"
export INPUT_DIM="${2:-3}"
export NUM_SAMPLES="${3:-5000}"

# Call MATLAB and run the script
matlab -nodisplay -nosplash -r "try, generate_training_data; catch e, disp(getReport(e)), exit(1); end; exit(0);"
