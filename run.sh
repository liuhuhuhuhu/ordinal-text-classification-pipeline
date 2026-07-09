#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running ordinal classification pipeline..."
python src/pipeline.py

echo "Submission files generated:"
ls -lh predictions_baseline.csv
ls -lh predictions_best.csv

echo "Finished successfully."
