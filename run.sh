#!/bin/bash

set -e

echo "Starting ordinal text classification pipeline..."

echo "Step 1: Cleaning raw data"
python src/clean_data.py

echo "Step 2: Training baseline model and generating baseline predictions"
python src/train_baseline.py

echo "Step 3: Training best continuous-score model and generating best predictions"
python src/train_best_model.py

echo "Pipeline completed successfully."
echo "Generated files:"
echo "- predictions_baseline.csv"
echo "- predictions_best.csv"
