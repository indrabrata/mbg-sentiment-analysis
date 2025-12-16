#!/usr/bin/env python3
"""
CLI script for model evaluation
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import evaluate_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model and export metrics")
    parser.add_argument("--test_path", required=True, help="Path to test CSV")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation results")

    args = parser.parse_args()
    evaluate_model(args.test_path, args.model_dir, args.output_dir)
