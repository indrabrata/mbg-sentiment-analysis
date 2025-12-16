#!/usr/bin/env python3
"""
CLI script for training models
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from src.utils.env import load_env_file
load_env_file()

# Import the train function from the training module
from src.training.trainer import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training CSV (weekly expert-labelled)")
    parser.add_argument("--val", required=True, help="Validation CSV")
    parser.add_argument("--model_dir", required=True, help="Previous model directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for new model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--freeze_layers", type=int, default=6)

    args = parser.parse_args()
    train(args)
