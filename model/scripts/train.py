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

    parser = argparse.ArgumentParser(
        description="""
        Automated IndoBERTweet Training Pipeline

        This script automatically:
        - Checks MLflow for previous models
        - If no previous model: trains from scratch (fresh training)
        - If previous model exists: downloads and does incremental training with layer freezing
        - Splits data into train/validation automatically
        - Logs all artifacts to MLflow
        - Saves model with timestamp

        All configuration (epochs, batch_size, freeze_layers, etc.) is in .env file
        """
    )
    parser.add_argument(
        "--train_data",
        required=True,
        help="Path to training CSV (will be split into train/val automatically)"
    )

    args = parser.parse_args()
    train(args)
