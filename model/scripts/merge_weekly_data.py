#!/usr/bin/env python3
"""
CLI script for merging weekly data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.merge_data import merge_weekly

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge weekly labeled data")
    parser.add_argument("--input_dir", default="./data/daily")
    parser.add_argument("--output_dir", default="./data/weekly")

    args = parser.parse_args()
    merge_weekly(args.input_dir, args.output_dir)
