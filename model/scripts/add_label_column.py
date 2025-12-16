#!/usr/bin/env python3
"""
CLI script for adding label column to data
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labeling import add_label_column

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add empty label column for expert annotator.")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--date", required=False)

    args = parser.parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    add_label_column(args.input, args.output_dir, date_str)
