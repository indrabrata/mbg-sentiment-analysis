#!/usr/bin/env python3
"""
CLI script for data preprocessing
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import clean_text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleaning text MBG (Scraped Twitter).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    clean_text(args.input, args.output_dir)
