#!/usr/bin/env python3
"""
CLI script for MLflow model registry operations
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from src.utils.env import load_env_file
load_env_file()

from src.mlops.model_registry import setup_mlflow_env, download_latest_model, push_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download latest model")
    download_parser.add_argument("--model_name", default="MBGModel")
    download_parser.add_argument("--target_dir", default="models/previous")

    # Push command
    push_parser = subparsers.add_parser("push", help="Push model to registry")
    push_parser.add_argument("--model_dir", required=True)
    push_parser.add_argument("--model_name", default="MBGModel")
    push_parser.add_argument("--stage", default="Staging")

    args = parser.parse_args()
    setup_mlflow_env()

    if args.command == "download":
        download_latest_model(args.model_name, args.target_dir)
    elif args.command == "push":
        push_model(args.model_dir, args.model_name, args.stage)
    else:
        parser.print_help()
