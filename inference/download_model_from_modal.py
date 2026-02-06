"""
Download the merged Trinity-Mini model weights from Modal volume to local disk.

Prerequisites:
    pip install modal
    modal setup  # authenticate with Modal

Usage:
    python download_model_from_modal.py
    python download_model_from_modal.py --output-dir ./my-model
"""

import argparse
import os

import modal

VOLUME_NAME = "arcee-vol"
MODEL_PATH_IN_VOLUME = "models/trinity-mini-merged"
DEFAULT_OUTPUT_DIR = "./trinity-mini"


def download_model(volume_name: str, model_path: str, output_dir: str):
    """Download model files from a Modal volume to local disk."""
    print(f"Connecting to Modal volume: {volume_name}")
    vol = modal.Volume.from_name(volume_name)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Listing files in {model_path}/ ...")
    entries = list(vol.listdir(model_path, recursive=True))

    if not entries:
        print(f"ERROR: No files found at '{model_path}' in volume '{volume_name}'.")
        print("Check that the volume name and path are correct.")
        return

    print(f"Found {len(entries)} entries. Downloading...")

    for entry in entries:
        remote_path = entry.path
        rel_path = os.path.relpath(remote_path, model_path)
        local_path = os.path.join(output_dir, rel_path)

        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            os.makedirs(local_path, exist_ok=True)
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"  Downloading: {rel_path}")
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(remote_path):
                f.write(chunk)

    print(f"\nDone! Model downloaded to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download merged Trinity-Mini model from Modal volume")
    parser.add_argument(
        "--volume-name", default=VOLUME_NAME,
        help=f"Modal volume name (default: {VOLUME_NAME})",
    )
    parser.add_argument(
        "--model-path", default=MODEL_PATH_IN_VOLUME,
        help=f"Path inside volume (default: {MODEL_PATH_IN_VOLUME})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    download_model(args.volume_name, args.model_path, args.output_dir)
