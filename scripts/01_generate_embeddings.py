#!/usr/bin/env python3
"""Generate embeddings for all articles for a specific result version."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import generate_embeddings
from src.versions import get_version_config, update_pipeline_status
from src.db import get_db


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for articles")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the result version"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)"
    )
    args = parser.parse_args()

    # Get version configuration
    version_config = get_version_config(args.version_id)
    if not version_config:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    # Extract parameters from config
    embeddings_config = version_config.get("embeddings", {})
    random_seed = version_config.get("random_seed", 42)

    print(f"\nGenerating embeddings for version: {args.version_id}")
    print(f"  Model: {embeddings_config.get('model', 'all-mpnet-base-v2')}")
    print(f"  Random seed: {random_seed}")
    print(f"  Batch size: {args.batch_size}\n")

    # Generate embeddings
    generate_embeddings(
        result_version_id=args.version_id,
        batch_size=args.batch_size,
        random_seed=random_seed
    )

    # Update pipeline status
    update_pipeline_status(args.version_id, "embeddings", True)
    print(f"\n✓ Embeddings step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
