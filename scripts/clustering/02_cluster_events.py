#!/usr/bin/env python3
"""Cluster articles into events for clustering analysis."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clustering import cluster_articles
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Cluster articles into events")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the clustering result version"
    )
    args = parser.parse_args()

    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "clustering":
        print(f"Error: Version {args.version_id} is not a clustering analysis version (type: {version['analysis_type']})")
        print("Use scripts/clustering/ for clustering analysis versions only")
        sys.exit(1)

    version_config = get_version_config(args.version_id)

    print("=" * 60)
    print("Event Clustering")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print()

    cluster_config = version_config.get("clustering", {})
    embeddings_config = version_config.get("embeddings", {})

    summary = cluster_articles(
        result_version_id=args.version_id,
        storage_threshold=cluster_config.get("storage_threshold", 0.5),
        embeddings_config=embeddings_config
    )

    print("\n" + "=" * 60)
    print("Edge Storage Summary:")
    print("=" * 60)
    print(f"Total articles: {summary['total_articles']}")
    print(f"Total edges stored: {summary['total_edges']}")
    print(f"Storage threshold: {summary['storage_threshold']}")

    # Update pipeline status
    update_pipeline_status(args.version_id, "clustering", True)
    print(f"\n✓ Clustering step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
