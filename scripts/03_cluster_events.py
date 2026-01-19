#!/usr/bin/env python3
"""Cluster articles into events."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering import cluster_articles, get_cluster_stats
from src.db import load_config


def main():
    print("=" * 60)
    print("Event Clustering")
    print("=" * 60)
    print()

    config = load_config()
    cluster_config = config.get("clustering", {})

    summary = cluster_articles(
        similarity_threshold=cluster_config.get("similarity_threshold", 0.8),
        time_window_days=cluster_config.get("time_window_days", 7),
        min_cluster_size=cluster_config.get("min_cluster_size", 2)
    )

    print("\n" + "=" * 60)
    print("Top Event Clusters:")
    print("=" * 60)

    stats = get_cluster_stats()
    for cluster in stats["top_clusters"]:
        print(f"\n{cluster['cluster_name'][:80]}...")
        print(f"  Articles: {cluster['article_count']}, Sources: {cluster['sources_count']}")
        if cluster['date_start'] and cluster['date_end']:
            print(f"  Date range: {cluster['date_start']} to {cluster['date_end']}")


if __name__ == "__main__":
    main()
