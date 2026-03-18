#!/usr/bin/env python3
"""Discover topics using Latent Dirichlet Allocation (LDA).

Usage:
    python3 scripts/topics/03_lda_topics.py --version-id <uuid>

Create a version first:
    python3 -c "
    from src.versions import create_version, get_default_lda_config
    vid = create_version('lda-baseline', 'LDA topic analysis', get_default_lda_config(), 'topics')
    print('Version ID:', vid)
    "
"""

import os
import sys
import argparse
from pathlib import Path

# Single-threaded for reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.topics_lda import discover_lda_topics
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Discover topics using LDA")
    parser.add_argument("--version-id", required=True, help="UUID of the topic result version")
    args = parser.parse_args()

    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "topics":
        print(f"Error: Version {args.version_id} is not a topic analysis version "
              f"(type: {version['analysis_type']})")
        sys.exit(1)

    version_config = get_version_config(args.version_id)

    print("=" * 60)
    print("Topic Discovery with LDA")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print()

    topic_config = version_config.get("topics", {})
    summary = discover_lda_topics(
        result_version_id=args.version_id,
        topic_config=topic_config,
    )

    print("\n" + "=" * 60)
    print("Discovered Topics:")
    print("=" * 60)
    for topic in sorted(summary["topics"], key=lambda x: x["article_count"], reverse=True):
        print(f"\n[Topic {topic['topic_id']}] {topic['name']}")
        print(f"  Articles: {topic['article_count']}")
        print(f"  Keywords: {', '.join(topic['keywords'][:5])}")

    update_pipeline_status(args.version_id, "topics", True)
    print(f"\n✓ Topics step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
