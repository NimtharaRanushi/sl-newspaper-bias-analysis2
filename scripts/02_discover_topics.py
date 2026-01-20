#!/usr/bin/env python3
"""Discover topics using BERTopic for a specific result version."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topics import discover_topics
from src.versions import get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Discover topics using BERTopic")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the result version"
    )
    parser.add_argument(
        "--nr-topics",
        type=int,
        default=None,
        help="Target number of topics (default: auto)"
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Don't save the trained model"
    )
    args = parser.parse_args()

    # Get version configuration
    version_config = get_version_config(args.version_id)
    if not version_config:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    print("=" * 60)
    print("Topic Discovery with BERTopic")
    print("=" * 60)
    print(f"Version: {args.version_id}")
    print()

    # Extract topic configuration
    topic_config = version_config.get("topics", {})

    # Discover topics
    summary = discover_topics(
        result_version_id=args.version_id,
        topic_config=topic_config,
        nr_topics=args.nr_topics,
        save_model=not args.no_save_model
    )

    # Print discovered topics
    print("\n" + "=" * 60)
    print("Discovered Topics:")
    print("=" * 60)

    for topic in sorted(summary["topics"], key=lambda x: x["article_count"], reverse=True):
        if topic["topic_id"] == -1:
            continue
        print(f"\n[Topic {topic['topic_id']}] {topic['name']}")
        print(f"  Articles: {topic['article_count']}")
        print(f"  Keywords: {', '.join(topic['keywords'][:5])}")
        if topic["description"]:
            print(f"  Description: {topic['description']}")

    # Update pipeline status
    update_pipeline_status(args.version_id, "topics", True)
    print(f"\n✓ Topics step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
