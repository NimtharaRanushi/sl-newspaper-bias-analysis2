#!/usr/bin/env python3
"""Discover topics using BERTopic."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topics import discover_topics


def main():
    print("=" * 60)
    print("Topic Discovery with BERTopic")
    print("=" * 60)
    print()

    # Discover topics (auto number)
    summary = discover_topics(nr_topics=None, save_model=True)

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


if __name__ == "__main__":
    main()
