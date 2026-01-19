#!/usr/bin/env python3
"""Generate embeddings for all articles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import generate_embeddings, get_embedding_stats


def main():
    # Show current stats
    stats = get_embedding_stats()
    print(f"\nCurrent status:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  With embeddings: {stats['with_embeddings']}")
    print(f"  Remaining: {stats['without_embeddings']}")
    print(f"  Progress: {stats['completion_pct']}%\n")

    if stats['without_embeddings'] == 0:
        print("All articles already have embeddings!")
        return

    # Generate embeddings
    print("Starting embedding generation...")
    print("(This uses OpenAI's text-embedding-3-large model)\n")

    generate_embeddings(batch_size=100)

    # Show final stats
    stats = get_embedding_stats()
    print(f"\nFinal status: {stats['completion_pct']}% complete")


if __name__ == "__main__":
    main()
