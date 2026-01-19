"""Embedding generation for articles."""

from typing import List, Dict
from tqdm import tqdm

from .db import Database, get_db
from .llm import get_embeddings_client, load_config


def generate_embeddings(
    batch_size: int = 50,
    limit: int = None,
    show_progress: bool = True
):
    """
    Generate embeddings for all articles that don't have them yet.

    Args:
        batch_size: Number of articles to process per batch
        limit: Maximum articles to process (None = all)
        show_progress: Show progress bar
    """
    config = load_config()
    embed_client = get_embeddings_client(config.get("embeddings"))

    with get_db() as db:
        # Get articles without embeddings
        articles = db.get_articles_without_embeddings(limit=limit)
        total = len(articles)

        if total == 0:
            print("All articles already have embeddings.")
            return

        print(f"Generating embeddings for {total} articles...")

        # Process in batches
        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches")

        processed = 0
        for i in iterator:
            batch = articles[i:i + batch_size]

            # Prepare texts: title + content
            texts = [
                f"{a['title']}\n\n{a['content'][:8000]}"  # Truncate long content
                for a in batch
            ]

            embeddings = embed_client.embed(texts)

            # Prepare for storage
            embedding_records = [
                {
                    "article_id": str(batch[j]["id"]),
                    "embedding": embeddings[j],
                    "model": embed_client.model
                }
                for j in range(len(batch))
            ]

            db.store_embeddings(embedding_records)
            processed += len(batch)

        print(f"Generated embeddings for {processed} articles.")


def get_embedding_stats() -> Dict:
    """Get statistics about embeddings."""
    with get_db() as db:
        total_articles = db.get_article_count()
        embedded = db.get_embedding_count()

        return {
            "total_articles": total_articles,
            "with_embeddings": embedded,
            "without_embeddings": total_articles - embedded,
            "completion_pct": round(100 * embedded / total_articles, 1) if total_articles > 0 else 0
        }


if __name__ == "__main__":
    # Show current stats
    stats = get_embedding_stats()
    print(f"Current status: {stats['with_embeddings']}/{stats['total_articles']} articles embedded ({stats['completion_pct']}%)")

    # Generate remaining
    generate_embeddings()
