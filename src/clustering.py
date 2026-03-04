"""Event clustering: store pairwise similarity edges for dynamic runtime clustering."""

import numpy as np
from typing import Dict, List, Set
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .db import get_db, ditwah_filters


def cluster_articles(
    result_version_id: str,
    storage_threshold: float = 0.5,
    embeddings_config: dict = None
) -> Dict:
    """
    Compute pairwise cosine similarities and store all edges above storage_threshold.

    No cluster assignment happens here — clustering is computed at dashboard render time
    using the stored edges, with user-controlled threshold and date-window sliders.

    Args:
        result_version_id: UUID of the result version
        storage_threshold: Minimum similarity to persist (default 0.5); the UI
            slider will only allow values >= storage_threshold
        embeddings_config: Embeddings configuration dict (contains model name)

    Returns:
        Summary dict with total_articles, total_edges, storage_threshold
    """
    embedding_model = (embeddings_config or {}).get("model", "all-mpnet-base-v2")
    filters = ditwah_filters()
    print(f"Loading Ditwah articles with embeddings for model '{embedding_model}'...")
    with get_db() as db:
        data = db.get_all_embeddings(embedding_model=embedding_model, filters=filters)

    if len(data) == 0:
        print(f"No embeddings found for model '{embedding_model}'. Generating automatically...")
        from .embeddings import generate_embeddings
        generate_embeddings(embedding_model=embedding_model, embeddings_config=embeddings_config)
        with get_db() as db:
            data = db.get_all_embeddings(embedding_model=embedding_model, filters=filters)

    print(f"Loaded {len(data)} articles")

    article_ids = [str(d['article_id']) for d in data]
    embeddings = np.array([d['embedding'] for d in data])

    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    print(f"Extracting pairs with similarity >= {storage_threshold}...")
    edges = []
    n = len(article_ids)
    for i in tqdm(range(n), desc="Extracting edges"):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim >= storage_threshold:
                id_a, id_b = article_ids[i], article_ids[j]
                # Enforce id_a < id_b to satisfy the CHECK constraint
                if id_a > id_b:
                    id_a, id_b = id_b, id_a
                edges.append((id_a, id_b, sim))

    print(f"Found {len(edges)} edges above threshold {storage_threshold}")

    print("Storing edges to database...")
    with get_db() as db:
        db.store_similarity_edges(edges, result_version_id)

    summary = {
        "total_articles": n,
        "total_edges": len(edges),
        "storage_threshold": storage_threshold,
    }

    print("\nEdge Storage Complete:")
    print(f"  Total articles: {summary['total_articles']}")
    print(f"  Total edges stored: {summary['total_edges']}")
    print(f"  Storage threshold: {summary['storage_threshold']}")

    return summary


def compute_clusters_from_edges(edges, min_cluster_size: int = 2):
    """Compute connected components (event clusters) from similarity edges.

    Uses networkx connected components, equivalent to Union-Find.

    Args:
        edges: Iterable of edge dicts (or tuples) with article_id_a, article_id_b,
               similarity_score. Dict keys or positional access both work.
        min_cluster_size: Minimum component size to include

    Returns:
        List of sets, each set containing article ID strings for one cluster.
        Sorted by size descending.
    """
    G = nx.Graph()
    for edge in edges:
        if isinstance(edge, dict):
            id_a = str(edge["article_id_a"])
            id_b = str(edge["article_id_b"])
            score = float(edge["similarity_score"])
        else:
            id_a, id_b, score = str(edge[0]), str(edge[1]), float(edge[2])
        G.add_edge(id_a, id_b, weight=score)

    components = [c for c in nx.connected_components(G) if len(c) >= min_cluster_size]
    components.sort(key=len, reverse=True)
    return components, G


def get_cluster_stats() -> Dict:
    """Get statistics about stored similarity edges (legacy compat)."""
    with get_db() as db:
        with db.cursor() as cur:
            schema = db.config["schema"]
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.article_similarity_edges")
            total_edges = cur.fetchone()["count"]

    return {
        "total_edges": total_edges,
    }


if __name__ == "__main__":
    print("Please use scripts/clustering/02_cluster_events.py instead.")
    print("Usage: python3 scripts/clustering/02_cluster_events.py --version-id <uuid>")
