"""Topic modeling using Stochastic Block Model (SBM) via Spectral Co-clustering.

Stochastic Block Models discover block structure in networks. For topic modeling,
we build a word-document bipartite graph where blocks correspond to topics.

This implementation uses Spectral Co-clustering (sklearn), which is equivalent
to fitting a flat SBM on the bipartite word-document graph. It simultaneously
clusters both documents (rows) and words (columns) of the TF-IDF matrix,
producing coherent topics where words and documents within a block share
similar co-occurrence patterns.

Reference: Dhillon (2001) "Co-clustering documents and words using bipartite
spectral graph partitioning."
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import SpectralCoclustering
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.sparse import issparse

from .db import get_db, load_config, ditwah_filters
from .preprocessing import DOMAIN_STOP_WORDS, TOKEN_PATTERN, clean_text


class SBMTopicModeler:
    """Discovers topics via Stochastic Block Model on the word-document bipartite graph.

    Spectral Co-clustering decomposes the TF-IDF matrix using SVD and then
    applies k-means in the resulting spectral embedding space. This is
    mathematically equivalent to inferring block structure in the bipartite
    word-document graph, analogous to a flat degree-corrected SBM.
    """

    def __init__(
        self,
        nr_topics: int = 20,
        min_df: int = 5,
        max_df: float = 0.95,
        max_features: int = 8000,
        stop_words: List[str] = None,
        random_seed: int = 42,
        n_svd_vecs: Optional[int] = None,
        mini_batch: bool = False,
    ):
        self.nr_topics = nr_topics
        self.random_seed = random_seed

        custom_stop_words = sorted(
            set(ENGLISH_STOP_WORDS) | set(DOMAIN_STOP_WORDS) | set(stop_words or [])
        )

        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=custom_stop_words,
            ngram_range=(1, 1),  # Unigrams only for co-clustering stability
            sublinear_tf=True,
            token_pattern=TOKEN_PATTERN,
        )

        cocluster_kwargs = {
            "n_clusters": nr_topics,
            "random_state": random_seed,
            "mini_batch": mini_batch,
        }
        if n_svd_vecs is not None:
            cocluster_kwargs["n_svd_vecs"] = n_svd_vecs

        self.model = SpectralCoclustering(**cocluster_kwargs)
        self.feature_names = None
        self.tfidf_matrix = None
        self.doc_labels = None
        self.word_labels = None

    def fit(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        """Fit SBM model and return (topic_assignments, confidences).

        Args:
            documents: List of article texts

        Returns:
            topic_assignments: Block (topic) index for each document
            topic_confidences: Normalised intra-block TF-IDF mass as confidence
        """
        print(f"Building TF-IDF matrix for {len(documents)} documents...")
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.feature_names)}, Matrix shape: {self.tfidf_matrix.shape}")

        print(f"Fitting Spectral Co-clustering SBM with {self.nr_topics} blocks...")
        self.model.fit(self.tfidf_matrix)

        self.doc_labels = self.model.row_labels_.tolist()
        self.word_labels = self.model.column_labels_.tolist()

        # Confidence: average TF-IDF weight of words in the document's block
        doc_confidences = self._compute_confidences()
        n_topics_used = len(set(self.doc_labels))
        print(f"Topics (blocks) used: {n_topics_used}")
        return self.doc_labels, doc_confidences

    def _compute_confidences(self) -> List[float]:
        """Compute per-document confidence as intra-block TF-IDF fraction."""
        confidences = []
        X = self.tfidf_matrix
        word_labels = np.array(self.word_labels)

        for doc_idx, topic_id in enumerate(self.doc_labels):
            row = X[doc_idx]
            if issparse(row):
                row = row.toarray().flatten()
            else:
                row = np.asarray(row).flatten()

            total = row.sum()
            if total == 0:
                confidences.append(0.0)
                continue
            intra_block = row[word_labels == topic_id].sum()
            confidences.append(float(intra_block / total))

        return confidences

    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[str]:
        """Get top keywords for a topic ranked by average TF-IDF within the block."""
        if self.feature_names is None or self.tfidf_matrix is None:
            return []

        word_labels = np.array(self.word_labels)
        doc_labels = np.array(self.doc_labels)

        # Words in this block
        word_indices = np.where(word_labels == topic_id)[0]
        if len(word_indices) == 0:
            return []

        # Documents in this block
        doc_indices = np.where(doc_labels == topic_id)[0]
        if len(doc_indices) == 0:
            return [self.feature_names[i] for i in word_indices[:n_words]]

        # Submatrix: docs in block × words in block
        submatrix = self.tfidf_matrix[np.ix_(doc_indices, word_indices)]
        if issparse(submatrix):
            submatrix = submatrix.toarray()

        avg_tfidf = submatrix.mean(axis=0)
        top_relative = np.argsort(avg_tfidf)[::-1][:n_words]
        return [self.feature_names[word_indices[i]] for i in top_relative]

    def get_topic_info(self) -> List[Dict]:
        """Get info (article count + keywords) for all topics."""
        if self.doc_labels is None:
            return []
        doc_labels = np.array(self.doc_labels)
        topic_counts = np.bincount(doc_labels, minlength=self.nr_topics)
        return [
            {
                "topic_id": i,
                "count": int(topic_counts[i]),
                "keywords": self.get_topic_keywords(i),
            }
            for i in range(self.nr_topics)
        ]



def _label_topics(modeler: SBMTopicModeler) -> List[Dict]:
    """Generate topic labels from SBM block keywords."""
    labeled_topics = []
    print("Generating SBM topic labels from block keywords...")
    for info in modeler.get_topic_info():
        keywords = info["keywords"]
        name = ", ".join(keywords[:5]).title() if keywords else f"Topic {info['topic_id']}"
        labeled_topics.append(
            {
                "topic_id": info["topic_id"],
                "name": name,
                "description": f"Articles about: {', '.join(keywords[:5])}",
                "keywords": keywords,
                "article_count": info["count"],
            }
        )
    return labeled_topics


def discover_sbm_topics(
    result_version_id: str,
    topic_config: Dict = None,
) -> Dict:
    """Discover topics from the article corpus using Stochastic Block Model.

    Args:
        result_version_id: UUID of the result version
        topic_config: Topic configuration dict (from version config)

    Returns:
        Summary dict with topic counts and labels
    """
    if topic_config is None:
        config = load_config()
        topic_config = config.get("topics", {})

    sbm_params = topic_config.get("sbm", {})
    nr_topics = topic_config.get("nr_topics", 20)
    stop_words = topic_config.get("stop_words", ["sri", "lanka", "lankan"])

    print("Loading articles from database...")
    with get_db() as db:
        articles = db.get_articles(filters=ditwah_filters())

    if not articles:
        raise ValueError("No articles found in the database.")

    print(f"Loaded {len(articles)} articles")
    documents = [clean_text(f"{a['title']}\n\n{a['content'][:8000]}") for a in articles]
    article_ids = [str(a["id"]) for a in articles]

    modeler = SBMTopicModeler(
        nr_topics=nr_topics,
        stop_words=stop_words,
        min_df=sbm_params.get("min_df", 5),
        max_df=sbm_params.get("max_df", 0.95),
        max_features=sbm_params.get("max_features", 8000),
        mini_batch=sbm_params.get("mini_batch", False),
        n_svd_vecs=sbm_params.get("n_svd_vecs"),
    )

    topic_assignments, topic_confidences = modeler.fit(documents)
    labeled_topics = _label_topics(modeler)

    print("Saving SBM topics to database...")
    with get_db() as db:
        db.store_topics(labeled_topics, result_version_id)

        assignments = [
            {
                "article_id": article_ids[i],
                "topic_id": topic_assignments[i],
                "confidence": float(topic_confidences[i]),
            }
            for i in range(len(article_ids))
        ]
        db.store_article_topics(assignments, result_version_id)

    n_topics = len(labeled_topics)
    print(f"\nSBM Topic Discovery Complete:")
    print(f"  Total articles: {len(documents)}")
    print(f"  Topics (blocks): {n_topics}")

    return {
        "total_articles": len(documents),
        "topics_discovered": n_topics,
        "topics": labeled_topics,
    }
