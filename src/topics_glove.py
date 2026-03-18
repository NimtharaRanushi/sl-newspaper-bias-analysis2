"""Topic modeling using Global Vectors (GloVe) word embeddings + k-means clustering.

GloVe (Pennington et al., 2014) produces dense word representations from
global word co-occurrence statistics. For topic modeling we:
  1. Load pre-trained GloVe word vectors via gensim's model zoo.
  2. Represent each document as the TF-IDF-weighted average of its word vectors.
  3. Apply k-means clustering on the resulting document vectors.
  4. Label each topic cluster by finding the vocabulary words whose GloVe
     vectors are nearest to the cluster centroid.

This surface-level GloVe document representation captures semantic similarity
better than bag-of-words approaches and produces geometrically meaningful clusters.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

from .db import get_db, load_config, ditwah_filters
from .preprocessing import DOMAIN_STOP_WORDS, TOKEN_PATTERN, clean_text


_GLOVE_MODELS = {
    "glove-wiki-gigaword-50": 50,
    "glove-wiki-gigaword-100": 100,
    "glove-wiki-gigaword-200": 200,
    "glove-wiki-gigaword-300": 300,
    "glove-twitter-25": 25,
    "glove-twitter-50": 50,
    "glove-twitter-100": 100,
    "glove-twitter-200": 200,
}


class GloVeTopicModeler:
    """Discovers topics via GloVe document embeddings + k-means clustering."""

    def __init__(
        self,
        nr_topics: int = 20,
        pretrained_model: str = "glove-wiki-gigaword-100",
        min_df: int = 3,
        max_df: float = 0.95,
        stop_words: List[str] = None,
        random_seed: int = 42,
        kmeans_n_init: int = 10,
        kmeans_max_iter: int = 300,
        top_n_label_words: int = 20,
    ):
        self.nr_topics = nr_topics
        self.pretrained_model = pretrained_model
        self.random_seed = random_seed
        self.top_n_label_words = top_n_label_words

        custom_stop_words = sorted(
            set(ENGLISH_STOP_WORDS) | set(DOMAIN_STOP_WORDS) | set(stop_words or [])
        )

        # TF-IDF used for weighting word vectors during document representation
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=custom_stop_words,
            ngram_range=(1, 1),  # Only unigrams can be looked up in GloVe vocab
            token_pattern=TOKEN_PATTERN,
        )

        self.kmeans = KMeans(
            n_clusters=nr_topics,
            random_state=random_seed,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
        )

        self.word_vectors = None       # gensim KeyedVectors
        self.feature_names = None      # TF-IDF vocabulary
        self.doc_vectors = None        # (n_docs, dim) document embeddings
        self.cluster_labels = None     # (n_docs,) cluster assignments
        self.cluster_centroids = None  # (k, dim) centroids

    def _load_glove(self):
        """Download and cache pre-trained GloVe vectors via gensim."""
        try:
            import gensim.downloader as gensim_dl
        except ImportError:
            raise ImportError(
                "gensim is required for GloVe topic modeling. "
                "Install it with: pip install gensim"
            )

        if self.pretrained_model not in _GLOVE_MODELS:
            raise ValueError(
                f"Unknown GloVe model '{self.pretrained_model}'. "
                f"Available models: {list(_GLOVE_MODELS)}"
            )

        print(f"Loading pre-trained GloVe model '{self.pretrained_model}' "
              f"(will download ~{_GLOVE_MODELS[self.pretrained_model] * 4 // 10}MB on first run)...")
        self.word_vectors = gensim_dl.load(self.pretrained_model)
        print(f"Loaded {len(self.word_vectors)} word vectors "
              f"(dim={self.word_vectors.vector_size})")

    def _build_doc_vectors(self, tfidf_matrix, feature_names) -> np.ndarray:
        """Build TF-IDF-weighted GloVe document vectors.

        For each document, compute the weighted average of GloVe vectors for
        the words in its TF-IDF representation. Words not in GloVe vocab are
        skipped; documents with no coverage get zero vectors.
        """
        from scipy.sparse import issparse

        dim = self.word_vectors.vector_size
        n_docs = tfidf_matrix.shape[0]
        doc_vecs = np.zeros((n_docs, dim), dtype=np.float32)

        # Pre-fetch GloVe vectors for in-vocabulary words
        word_vecs = {}
        for word in feature_names:
            if word in self.word_vectors:
                word_vecs[word] = self.word_vectors[word]

        coverage = len(word_vecs) / len(feature_names) * 100
        print(f"GloVe vocabulary coverage: {len(word_vecs)}/{len(feature_names)} words ({coverage:.1f}%)")

        for doc_idx in range(n_docs):
            row = tfidf_matrix[doc_idx]
            if issparse(row):
                row = row.toarray().flatten()
            else:
                row = np.asarray(row).flatten()

            weighted_sum = np.zeros(dim, dtype=np.float32)
            weight_total = 0.0

            for word_idx in np.nonzero(row)[0]:
                word = feature_names[word_idx]
                if word in word_vecs:
                    w = float(row[word_idx])
                    weighted_sum += w * word_vecs[word]
                    weight_total += w

            if weight_total > 0:
                doc_vecs[doc_idx] = weighted_sum / weight_total

        zero_docs = np.sum(np.all(doc_vecs == 0, axis=1))
        if zero_docs > 0:
            print(f"  Note: {zero_docs} documents had no GloVe coverage (assigned zero vector).")

        return doc_vecs

    def fit(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        """Fit GloVe topic model and return (topic_assignments, confidences).

        Args:
            documents: List of article texts

        Returns:
            topic_assignments: Cluster (topic) index for each document
            topic_confidences: Normalised inverse distance to centroid (0–1)
        """
        self._load_glove()

        print(f"Building TF-IDF matrix for {len(documents)} documents...")
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.feature_names)}")

        print("Building GloVe document vectors...")
        self.doc_vectors = self._build_doc_vectors(tfidf_matrix, self.feature_names)

        # Normalise document vectors before k-means (cosine-style clustering)
        doc_vecs_norm = normalize(self.doc_vectors, norm="l2")

        print(f"Clustering {len(documents)} documents into {self.nr_topics} topics with k-means...")
        self.cluster_labels = self.kmeans.fit_predict(doc_vecs_norm).tolist()
        self.cluster_centroids = self.kmeans.cluster_centers_

        # Confidence: 1 - normalised distance to assigned centroid
        distances = self.kmeans.transform(doc_vecs_norm)  # (n_docs, k)
        assigned_distances = np.array([distances[i, self.cluster_labels[i]] for i in range(len(documents))])
        max_dist = assigned_distances.max() if assigned_distances.max() > 0 else 1.0
        confidences = (1.0 - assigned_distances / max_dist).tolist()

        n_topics_used = len(set(self.cluster_labels))
        print(f"Topics used: {n_topics_used}")
        return self.cluster_labels, confidences

    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[str]:
        """Get top keywords for a topic by finding GloVe vocab words nearest to centroid."""
        if self.cluster_centroids is None:
            return []

        centroid = self.cluster_centroids[topic_id]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

        # Find nearest words in GloVe vocabulary (cosine similarity)
        similar_words = self.word_vectors.similar_by_vector(centroid_norm, topn=self.top_n_label_words)

        # Filter stop words, short tokens, numeric tokens, and domain noise
        combined_stop = set(ENGLISH_STOP_WORDS) | set(DOMAIN_STOP_WORDS)
        keywords = [
            word for word, _ in similar_words
            if word not in combined_stop
            and len(word) >= 3
            and word.isalpha()          # pure alphabetic only
        ]
        return keywords[:n_words]

    def get_topic_info(self) -> List[Dict]:
        """Get info (article count + keywords) for all topics."""
        if self.cluster_labels is None:
            return []
        labels = np.array(self.cluster_labels)
        topic_counts = np.bincount(labels, minlength=self.nr_topics)
        return [
            {
                "topic_id": i,
                "count": int(topic_counts[i]),
                "keywords": self.get_topic_keywords(i),
            }
            for i in range(self.nr_topics)
        ]


def _label_topics(modeler: GloVeTopicModeler) -> List[Dict]:
    """Generate topic labels from GloVe centroid nearest words."""
    labeled_topics = []
    print("Generating GloVe topic labels from centroid nearest words...")
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


def discover_glove_topics(
    result_version_id: str,
    topic_config: Dict = None,
) -> Dict:
    """Discover topics from the article corpus using GloVe embeddings + k-means.

    Args:
        result_version_id: UUID of the result version
        topic_config: Topic configuration dict (from version config)

    Returns:
        Summary dict with topic counts and labels
    """
    if topic_config is None:
        config = load_config()
        topic_config = config.get("topics", {})

    glove_params = topic_config.get("glove", {})
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

    modeler = GloVeTopicModeler(
        nr_topics=nr_topics,
        stop_words=stop_words,
        pretrained_model=glove_params.get("pretrained_model", "glove-wiki-gigaword-100"),
        min_df=glove_params.get("min_df", 3),
        max_df=glove_params.get("max_df", 0.95),
        kmeans_n_init=glove_params.get("kmeans_n_init", 10),
        kmeans_max_iter=glove_params.get("kmeans_max_iter", 300),
        top_n_label_words=glove_params.get("top_n_label_words", 20),
    )

    topic_assignments, topic_confidences = modeler.fit(documents)
    labeled_topics = _label_topics(modeler)

    print("Saving GloVe topics to database...")
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
    print(f"\nGloVe Topic Discovery Complete:")
    print(f"  Total articles: {len(documents)}")
    print(f"  Topics: {n_topics}")

    return {
        "total_articles": len(documents),
        "topics_discovered": n_topics,
        "topics": labeled_topics,
    }
