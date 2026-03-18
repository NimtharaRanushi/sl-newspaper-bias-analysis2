"""Topic modeling using Latent Dirichlet Allocation (LDA)."""

import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from .db import get_db, load_config, ditwah_filters
from .preprocessing import DOMAIN_STOP_WORDS, TOKEN_PATTERN, clean_text


class LDATopicModeler:
    """Discovers topics using Latent Dirichlet Allocation.

    LDA is a generative probabilistic model that treats each document as a
    mixture of topics, and each topic as a distribution over words.
    """

    def __init__(
        self,
        nr_topics: int = 20,
        min_df: int = 5,
        max_df: float = 0.95,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        stop_words: List[str] = None,
        random_seed: int = 42,
        max_iter: int = 25,
        learning_method: str = "online",
        doc_topic_prior: Optional[float] = None,
        topic_word_prior: Optional[float] = None,
    ):
        self.nr_topics = nr_topics
        self.random_seed = random_seed

        custom_stop_words = sorted(
            set(ENGLISH_STOP_WORDS) | set(DOMAIN_STOP_WORDS) | set(stop_words or [])
        )

        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=custom_stop_words,
            token_pattern=TOKEN_PATTERN,
        )

        lda_kwargs = {
            "n_components": nr_topics,
            "random_state": random_seed,
            "max_iter": max_iter,
            "learning_method": learning_method,
            "verbose": 1,
        }
        if doc_topic_prior is not None:
            lda_kwargs["doc_topic_prior"] = doc_topic_prior
        if topic_word_prior is not None:
            lda_kwargs["topic_word_prior"] = topic_word_prior

        self.model = LatentDirichletAllocation(**lda_kwargs)
        self.feature_names = None
        self.doc_topic_matrix = None

    def fit(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        """Fit LDA model and return (topic_assignments, confidences).

        Args:
            documents: List of article texts

        Returns:
            topic_assignments: Dominant topic index for each document
            topic_confidences: Topic probability for the assigned topic
        """
        print(f"Building document-term matrix for {len(documents)} documents...")
        dtm = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.feature_names)}, Matrix shape: {dtm.shape}")

        print(f"Fitting LDA with {self.nr_topics} topics (max_iter={self.model.max_iter})...")
        self.doc_topic_matrix = self.model.fit_transform(dtm)

        topic_assignments = np.argmax(self.doc_topic_matrix, axis=1).tolist()
        topic_confidences = np.max(self.doc_topic_matrix, axis=1).tolist()

        n_topics_used = len(set(topic_assignments))
        print(f"Topics discovered: {n_topics_used} (of {self.nr_topics} requested)")
        return topic_assignments, topic_confidences

    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[str]:
        """Get top keywords for a topic by word-topic probability."""
        if self.feature_names is None:
            return []
        topic_word_dist = self.model.components_[topic_id]
        top_indices = np.argsort(topic_word_dist)[::-1][:n_words]
        return [self.feature_names[i] for i in top_indices]

    def get_topic_info(self) -> List[Dict]:
        """Get info (article count + keywords) for all topics."""
        if self.doc_topic_matrix is None:
            return []
        topic_assignments = np.argmax(self.doc_topic_matrix, axis=1)
        topic_counts = np.bincount(topic_assignments, minlength=self.nr_topics)
        return [
            {
                "topic_id": i,
                "count": int(topic_counts[i]),
                "keywords": self.get_topic_keywords(i),
            }
            for i in range(self.nr_topics)
        ]


def _label_topics(modeler: LDATopicModeler) -> List[Dict]:
    """Generate topic labels from LDA keyword distributions."""
    labeled_topics = []
    print("Generating LDA topic labels from keywords...")
    for info in modeler.get_topic_info():
        keywords = info["keywords"]
        name = ", ".join(keywords[:5]).title()
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


def discover_lda_topics(
    result_version_id: str,
    topic_config: Dict = None,
) -> Dict:
    """Discover topics from the article corpus using LDA.

    Args:
        result_version_id: UUID of the result version
        topic_config: Topic configuration dict (from version config)

    Returns:
        Summary dict with topic counts and labels
    """
    if topic_config is None:
        config = load_config()
        topic_config = config.get("topics", {})

    lda_params = topic_config.get("lda", {})
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

    modeler = LDATopicModeler(
        nr_topics=nr_topics,
        stop_words=stop_words,
        min_df=lda_params.get("min_df", 5),
        max_df=lda_params.get("max_df", 0.95),
        max_features=lda_params.get("max_features", 10000),
        ngram_range=tuple(lda_params.get("ngram_range", [1, 2])),
        max_iter=lda_params.get("max_iter", 25),
        learning_method=lda_params.get("learning_method", "online"),
        doc_topic_prior=lda_params.get("doc_topic_prior"),
        topic_word_prior=lda_params.get("topic_word_prior"),
    )

    topic_assignments, topic_confidences = modeler.fit(documents)
    labeled_topics = _label_topics(modeler)

    print("Saving LDA topics to database...")
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
    print(f"\nLDA Topic Discovery Complete:")
    print(f"  Total articles: {len(documents)}")
    print(f"  Topics: {n_topics}")

    return {
        "total_articles": len(documents),
        "topics_discovered": n_topics,
        "topics": labeled_topics,
    }
