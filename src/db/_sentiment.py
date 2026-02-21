"""Sentiment analysis database operations."""

import json
from typing import List, Dict
from psycopg2.extras import execute_values


class SentimentMixin:
    """Sentiment analysis database operations."""

    def store_sentiment_analyses(self, analyses: List[Dict]):
        """Store sentiment analysis results in batch.

        Args:
            analyses: List of dicts with sentiment analysis results
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.sentiment_analyses
                (article_id, model_type, model_name, overall_sentiment, overall_confidence,
                 headline_sentiment, headline_confidence, sentiment_reasoning,
                 sentiment_aspects, processing_time_ms)
                VALUES %s
                ON CONFLICT (article_id, model_type) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    overall_sentiment = EXCLUDED.overall_sentiment,
                    overall_confidence = EXCLUDED.overall_confidence,
                    headline_sentiment = EXCLUDED.headline_sentiment,
                    headline_confidence = EXCLUDED.headline_confidence,
                    sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                    sentiment_aspects = EXCLUDED.sentiment_aspects,
                    processing_time_ms = EXCLUDED.processing_time_ms,
                    processed_at = NOW()
                """,
                [
                    (
                        a["article_id"],
                        a["model_type"],
                        a.get("model_name"),
                        a["overall_sentiment"],
                        a.get("overall_confidence"),
                        a["headline_sentiment"],
                        a.get("headline_confidence"),
                        a.get("sentiment_reasoning"),
                        json.dumps(a.get("sentiment_aspects")) if a.get("sentiment_aspects") else None,
                        a.get("processing_time_ms")
                    )
                    for a in analyses
                ],
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )

    def get_articles_without_sentiment(
        self,
        model_type: str,
        limit: int = None
    ) -> List[Dict]:
        """Get articles that haven't been analyzed for sentiment with given model.

        Args:
            model_type: Type of model ('llm', 'local', 'hybrid')
            limit: Maximum number of articles to return
        """
        schema = self.config["schema"]
        query = f"""
            SELECT a.id, a.title, a.content, a.date_posted, a.source_id
            FROM {schema}.news_articles a
            LEFT JOIN {schema}.sentiment_analyses sa
                ON a.id = sa.article_id AND sa.model_type = %s
            WHERE sa.id IS NULL
              AND a.content IS NOT NULL
              AND a.content != ''
              AND a.is_ditwah_cyclone = 1
              AND a.date_posted >= '2025-11-22' AND a.date_posted <= '2025-12-31'
            ORDER BY a.date_posted
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, (model_type,))
            return cur.fetchall()

    def get_sentiment_by_model(
        self,
        model_type: str = None,
        source_id: str = None,
        limit: int = None
    ) -> List[Dict]:
        """Get sentiment analysis results.

        Args:
            model_type: Filter by model type ('llm', 'local', 'hybrid')
            source_id: Filter by news source
            limit: Maximum number of results
        """
        schema = self.config["schema"]
        query = f"""
            SELECT sa.*, a.title, a.source_id, a.date_posted
            FROM {schema}.sentiment_analyses sa
            JOIN {schema}.news_articles a ON sa.article_id = a.id
            WHERE a.is_ditwah_cyclone = 1
              AND a.date_posted >= '2025-11-22' AND a.date_posted <= '2025-12-31'
        """
        params = []

        if model_type:
            query += " AND sa.model_type = %s"
            params.append(model_type)

        if source_id:
            query += " AND a.source_id = %s"
            params.append(source_id)

        query += " ORDER BY a.date_posted DESC"

        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_sentiment_comparison(self, limit: int = 100) -> List[Dict]:
        """Get sentiment results from all models for comparison.

        Args:
            limit: Maximum number of articles to compare
        """
        schema = self.config["schema"]
        query = f"""
            SELECT
                a.id as article_id,
                a.title,
                a.source_id,
                a.date_posted,
                MAX(CASE WHEN sa.model_type = 'llm' THEN sa.overall_sentiment END) as llm_sentiment,
                MAX(CASE WHEN sa.model_type = 'local' THEN sa.overall_sentiment END) as local_sentiment,
                MAX(CASE WHEN sa.model_type = 'hybrid' THEN sa.overall_sentiment END) as hybrid_sentiment,
                MAX(CASE WHEN sa.model_type = 'llm' THEN sa.sentiment_reasoning END) as llm_reasoning
            FROM {schema}.news_articles a
            JOIN {schema}.sentiment_analyses sa ON a.id = sa.article_id
            WHERE a.is_ditwah_cyclone = 1
              AND a.date_posted >= '2025-11-22' AND a.date_posted <= '2025-12-31'
            GROUP BY a.id, a.title, a.source_id, a.date_posted
            HAVING COUNT(DISTINCT sa.model_type) >= 2
            ORDER BY a.date_posted DESC
            LIMIT {limit}
        """

        with self.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    def get_sentiment_stats(self, model_type: str) -> Dict:
        """Get statistics for sentiment analysis by model type.

        Args:
            model_type: Type of model ('llm', 'local', 'hybrid')
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) as total_analyzed,
                    AVG(overall_sentiment) as avg_sentiment,
                    STDDEV(overall_sentiment) as stddev_sentiment,
                    MIN(overall_sentiment) as min_sentiment,
                    MAX(overall_sentiment) as max_sentiment,
                    AVG(overall_confidence) as avg_confidence
                FROM {schema}.sentiment_analyses
                WHERE model_type = %s
            """, (model_type,))
            return cur.fetchone()

    def refresh_sentiment_summary(self):
        """Refresh the sentiment_summary materialized view."""
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"REFRESH MATERIALIZED VIEW {schema}.sentiment_summary")

    def get_sentiment_for_article(self, article_id: int, model_type: str = 'roberta') -> Dict:
        """Fetch sentiment analysis for article and model.

        Args:
            article_id: The article ID
            model_type: The sentiment model type (default: roberta)

        Returns:
            Sentiment dict or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT overall_sentiment, headline_sentiment, overall_confidence,
                       headline_confidence, sentiment_reasoning, model_type, model_name
                FROM {schema}.sentiment_analyses
                WHERE article_id = %s AND model_type = %s
            """, (article_id, model_type))
            return cur.fetchone()
