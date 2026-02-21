"""Topic database operations."""

from typing import List, Dict
from psycopg2.extras import execute_values


class TopicMixin:
    """Topic-related database operations."""

    def store_topics(self, topics: List[Dict], result_version_id: str):
        """Store discovered topics for a specific version."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            for topic in topics:
                cur.execute(f"""
                    INSERT INTO {schema}.topics
                    (topic_id, result_version_id, parent_topic_id, name, description, keywords, article_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (topic_id, result_version_id) DO UPDATE SET
                        parent_topic_id = EXCLUDED.parent_topic_id,
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        article_count = EXCLUDED.article_count
                """, (
                    topic["topic_id"],
                    result_version_id,
                    topic.get("parent_topic_id"),
                    topic["name"],
                    topic.get("description"),
                    topic.get("keywords", []),
                    topic.get("article_count", 0)
                ))

    def store_article_topics(self, assignments: List[Dict], result_version_id: str):
        """Store topic assignments for articles for a specific version."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            # First, build a mapping from BERTopic topic_id to database id
            cur.execute(
                f"""
                SELECT id, topic_id FROM {schema}.topics
                WHERE result_version_id = %s
                """,
                (result_version_id,)
            )
            topic_id_to_db_id = {row[1]: row[0] for row in cur.fetchall()}

            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.article_analysis
                (article_id, result_version_id, primary_topic_id, topic_confidence)
                VALUES %s
                ON CONFLICT (article_id, result_version_id) DO UPDATE SET
                    primary_topic_id = EXCLUDED.primary_topic_id,
                    topic_confidence = EXCLUDED.topic_confidence
                """,
                [(a["article_id"], result_version_id, topic_id_to_db_id.get(a["topic_id"]), a.get("confidence", 0.0))
                 for a in assignments]
            )

    def get_topic_for_article(self, article_id: int, version_id: str) -> Dict:
        """Fetch topic assignment for article.

        Args:
            article_id: The article ID
            version_id: The topic version ID

        Returns:
            Topic dict with name and confidence or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT t.id as topic_id, t.topic_id as bertopic_id, t.name as topic_name,
                       aa.topic_confidence
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE aa.article_id = %s AND aa.result_version_id = %s
                  AND t.result_version_id = %s
            """, (article_id, version_id, version_id))
            return cur.fetchone()

    def get_articles_by_topic(self, topic_id: int, version_id: str) -> List[Dict]:
        """Fetch all articles assigned to a specific topic.

        Args:
            topic_id: The topic ID (from topics table, not BERTopic topic_id)
            version_id: The result version ID

        Returns:
            List of article dicts with id, title, content, source_id, date_posted
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT a.id, a.title, a.content, a.source_id, a.date_posted
                FROM {schema}.news_articles a
                JOIN {schema}.article_analysis aa ON a.id = aa.article_id
                WHERE aa.primary_topic_id = %s
                  AND aa.result_version_id = %s
                  AND a.is_ditwah_cyclone = 1
                  AND a.date_posted >= '2025-11-22' AND a.date_posted <= '2025-12-31'
                ORDER BY a.date_posted
            """, (topic_id, version_id))
            return cur.fetchall()

    def get_all_topics_with_counts(
        self,
        version_id: str,
        min_article_count: int = 3
    ) -> List[Dict]:
        """Get all topics with article counts above threshold.

        Args:
            version_id: The result version ID
            min_article_count: Minimum articles per topic (default: 3)

        Returns:
            List of topic dicts with id, topic_id, name, keywords, article_count
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT t.id, t.topic_id, t.name, t.keywords, t.article_count
                FROM {schema}.topics t
                WHERE t.result_version_id = %s
                  AND t.topic_id != -1
                  AND t.article_count >= %s
                ORDER BY t.article_count DESC
            """, (version_id, min_article_count))
            return cur.fetchall()
