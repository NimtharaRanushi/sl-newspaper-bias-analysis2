"""Chunk-level topic database operations."""

from typing import List, Dict, Optional
from psycopg2.extras import execute_values


class ChunkTopicMixin:
    """Chunk-level topic database operations."""

    def store_chunks(self, chunks: List[Dict], result_version_id: str):
        """Store article chunks in batch.

        Args:
            chunks: List of dicts with article_id, chunk_index, start_char, end_char
            result_version_id: UUID of the version
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.article_chunks
                (article_id, result_version_id, chunk_index, start_char, end_char)
                VALUES %s
                ON CONFLICT (article_id, result_version_id, chunk_index) DO UPDATE SET
                    start_char = EXCLUDED.start_char,
                    end_char = EXCLUDED.end_char
                """,
                [(
                    c["article_id"],
                    result_version_id,
                    c["chunk_index"],
                    c["start_char"],
                    c["end_char"],
                ) for c in chunks]
            )

    def get_chunks_for_version(self, result_version_id: str) -> List[Dict]:
        """Get all chunks for a version with article metadata.

        Chunk text is reconstructed from article content via SUBSTRING.

        Args:
            result_version_id: UUID of the version

        Returns:
            List of chunk dicts with id, article_id, source_id, chunk_index, chunk_text
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT c.id, c.article_id, c.chunk_index,
                       SUBSTRING(a.content FROM c.start_char + 1 FOR c.end_char - c.start_char) AS chunk_text,
                       c.start_char, c.end_char,
                       a.source_id, a.title, a.date_posted
                FROM {schema}.article_chunks c
                JOIN {schema}.news_articles a ON c.article_id = a.id
                WHERE c.result_version_id = %s
                ORDER BY a.date_posted, a.id, c.chunk_index
            """, (result_version_id,))
            return cur.fetchall()

    def store_chunk_topics(self, topics: List[Dict], result_version_id: str):
        """Store chunk-level topics.

        Args:
            topics: List of dicts with topic_id, name, description, keywords, chunk_count
            result_version_id: UUID of the version
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            for topic in topics:
                cur.execute(f"""
                    INSERT INTO {schema}.chunk_topics
                    (topic_id, result_version_id, name, description, keywords, chunk_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (topic_id, result_version_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        chunk_count = EXCLUDED.chunk_count
                """, (
                    topic["topic_id"],
                    result_version_id,
                    topic["name"],
                    topic.get("description"),
                    topic.get("keywords", []),
                    topic.get("chunk_count", 0)
                ))

    def store_chunk_topic_assignments(self, assignments: List[Dict], result_version_id: str):
        """Store chunk-to-topic assignments.

        Args:
            assignments: List of dicts with chunk_id, topic_id (BERTopic), confidence
            result_version_id: UUID of the version
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            # Build mapping from BERTopic topic_id to database serial id
            cur.execute(
                f"SELECT id, topic_id FROM {schema}.chunk_topics WHERE result_version_id = %s",
                (result_version_id,)
            )
            topic_id_to_db_id = {row[1]: row[0] for row in cur.fetchall()}

            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.chunk_topic_assignments
                (chunk_id, result_version_id, topic_id, confidence)
                VALUES %s
                ON CONFLICT (chunk_id, result_version_id) DO UPDATE SET
                    topic_id = EXCLUDED.topic_id,
                    confidence = EXCLUDED.confidence
                """,
                [(
                    a["chunk_id"],
                    result_version_id,
                    topic_id_to_db_id.get(a["topic_id"]),
                    a.get("confidence", 0.0),
                ) for a in assignments]
            )

    def get_chunk_topics_with_counts(
        self,
        version_id: str,
        min_count: int = 3
    ) -> List[Dict]:
        """Get all chunk topics with counts above threshold.

        Args:
            version_id: The result version ID
            min_count: Minimum chunks per topic

        Returns:
            List of topic dicts with id, topic_id, name, keywords, chunk_count
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, topic_id, name, description, keywords, chunk_count
                FROM {schema}.chunk_topics
                WHERE result_version_id = %s
                  AND topic_id != -1
                  AND chunk_count >= %s
                ORDER BY chunk_count DESC
            """, (version_id, min_count))
            return cur.fetchall()

    def get_chunks_by_topic(
        self,
        topic_db_id: int,
        version_id: str,
        source_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get chunks assigned to a specific topic.

        Chunk text is reconstructed from article content via SUBSTRING.

        Args:
            topic_db_id: The database serial ID from chunk_topics table
            version_id: The result version ID
            source_id: Optional filter by source
            limit: Maximum chunks to return

        Returns:
            List of dicts with chunk_text, article title, source_id, date_posted, url, chunk_index
        """
        schema = self.config["schema"]

        conditions = [
            "cta.topic_id = %s",
            "cta.result_version_id = %s",
        ]
        params = [topic_db_id, version_id]

        if source_id:
            conditions.append("a.source_id = %s")
            params.append(source_id)

        params.append(limit)
        where = " AND ".join(conditions)

        with self.cursor() as cur:
            cur.execute(f"""
                SELECT SUBSTRING(a.content FROM c.start_char + 1 FOR c.end_char - c.start_char) AS chunk_text,
                       c.chunk_index,
                       a.title, a.source_id, a.date_posted, a.url,
                       cta.confidence
                FROM {schema}.chunk_topic_assignments cta
                JOIN {schema}.article_chunks c ON cta.chunk_id = c.id
                JOIN {schema}.news_articles a ON c.article_id = a.id
                WHERE {where}
                ORDER BY cta.confidence DESC, a.date_posted DESC
                LIMIT %s
            """, params)
            return cur.fetchall()

    def get_chunk_topic_by_source(self, version_id: str) -> List[Dict]:
        """Get chunk counts per topic per source.

        Args:
            version_id: The result version ID

        Returns:
            List of dicts with topic_name, topic_id (BERTopic), source_id, count
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT ct.name as topic_name, ct.topic_id, a.source_id, COUNT(*) as count
                FROM {schema}.chunk_topic_assignments cta
                JOIN {schema}.chunk_topics ct ON cta.topic_id = ct.id
                JOIN {schema}.article_chunks c ON cta.chunk_id = c.id
                JOIN {schema}.news_articles a ON c.article_id = a.id
                WHERE cta.result_version_id = %s
                  AND ct.topic_id != -1
                GROUP BY ct.name, ct.topic_id, a.source_id
                ORDER BY count DESC
            """, (version_id,))
            return cur.fetchall()

    def get_chunk_topic_stats(self, version_id: str) -> Dict:
        """Get summary statistics for chunk topic analysis.

        Args:
            version_id: The result version ID

        Returns:
            Dict with total_chunks, total_articles, total_topics, outlier_count
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            # Total chunks
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.article_chunks WHERE result_version_id = %s",
                (version_id,)
            )
            total_chunks = cur.fetchone()["count"]

            # Total articles
            cur.execute(
                f"SELECT COUNT(DISTINCT article_id) as count FROM {schema}.article_chunks WHERE result_version_id = %s",
                (version_id,)
            )
            total_articles = cur.fetchone()["count"]

            # Total topics (excluding outliers)
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.chunk_topics WHERE result_version_id = %s AND topic_id != -1",
                (version_id,)
            )
            total_topics = cur.fetchone()["count"]

            # Outlier count (chunks with topic_id -1)
            cur.execute(f"""
                SELECT COUNT(*) as count FROM {schema}.chunk_topics ct
                WHERE ct.result_version_id = %s AND ct.topic_id = -1
            """, (version_id,))
            outlier_row = cur.fetchone()
            outlier_count = outlier_row["count"] if outlier_row else 0

            # Get actual outlier chunk count from the chunk_count column
            cur.execute(f"""
                SELECT chunk_count FROM {schema}.chunk_topics
                WHERE result_version_id = %s AND topic_id = -1
            """, (version_id,))
            outlier_row = cur.fetchone()
            outlier_chunks = outlier_row["chunk_count"] if outlier_row else 0

        return {
            "total_chunks": total_chunks,
            "total_articles": total_articles,
            "total_topics": total_topics,
            "outlier_chunks": outlier_chunks,
        }

    def get_outlier_chunks(
        self,
        version_id: str,
        source_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get chunks assigned to the outlier topic (topic_id = -1).

        Args:
            version_id: The result version ID
            source_id: Optional filter by source
            limit: Maximum chunks to return

        Returns:
            List of dicts with chunk_text, article title, source_id, date_posted, url, chunk_index
        """
        schema = self.config["schema"]

        conditions = [
            "c.result_version_id = %s",
            "cta.id IS NULL",
        ]
        params: list = [version_id]

        if source_id:
            conditions.append("a.source_id = %s")
            params.append(source_id)

        params.append(limit)
        where = " AND ".join(conditions)

        with self.cursor() as cur:
            cur.execute(f"""
                SELECT SUBSTRING(a.content FROM c.start_char + 1 FOR c.end_char - c.start_char) AS chunk_text,
                       c.chunk_index,
                       a.title, a.source_id, a.date_posted, a.url,
                       NULL AS confidence
                FROM {schema}.article_chunks c
                LEFT JOIN {schema}.chunk_topic_assignments cta
                    ON cta.chunk_id = c.id AND cta.result_version_id = c.result_version_id
                JOIN {schema}.news_articles a ON c.article_id = a.id
                WHERE {where}
                ORDER BY a.date_posted DESC
                LIMIT %s
            """, params)
            return cur.fetchall()

    def update_chunk_topic_description(self, topic_db_id: int, description_json: str):
        """Update the description field for a chunk topic.

        Args:
            topic_db_id: The database serial ID from chunk_topics table
            description_json: JSON string with claim label and description
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            cur.execute(
                f"UPDATE {schema}.chunk_topics SET description = %s WHERE id = %s",
                (description_json, topic_db_id)
            )

    def get_chunk_outlet_totals(self, version_id: str) -> Dict[str, int]:
        """Get total chunk counts (including outliers/unassigned chunks) per outlet.

        Args:
            version_id: The result version ID

        Returns:
            Dict mapping source_id to chunk count
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT a.source_id, COUNT(*) as count
                FROM {schema}.article_chunks c
                JOIN {schema}.news_articles a ON c.article_id = a.id
                WHERE c.result_version_id = %s
                GROUP BY a.source_id
            """, (version_id,))
            return {row["source_id"]: row["count"] for row in cur.fetchall()}
