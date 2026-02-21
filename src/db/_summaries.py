"""Summary database operations."""

from typing import List, Dict


class SummaryMixin:
    """Summary-related database operations."""

    def get_summary_for_article(self, article_id: int, version_id: str) -> Dict:
        """Fetch summary for article and version.

        Args:
            article_id: The article ID
            version_id: The summarization version ID

        Returns:
            Summary dict or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT summary_text, method, compression_ratio, word_count,
                       sentence_count, summary_length, processing_time_ms
                FROM {schema}.article_summaries
                WHERE article_id = %s AND result_version_id = %s
            """, (article_id, version_id))
            return cur.fetchone()

    def get_multi_doc_summary(self, group_type: str, group_id: str, version_id: str, source_version_id: str):
        """Get multi-document summary for a topic or cluster.

        Args:
            group_type: 'topic' or 'cluster'
            group_id: Topic ID or cluster ID
            version_id: Multi-doc summarization version ID
            source_version_id: Topic or cluster version ID

        Returns:
            Dict with summary data or None if not found
        """
        schema = self.config['schema']
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT
                    id,
                    summary_text,
                    method,
                    llm_model,
                    article_count,
                    source_count,
                    word_count,
                    processing_time_ms,
                    created_at
                FROM {schema}.multi_doc_summaries
                WHERE group_type = %s
                  AND group_id = %s
                  AND result_version_id = %s
                  AND source_version_id = %s
            """, (group_type, group_id, version_id, source_version_id))
            return cur.fetchone()

    def store_multi_doc_summary(self, group_type: str, group_id: str, version_id: str, source_version_id: str,
                               summary_text: str, method: str, llm_model: str,
                               article_count: int, source_count: int,
                               word_count: int, processing_time_ms: int) -> str:
        """Store a multi-document summary.

        Args:
            version_id: Multi-doc summarization version ID
            source_version_id: Topic or cluster version ID

        Returns:
            UUID of created summary
        """
        schema = self.config['schema']
        with self.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {schema}.multi_doc_summaries
                (group_type, group_id, result_version_id, source_version_id, summary_text, method, llm_model,
                 article_count, source_count, word_count, processing_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (group_type, group_id, result_version_id, source_version_id)
                DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    method = EXCLUDED.method,
                    llm_model = EXCLUDED.llm_model,
                    article_count = EXCLUDED.article_count,
                    source_count = EXCLUDED.source_count,
                    word_count = EXCLUDED.word_count,
                    processing_time_ms = EXCLUDED.processing_time_ms,
                    created_at = NOW()
                RETURNING id
            """, (group_type, group_id, version_id, source_version_id, summary_text, method, llm_model,
                  article_count, source_count, word_count, processing_time_ms))
            return str(cur.fetchone()['id'])
