"""Word frequency database operations."""

from typing import List, Dict
from psycopg2.extras import execute_values


class WordFrequencyMixin:
    """Word frequency database operations."""

    def store_word_frequencies(self, frequencies: List[Dict], result_version_id: str):
        """Store word frequency results for a specific version.

        Args:
            frequencies: List of dicts with 'source_id', 'word', 'frequency', 'tfidf_score', 'rank'
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.word_frequencies
                (result_version_id, source_id, word, frequency, tfidf_score, rank)
                VALUES %s
                ON CONFLICT (result_version_id, source_id, word) DO UPDATE SET
                    frequency = EXCLUDED.frequency,
                    tfidf_score = EXCLUDED.tfidf_score,
                    rank = EXCLUDED.rank,
                    created_at = NOW()
                """,
                [
                    (
                        result_version_id,
                        f["source_id"],
                        f["word"],
                        f["frequency"],
                        f.get("tfidf_score"),
                        f["rank"]
                    )
                    for f in frequencies
                ]
            )

    def get_word_frequencies(
        self,
        result_version_id: str,
        source_id: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get word frequencies for a specific version and optional source.

        Args:
            result_version_id: UUID of the result version
            source_id: Optional source filter
            limit: Maximum number of words to return per source

        Returns:
            List of dicts with word frequency data
        """
        schema = self.config["schema"]
        params = [result_version_id]

        query = f"""
            SELECT source_id, word, frequency, tfidf_score, rank
            FROM {schema}.word_frequencies
            WHERE result_version_id = %s
        """

        if source_id:
            query += " AND source_id = %s"
            params.append(source_id)

        query += " AND rank <= %s ORDER BY source_id, rank"
        params.append(limit)

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
