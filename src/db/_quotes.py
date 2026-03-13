"""Quote extraction database operations."""

from typing import List, Dict, Any, Optional


class QuotesMixin:
    """Quote extraction database operations."""

    def store_quotes(
        self,
        article_id: str,
        version_id: str,
        quotes: List[Dict[str, Any]]
    ) -> int:
        """Store extracted quotes for an article.

        Args:
            article_id: UUID of the article
            version_id: UUID of the result version
            quotes: List of dicts with keys: quote_content, quote_source, cue, quote_type, quote_order

        Returns:
            Number of quotes inserted
        """
        schema = self.config["schema"]
        inserted = 0

        with self.cursor() as cur:
            for q in quotes:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.article_quotes
                    (article_id, result_version_id, quote_content, quote_source, cue, quote_type, quote_order)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, result_version_id, quote_order) DO NOTHING
                    """,
                    (
                        article_id,
                        version_id,
                        q["quote_content"],
                        q.get("quote_source"),
                        q.get("cue"),
                        q["quote_type"],
                        q["quote_order"]
                    )
                )
                inserted += cur.rowcount

        return inserted

    def get_quotes_for_article(
        self,
        article_id: str,
        version_id: str
    ) -> List[Dict[str, Any]]:
        """Get all quotes for a specific article in document order.

        Args:
            article_id: UUID of the article
            version_id: UUID of the result version

        Returns:
            List of dicts with keys: content, source, cue, quote_type, quote_order
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT quote_content AS content, quote_source AS source,
                       cue, quote_type, quote_order
                FROM {schema}.article_quotes
                WHERE article_id = %s AND result_version_id = %s
                ORDER BY quote_order
                """,
                (article_id, version_id)
            )
            return cur.fetchall()

    def get_articles_with_quotes(self, version_id: str) -> List[str]:
        """Return list of article IDs that already have quotes for this version.

        Args:
            version_id: UUID of the result version

        Returns:
            List of article UUID strings
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT article_id::text
                FROM {schema}.article_quotes
                WHERE result_version_id = %s
                """,
                (version_id,)
            )
            return [row["article_id"] for row in cur.fetchall()]
