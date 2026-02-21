"""Article database operations."""

from typing import List, Dict


class ArticleMixin:
    """Article-related database operations."""

    def get_articles(
        self,
        limit: int = None,
        offset: int = 0,
        source_id: str = None
    ) -> List[Dict]:
        """Fetch articles from news_articles table."""
        schema = self.config["schema"]
        query = f"""
            SELECT id, url, title, content, date_posted, source_id, lang
            FROM {schema}.news_articles
            WHERE content IS NOT NULL AND content != '' AND is_ditwah_cyclone = 1
              AND date_posted >= '2025-11-22' AND date_posted <= '2025-12-31'
        """
        params = []

        if source_id:
            query += " AND source_id = %s"
            params.append(source_id)

        query += " ORDER BY date_posted, id"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_article_count(self) -> int:
        """Get total article count."""
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.news_articles
                WHERE content IS NOT NULL AND content != '' AND is_ditwah_cyclone = 1
                  AND date_posted >= '2025-11-22' AND date_posted <= '2025-12-31'
            """)
            return cur.fetchone()["count"]

    def get_article_by_url(self, url: str) -> Dict:
        """Fetch article by URL.

        Args:
            url: The article URL to search for

        Returns:
            Article dict with id, url, title, content, source_id, date_posted, or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, url, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE url = %s AND is_ditwah_cyclone = 1
                  AND date_posted >= '2025-11-22' AND date_posted <= '2025-12-31'
            """, (url,))
            return cur.fetchone()

    def get_articles_without_embeddings(self, embedding_model: str, limit: int = None) -> List[Dict]:
        """Get articles that don't have embeddings yet for a specific model.

        Args:
            embedding_model: Name of the embedding model (e.g., 'all-mpnet-base-v2')
            limit: Maximum number of articles to return
        """
        schema = self.config["schema"]

        query = f"""
            SELECT a.id, a.title, a.content, a.date_posted, a.source_id
            FROM {schema}.news_articles a
            LEFT JOIN {schema}.embeddings e ON a.id = e.article_id AND e.embedding_model = %s
            WHERE e.id IS NULL
              AND a.content IS NOT NULL
              AND a.content != ''
              AND a.is_ditwah_cyclone = 1
              AND a.date_posted >= '2025-11-22' AND a.date_posted <= '2025-12-31'
            ORDER BY a.date_posted, a.id
        """
        params = [embedding_model]

        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_article_by_id(self, article_id: int) -> Dict:
        """Fetch article metadata by ID.

        Args:
            article_id: The article ID

        Returns:
            Article dict with metadata or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id, date_posted, url, lang, is_ditwah_cyclone
                FROM {schema}.news_articles
                WHERE id = %s
            """, (article_id,))
            return cur.fetchone()

    def search_articles(self, title_search: str, limit: int = 50) -> List[Dict]:
        """Search articles by title.

        Args:
            title_search: Search term for title (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of article dicts with id, title, source_id, date_posted
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, source_id, date_posted
                FROM {schema}.news_articles
                WHERE title ILIKE %s AND is_ditwah_cyclone = 1
                  AND date_posted >= '2025-11-22' AND date_posted <= '2025-12-31'
                ORDER BY date_posted DESC
                LIMIT %s
            """, (f"%{title_search}%", limit))
            return cur.fetchall()
