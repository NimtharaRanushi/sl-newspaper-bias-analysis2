"""Named entity recognition database operations."""

from typing import List, Dict, Any


class NERMixin:
    """Named entity recognition database operations."""

    def store_named_entities(
        self,
        entities: List[Dict[str, Any]],
        result_version_id: str
    ) -> None:
        """
        Store named entities in the database.

        Args:
            entities: List of entity dictionaries
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            for entity in entities:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.named_entities
                    (result_version_id, article_id, entity_text, entity_type,
                     start_char, end_char, confidence, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, result_version_id, entity_text, entity_type, start_char)
                    DO NOTHING
                    """,
                    (
                        result_version_id,
                        entity["article_id"],
                        entity["entity_text"],
                        entity["entity_type"],
                        entity["start_char"],
                        entity["end_char"],
                        entity["confidence"],
                        entity.get("context", "")
                    )
                )

    def compute_entity_statistics(self, result_version_id: str) -> None:
        """
        Compute aggregated entity statistics per source.

        Args:
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.entity_statistics
                (result_version_id, entity_text, entity_type, source_id,
                 mention_count, article_count, avg_confidence)
                SELECT
                    ne.result_version_id,
                    ne.entity_text,
                    ne.entity_type,
                    na.source_id,
                    COUNT(*) as mention_count,
                    COUNT(DISTINCT ne.article_id) as article_count,
                    AVG(ne.confidence) as avg_confidence
                FROM {schema}.named_entities ne
                JOIN {schema}.news_articles na ON ne.article_id = na.id
                WHERE ne.result_version_id = %s
                  AND na.is_ditwah_cyclone = 1
                  AND na.date_posted >= '2025-11-22' AND na.date_posted <= '2025-12-31'
                GROUP BY ne.result_version_id, ne.entity_text, ne.entity_type, na.source_id
                ON CONFLICT (result_version_id, entity_text, entity_type, source_id)
                DO UPDATE SET
                    mention_count = EXCLUDED.mention_count,
                    article_count = EXCLUDED.article_count,
                    avg_confidence = EXCLUDED.avg_confidence
                """,
                (result_version_id,)
            )

    def get_entity_statistics(
        self,
        result_version_id: str,
        entity_type: str = None,
        source_id: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get entity statistics for a version.

        Args:
            result_version_id: UUID of the result version
            entity_type: Optional filter by entity type
            source_id: Optional filter by source
            limit: Maximum number of results

        Returns:
            List of entity statistics
        """
        schema = self.config["schema"]

        query = f"""
            SELECT entity_text, entity_type, source_id,
                   mention_count, article_count, avg_confidence
            FROM {schema}.entity_statistics
            WHERE result_version_id = %s
        """
        params = [result_version_id]

        if entity_type:
            query += " AND entity_type = %s"
            params.append(entity_type)

        if source_id:
            query += " AND source_id = %s"
            params.append(source_id)

        query += " ORDER BY mention_count DESC LIMIT %s"
        params.append(limit)

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_entities_for_article(
        self,
        article_id: str,
        result_version_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all named entities for a specific article.

        Args:
            article_id: The article ID
            result_version_id: UUID of the result version

        Returns:
            List of entity dicts with entity_text, entity_type, start_char, end_char, confidence
            Ordered by start_char for sequential processing
        """
        schema = self.config["schema"]

        query = f"""
            SELECT entity_text, entity_type, start_char, end_char, confidence
            FROM {schema}.named_entities
            WHERE article_id = %s AND result_version_id = %s
            ORDER BY start_char
        """

        with self.cursor() as cur:
            cur.execute(query, (article_id, result_version_id))
            return cur.fetchall()

    def get_unique_entity_texts(
        self,
        result_version_id: str = None,
        entity_types: List[str] = None,
        normalize: bool = True
    ) -> List[str]:
        """
        Get unique entity texts from NER analysis for use as stop words.

        Args:
            result_version_id: Optional NER version ID. If None, uses any completed NER version.
            entity_types: Optional list of entity types to filter (e.g., ['PERSON', 'ORG', 'GPE'])
            normalize: If True, lowercase and deduplicate entities

        Returns:
            List of unique entity text strings
        """
        schema = self.config["schema"]

        # If no version specified, find first completed NER version
        if result_version_id is None:
            with self.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id FROM {schema}.result_versions
                    WHERE analysis_type = 'ner'
                      AND (pipeline_status->>'ner')::boolean = true
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("No completed NER analysis found. Please run NER pipeline first.")
                result_version_id = str(row["id"])

        # Build query to get unique entity texts
        query = f"""
            SELECT DISTINCT entity_text
            FROM {schema}.named_entities
            WHERE result_version_id = %s
        """
        params = [result_version_id]

        # Filter by entity types if provided
        if entity_types:
            query += " AND entity_type = ANY(%s)"
            params.append(entity_types)

        with self.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        # Extract entity texts
        entity_texts = [row["entity_text"] for row in rows]

        # Normalize if requested
        if normalize:
            # Lowercase and deduplicate
            entity_texts = list(set(text.lower() for text in entity_texts))

        return sorted(entity_texts)
