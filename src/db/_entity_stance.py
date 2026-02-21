"""Entity stance detection database operations."""

from typing import List, Dict, Any, Optional


class EntityStanceMixin:
    """Entity stance detection database operations."""

    def store_entity_stances(
        self,
        stances: List[Dict[str, Any]],
        version_id: str,
        ner_version_id: str
    ) -> None:
        """Store entity stance results in batch.

        Args:
            stances: List of stance dicts
            version_id: Entity stance version UUID
            ner_version_id: NER version UUID used for entity source
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            for s in stances:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.entity_stance
                    (result_version_id, ner_version_id, article_id, chunk_index,
                     start_char, end_char, entity_text, entity_type,
                     stance_score, stance_label, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, result_version_id, chunk_index, entity_text)
                    DO UPDATE SET
                        stance_score = EXCLUDED.stance_score,
                        stance_label = EXCLUDED.stance_label,
                        confidence = EXCLUDED.confidence
                    """,
                    (
                        version_id, ner_version_id, s["article_id"],
                        s["chunk_index"], s["start_char"], s["end_char"],
                        s["entity_text"], s["entity_type"],
                        s["stance_score"], s["stance_label"], s["confidence"]
                    )
                )

    def get_entity_stance_summary(self, version_id: str) -> List[Dict]:
        """Get aggregated stance per entity x source.

        Args:
            version_id: Entity stance version UUID

        Returns:
            List of dicts with entity_text, entity_type, source_id,
            avg_stance, stance_count, avg_confidence
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    es.entity_text,
                    es.entity_type,
                    na.source_id,
                    AVG(es.stance_score) as avg_stance,
                    COUNT(*) as stance_count,
                    AVG(es.confidence) as avg_confidence
                FROM {schema}.entity_stance es
                JOIN {schema}.news_articles na ON es.article_id = na.id
                WHERE es.result_version_id = %s
                GROUP BY es.entity_text, es.entity_type, na.source_id
                ORDER BY COUNT(*) DESC
                """,
                (version_id,)
            )
            return cur.fetchall()

    def get_entity_stance_summary_by_topic(
        self,
        stance_version_id: str,
        topic_version_id: str,
        topic_bertopic_id: Optional[int] = None,
    ) -> List[Dict]:
        """Get aggregated stance per entity x source, optionally filtered by topic.

        Joins entity_stance -> article_analysis -> topics to filter by topic.

        Args:
            stance_version_id: Entity stance result_version UUID
            topic_version_id: Topics result_version UUID (for article_analysis join)
            topic_bertopic_id: BERTopic topic_id to filter by, or None for all topics

        Returns:
            List of dicts: entity_text, entity_type, source_id,
            avg_stance, stance_count, avg_confidence
        """
        schema = self.config["schema"]
        params = [stance_version_id, topic_version_id, topic_version_id]
        topic_filter = ""
        if topic_bertopic_id is not None:
            topic_filter = "AND t.topic_id = %s"
            params.append(topic_bertopic_id)

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    es.entity_text,
                    es.entity_type,
                    na.source_id,
                    AVG(es.stance_score) as avg_stance,
                    COUNT(*) as stance_count,
                    AVG(es.confidence) as avg_confidence
                FROM {schema}.entity_stance es
                JOIN {schema}.news_articles na ON es.article_id = na.id
                JOIN {schema}.article_analysis aa ON es.article_id = aa.article_id
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE es.result_version_id = %s
                  AND aa.result_version_id = %s
                  AND t.result_version_id = %s
                  {topic_filter}
                GROUP BY es.entity_text, es.entity_type, na.source_id
                ORDER BY COUNT(*) DESC
                """,
                params,
            )
            return cur.fetchall()

    def get_entity_stance_examples(
        self,
        version_id: str,
        entity_texts: List[str],
        limit: int = 200,
        topic_version_id: str = None,
        topic_bertopic_id: int = None,
    ) -> List[Dict]:
        """Fetch chunk-level stance rows with extracted chunk text for given entities.

        Returns all rows for the given entities, ordered by |stance_score| desc.
        Caller should group by (source_id, article_id, chunk_index) and pick top N chunks.

        Args:
            version_id: Entity stance result_version UUID
            entity_texts: List of entity strings to filter by
            limit: Max rows to return (default 200, enough for all sources)
            topic_version_id: Optional topics result_version UUID to filter by topic
            topic_bertopic_id: Optional BERTopic topic_id to filter by specific topic

        Returns:
            List of dicts: article_id, title, source_id, chunk_index,
            entity_text, entity_type, stance_score, stance_label,
            confidence, chunk_text
        """
        schema = self.config["schema"]
        topic_join = ""
        topic_filter = ""
        params: list = [version_id, entity_texts]
        if topic_version_id is not None:
            topic_join = (
                f"JOIN {schema}.article_analysis aa ON es.article_id = aa.article_id "
                f"JOIN {schema}.topics t ON aa.primary_topic_id = t.id"
            )
            topic_filter = "AND aa.result_version_id = %s AND t.result_version_id = %s"
            params += [topic_version_id, topic_version_id]
            if topic_bertopic_id is not None:
                topic_filter += " AND t.topic_id = %s"
                params.append(topic_bertopic_id)
        params.append(limit)
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    na.id as article_id,
                    na.title,
                    na.source_id,
                    es.chunk_index,
                    es.entity_text,
                    es.entity_type,
                    es.stance_score,
                    es.stance_label,
                    es.confidence,
                    SUBSTRING(na.content FROM es.start_char + 1
                              FOR es.end_char - es.start_char) AS chunk_text
                FROM {schema}.entity_stance es
                JOIN {schema}.news_articles na ON es.article_id = na.id
                {topic_join}
                WHERE es.result_version_id = %s
                  AND es.entity_text = ANY(%s)
                  {topic_filter}
                ORDER BY ABS(es.stance_score) DESC
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def get_entity_stance_for_article(
        self, article_id: str, version_id: str
    ) -> List[Dict]:
        """Get chunk-level stance detail for an article.

        Args:
            article_id: Article UUID
            version_id: Entity stance version UUID

        Returns:
            List of stance dicts ordered by chunk_index
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_index, start_char, end_char,
                       entity_text, entity_type,
                       stance_score, stance_label, confidence
                FROM {schema}.entity_stance
                WHERE article_id = %s AND result_version_id = %s
                ORDER BY chunk_index, entity_text
                """,
                (article_id, version_id)
            )
            return cur.fetchall()

    def get_most_polarizing_entities(
        self, version_id: str, limit: int = 20
    ) -> List[Dict]:
        """Get entities with highest cross-source stance variance.

        Args:
            version_id: Entity stance version UUID
            limit: Max entities to return

        Returns:
            List of dicts with entity_text, entity_type, stance_variance,
            avg_stance, source_count, mention_count
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                WITH per_source AS (
                    SELECT
                        es.entity_text,
                        es.entity_type,
                        na.source_id,
                        AVG(es.stance_score) as avg_stance
                    FROM {schema}.entity_stance es
                    JOIN {schema}.news_articles na ON es.article_id = na.id
                    WHERE es.result_version_id = %s
                    GROUP BY es.entity_text, es.entity_type, na.source_id
                )
                SELECT
                    entity_text,
                    entity_type,
                    VARIANCE(avg_stance) as stance_variance,
                    AVG(avg_stance) as avg_stance,
                    COUNT(DISTINCT source_id) as source_count,
                    SUM(1) as mention_count
                FROM per_source
                GROUP BY entity_text, entity_type
                HAVING COUNT(DISTINCT source_id) >= 2
                ORDER BY VARIANCE(avg_stance) DESC NULLS LAST
                LIMIT %s
                """,
                (version_id, limit)
            )
            return cur.fetchall()

    def get_articles_without_entity_stance(
        self, version_id: str, limit: int = None
    ) -> List[Dict]:
        """Get articles that haven't been processed for entity stance.

        Args:
            version_id: Entity stance version UUID
            limit: Max articles to return

        Returns:
            List of article dicts
        """
        schema = self.config["schema"]
        query = f"""
            SELECT na.id, na.title, na.content, na.source_id, na.date_posted
            FROM {schema}.news_articles na
            WHERE na.content IS NOT NULL AND na.content != ''
              AND na.is_ditwah_cyclone = 1
              AND na.date_posted >= '2025-11-22' AND na.date_posted <= '2025-12-31'
              AND na.id NOT IN (
                  SELECT DISTINCT article_id
                  FROM {schema}.entity_stance
                  WHERE result_version_id = %s
              )
            ORDER BY na.date_posted, na.id
        """
        params = [version_id]

        if limit:
            query += " LIMIT %s"
            params.append(limit)

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
