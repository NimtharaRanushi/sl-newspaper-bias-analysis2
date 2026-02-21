"""Event clustering database operations."""

from typing import List, Dict
from psycopg2.extras import execute_values


class ClusteringMixin:
    """Event clustering database operations."""

    def store_event_clusters(self, clusters: List[Dict], result_version_id: str):
        """Store event clusters for a specific version."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            for cluster in clusters:
                cur.execute(f"""
                    INSERT INTO {schema}.event_clusters
                    (id, result_version_id, cluster_name, cluster_description, representative_article_id,
                     article_count, sources_count, date_start, date_end, centroid_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        result_version_id = EXCLUDED.result_version_id,
                        cluster_name = EXCLUDED.cluster_name,
                        cluster_description = EXCLUDED.cluster_description,
                        representative_article_id = EXCLUDED.representative_article_id,
                        article_count = EXCLUDED.article_count,
                        sources_count = EXCLUDED.sources_count,
                        date_start = EXCLUDED.date_start,
                        date_end = EXCLUDED.date_end,
                        centroid_embedding = EXCLUDED.centroid_embedding
                """, (
                    cluster["id"],
                    result_version_id,
                    cluster["name"],
                    cluster.get("description"),
                    cluster["representative_article_id"],
                    cluster["article_count"],
                    cluster["sources_count"],
                    cluster["date_start"],
                    cluster["date_end"],
                    cluster.get("centroid")
                ))

                # Store article-cluster mappings
                if cluster.get("articles"):
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {schema}.article_clusters
                        (article_id, cluster_id, result_version_id, similarity_score)
                        VALUES %s
                        ON CONFLICT (article_id, cluster_id, result_version_id) DO UPDATE SET
                            similarity_score = EXCLUDED.similarity_score
                        """,
                        [(a["article_id"], cluster["id"], result_version_id, a.get("similarity", 0.0))
                         for a in cluster["articles"]]
                    )

    def get_cluster_for_article(self, article_id: int, version_id: str) -> Dict:
        """Fetch event cluster for article.

        Args:
            article_id: The article ID
            version_id: The clustering version ID

        Returns:
            Cluster dict with details or None if not in any cluster
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id as cluster_id, ec.cluster_name, ac.similarity_score,
                       ec.sources_count, ec.article_count, ec.date_start, ec.date_end,
                       ARRAY_AGG(DISTINCT n.source_id) as other_sources
                FROM {schema}.article_clusters ac
                JOIN {schema}.event_clusters ec ON ac.cluster_id = ec.id
                LEFT JOIN {schema}.article_clusters ac2 ON ec.id = ac2.cluster_id AND ac2.article_id != %s
                LEFT JOIN {schema}.news_articles n ON ac2.article_id = n.id
                WHERE ac.article_id = %s AND ac.result_version_id = %s
                  AND ec.result_version_id = %s
                  AND (n.id IS NULL OR (n.is_ditwah_cyclone = 1 AND n.date_posted >= '2025-11-22' AND n.date_posted <= '2025-12-31'))
                GROUP BY ec.id, ec.cluster_name, ac.similarity_score,
                         ec.sources_count, ec.article_count, ec.date_start, ec.date_end
            """, (article_id, article_id, version_id, version_id))
            return cur.fetchone()

    def get_articles_by_cluster(self, cluster_id: str, version_id: str) -> List[Dict]:
        """Fetch all articles in an event cluster.

        Args:
            cluster_id: The cluster UUID
            version_id: The result version ID

        Returns:
            List of article dicts with id, title, content, source_id, date_posted, similarity_score
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT a.id, a.title, a.content, a.source_id, a.date_posted,
                       ac.similarity_score
                FROM {schema}.news_articles a
                JOIN {schema}.article_clusters ac ON a.id = ac.article_id
                WHERE ac.cluster_id = %s
                  AND ac.result_version_id = %s
                  AND a.is_ditwah_cyclone = 1
                  AND a.date_posted >= '2025-11-22' AND a.date_posted <= '2025-12-31'
                ORDER BY ac.similarity_score DESC
            """, (cluster_id, version_id))
            return cur.fetchall()

    def get_all_clusters_with_counts(
        self,
        version_id: str,
        min_article_count: int = 3
    ) -> List[Dict]:
        """Get all event clusters with article counts above threshold.

        Args:
            version_id: The result version ID
            min_article_count: Minimum articles per cluster (default: 3)

        Returns:
            List of cluster dicts with id, cluster_name, article_count, sources_count, date_start, date_end
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id, ec.cluster_name, ec.article_count, ec.sources_count,
                       ec.date_start, ec.date_end
                FROM {schema}.event_clusters ec
                WHERE ec.result_version_id = %s
                  AND ec.article_count >= %s
                ORDER BY ec.article_count DESC
            """, (version_id, min_article_count))
            return cur.fetchall()
