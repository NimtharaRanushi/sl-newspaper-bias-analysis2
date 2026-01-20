"""Version management for result configurations."""

import json
from typing import Dict, List, Optional, Any
from src.db import Database, load_config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration from config.yaml.

    Returns:
        Dictionary with default configuration for embeddings, topics, and clustering.
    """
    config = load_config()

    return {
        "random_seed": 42,
        "embeddings": {
            "provider": config["embeddings"]["provider"],
            "model": config["embeddings"]["model"],
            "dimensions": config["embeddings"]["dimensions"]
        },
        "topics": {
            "min_topic_size": config["topics"]["min_topic_size"],
            "diversity": config["topics"].get("diversity", 0.5),
            "nr_topics": None,
            "stop_words": ["sri", "lanka", "lankan"],
            "embedding_model": config["embeddings"]["model"],
            "random_seed": 42,
            "umap": {
                "n_neighbors": 15,
                "n_components": 5,
                "min_dist": 0.0,
                "metric": "cosine",
                "random_state": 42
            },
            "hdbscan": {
                "min_cluster_size": config["topics"]["min_topic_size"],
                "metric": "euclidean",
                "cluster_selection_method": "eom"
            },
            "vectorizer": {
                "ngram_range": [1, 3],
                "min_df": 5
            }
        },
        "clustering": {
            "similarity_threshold": config["clustering"]["similarity_threshold"],
            "time_window_days": config["clustering"]["time_window_days"],
            "min_cluster_size": config["clustering"]["min_cluster_size"]
        }
    }


def create_version(
    name: str,
    description: str = "",
    configuration: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a new result version.

    Args:
        name: Unique name for this version
        description: Optional description of this version
        configuration: Configuration dictionary (uses default if not provided)

    Returns:
        UUID of the created version

    Raises:
        ValueError: If version name already exists
    """
    if configuration is None:
        configuration = get_default_config()

    with Database() as db:
        schema = db.config["schema"]

        # Check if name already exists
        with db.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {schema}.result_versions WHERE name = %s",
                (name,)
            )
            if cur.fetchone():
                raise ValueError(f"Version with name '{name}' already exists")

        # Insert new version
        with db.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.result_versions
                (name, description, configuration)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (name, description, json.dumps(configuration))
            )
            result = cur.fetchone()
            return str(result["id"])


def get_version(version_id: str) -> Optional[Dict[str, Any]]:
    """
    Get version metadata by ID.

    Args:
        version_id: UUID of the version

    Returns:
        Dictionary with version metadata or None if not found
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, description, configuration, is_complete,
                       pipeline_status, created_at, updated_at
                FROM {schema}.result_versions
                WHERE id = %s
                """,
                (version_id,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None


def get_version_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get version metadata by name.

    Args:
        name: Name of the version

    Returns:
        Dictionary with version metadata or None if not found
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, description, configuration, is_complete,
                       pipeline_status, created_at, updated_at
                FROM {schema}.result_versions
                WHERE name = %s
                """,
                (name,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None


def list_versions() -> List[Dict[str, Any]]:
    """
    List all versions.

    Returns:
        List of dictionaries with version metadata
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, description, configuration, is_complete,
                       pipeline_status, created_at, updated_at
                FROM {schema}.result_versions
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]


def find_version_by_config(configuration: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a version with matching configuration.

    Args:
        configuration: Configuration dictionary to match

    Returns:
        Dictionary with version metadata or None if no match found
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, description, configuration, is_complete,
                       pipeline_status, created_at, updated_at
                FROM {schema}.result_versions
                WHERE configuration = %s::jsonb
                """,
                (json.dumps(configuration),)
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None


def update_pipeline_status(
    version_id: str,
    step: str,
    complete: bool
) -> None:
    """
    Update pipeline completion status for a specific step.

    Args:
        version_id: UUID of the version
        step: Pipeline step name ('embeddings', 'topics', or 'clustering')
        complete: Whether the step is complete
    """
    valid_steps = ['embeddings', 'topics', 'clustering']
    if step not in valid_steps:
        raise ValueError(f"Invalid step: {step}. Must be one of {valid_steps}")

    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Update the specific step status
            cur.execute(
                f"""
                UPDATE {schema}.result_versions
                SET pipeline_status = jsonb_set(
                    pipeline_status,
                    %s,
                    %s
                ),
                updated_at = NOW()
                WHERE id = %s
                """,
                (f'{{{step}}}', json.dumps(complete), version_id)
            )

            # Check if all steps are complete and update is_complete
            cur.execute(
                f"""
                UPDATE {schema}.result_versions
                SET is_complete = (
                    (pipeline_status->>'embeddings')::boolean AND
                    (pipeline_status->>'topics')::boolean AND
                    (pipeline_status->>'clustering')::boolean
                )
                WHERE id = %s
                """,
                (version_id,)
            )


def get_version_config(version_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific version.

    Args:
        version_id: UUID of the version

    Returns:
        Configuration dictionary or None if version not found
    """
    version = get_version(version_id)
    return version["configuration"] if version else None


def delete_version(version_id: str) -> bool:
    """
    Delete a version and all its associated results.

    Args:
        version_id: UUID of the version to delete

    Returns:
        True if deleted, False if version not found

    Note:
        This will cascade delete all associated embeddings, topics,
        article_analysis, event_clusters, and article_clusters.
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"DELETE FROM {schema}.result_versions WHERE id = %s",
                (version_id,)
            )
            return cur.rowcount > 0


def get_version_statistics(version_id: str) -> Dict[str, int]:
    """
    Get statistics for a version (counts of embeddings, topics, clusters, etc.).

    Args:
        version_id: UUID of the version

    Returns:
        Dictionary with counts for various entities
    """
    with Database() as db:
        schema = db.config["schema"]
        stats = {}

        # Count embeddings
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.embeddings WHERE result_version_id = %s",
                (version_id,)
            )
            stats["embeddings"] = cur.fetchone()["count"]

        # Count topics
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.topics WHERE result_version_id = %s",
                (version_id,)
            )
            stats["topics"] = cur.fetchone()["count"]

        # Count article analyses
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.article_analysis WHERE result_version_id = %s",
                (version_id,)
            )
            stats["article_analysis"] = cur.fetchone()["count"]

        # Count event clusters
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE result_version_id = %s",
                (version_id,)
            )
            stats["event_clusters"] = cur.fetchone()["count"]

        # Count article-cluster mappings
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.article_clusters WHERE result_version_id = %s",
                (version_id,)
            )
            stats["article_clusters"] = cur.fetchone()["count"]

        return stats
