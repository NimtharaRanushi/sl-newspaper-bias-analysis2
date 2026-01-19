#!/usr/bin/env python3
"""
Helper script to safely re-run the analysis pipeline.

Usage:
    python3 scripts/rerun_pipeline.py                    # Re-run all steps
    python3 scripts/rerun_pipeline.py --topics           # Re-run topics only
    python3 scripts/rerun_pipeline.py --clusters         # Re-run clusters only
    python3 scripts/rerun_pipeline.py --from-topics      # Re-run from topics onwards
    python3 scripts/rerun_pipeline.py --clear-all        # Clear all data and start fresh
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db import get_db, load_config


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} (y/N): ").strip().lower()
    return response == 'y'


def clear_topics_data():
    """Clear all topic-related data."""
    print("\n🗑️  Clearing topic data...")
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor(dict_cursor=False) as cur:
            cur.execute(f"DELETE FROM {schema}.article_analysis WHERE primary_topic_id IS NOT NULL")
            deleted_assignments = cur.rowcount
            cur.execute(f"DELETE FROM {schema}.topics")
            deleted_topics = cur.rowcount
    print(f"   Deleted {deleted_topics} topics and {deleted_assignments} topic assignments")


def clear_clusters_data():
    """Clear all cluster-related data."""
    print("\n🗑️  Clearing cluster data...")
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor(dict_cursor=False) as cur:
            cur.execute(f"DELETE FROM {schema}.article_clusters")
            deleted_mappings = cur.rowcount
            cur.execute(f"DELETE FROM {schema}.event_clusters")
            deleted_clusters = cur.rowcount
    print(f"   Deleted {deleted_clusters} clusters and {deleted_mappings} cluster mappings")


def clear_embeddings_data():
    """Clear all embedding data."""
    print("\n🗑️  Clearing embedding data...")
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor(dict_cursor=False) as cur:
            cur.execute(f"DELETE FROM {schema}.embeddings")
            deleted_embeddings = cur.rowcount
    print(f"   Deleted {deleted_embeddings} embeddings")


def run_step(script_name: str, description: str) -> bool:
    """Run a pipeline step."""
    print(f"\n▶️  {description}...")
    script_path = Path(__file__).parent / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Re-run the analysis pipeline")
    parser.add_argument("--embeddings", action="store_true", help="Re-run embeddings only")
    parser.add_argument("--topics", action="store_true", help="Re-run topics only")
    parser.add_argument("--clusters", action="store_true", help="Re-run clusters only")
    parser.add_argument("--from-embeddings", action="store_true", help="Re-run from embeddings onwards")
    parser.add_argument("--from-topics", action="store_true", help="Re-run from topics onwards")
    parser.add_argument("--clear-all", action="store_true", help="Clear all data and start fresh")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    # Determine what to run
    run_embeddings = args.embeddings or args.from_embeddings or args.clear_all
    run_topics = args.topics or args.from_embeddings or args.from_topics or args.clear_all
    run_clusters = args.clusters or args.from_embeddings or args.from_topics or args.clear_all

    # Default: run all if no specific flags
    if not any([args.embeddings, args.topics, args.clusters,
                args.from_embeddings, args.from_topics, args.clear_all]):
        run_topics = True
        run_clusters = True

    print("=" * 60)
    print("📊 Media Bias Analysis Pipeline Re-run")
    print("=" * 60)

    # Show what will be done
    print("\nPipeline steps to run:")
    if run_embeddings:
        print("  1. Generate embeddings")
    if run_topics:
        print(f"  {'2' if run_embeddings else '1'}. Discover topics")
    if run_clusters:
        step_num = sum([run_embeddings, run_topics]) + 1
        print(f"  {step_num}. Cluster events")

    # Confirm before proceeding
    if not args.no_confirm:
        if not confirm_action("\nProceed with re-run?"):
            print("Aborted.")
            return

    # Clear data if needed
    if args.clear_all:
        if not args.no_confirm:
            if not confirm_action("\n⚠️  This will DELETE all analysis data. Continue?"):
                print("Aborted.")
                return
        clear_clusters_data()
        clear_topics_data()
        clear_embeddings_data()
    else:
        # Clear only what we're regenerating
        if run_clusters:
            clear_clusters_data()
        if run_topics:
            clear_topics_data()
        if run_embeddings:
            clear_embeddings_data()

    # Run pipeline steps
    success = True

    if run_embeddings and success:
        success = run_step("01_generate_embeddings.py", "Generating embeddings")

    if run_topics and success:
        success = run_step("02_discover_topics.py", "Discovering topics")

    if run_clusters and success:
        success = run_step("03_cluster_events.py", "Clustering events")

    # Final status
    print("\n" + "=" * 60)
    if success:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed. Check the output above for errors.")
    print("=" * 60)


if __name__ == "__main__":
    main()
