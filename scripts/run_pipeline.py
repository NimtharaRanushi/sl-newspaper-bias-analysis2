#!/usr/bin/env python3
"""Run the full analysis pipeline for a result version."""

import sys
import argparse
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.versions import (
    create_version,
    get_version,
    get_version_by_name,
    get_default_config,
    find_version_by_config
)


def load_config_file(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_step(script_name: str, version_id: str, extra_args: list = None) -> bool:
    """
    Run a pipeline step script.

    Args:
        script_name: Name of the script to run (e.g., "01_generate_embeddings.py")
        version_id: UUID of the result version
        extra_args: Additional arguments to pass to the script

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name
    cmd = ["python3", str(script_path), "--version-id", version_id]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline for a result version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with auto-generated version name
  python3 scripts/run_pipeline.py

  # Run pipeline for existing version
  python3 scripts/run_pipeline.py --version-id <uuid>

  # Create new version with custom config file and run pipeline
  python3 scripts/run_pipeline.py --config my_config.yaml --name "my-experiment"

  # Create new version with default config
  python3 scripts/run_pipeline.py --name "baseline"

  # Create version with inline description
  python3 scripts/run_pipeline.py --name "no-stopwords" \\
      --description "Topic discovery without location stop words"
        """
    )

    # Version selection
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--version-id",
        help="UUID of existing result version to run pipeline for"
    )
    group.add_argument(
        "--name",
        help="Name for new result version (creates version if doesn't exist, auto-generated if omitted)"
    )

    # Version creation options (only used with --name)
    parser.add_argument(
        "--config",
        help="Path to custom configuration YAML file"
    )
    parser.add_argument(
        "--description",
        default="",
        help="Description of this result version"
    )

    # Pipeline options
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation step"
    )
    parser.add_argument(
        "--skip-topics",
        action="store_true",
        help="Skip topic discovery step"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip event clustering step"
    )

    args = parser.parse_args()

    # Auto-generate name if neither version-id nor name provided
    if not args.version_id and not args.name:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        args.name = f"v{timestamp}"
        print(f"Auto-generated version name: {args.name}")

    # Determine version ID
    if args.version_id:
        version_id = args.version_id
        version = get_version(version_id)
        if not version:
            print(f"Error: Version {version_id} not found")
            sys.exit(1)
        print(f"Using existing version: {version['name']} ({version_id})")

    else:  # --name provided (or auto-generated)
        # Check if version with this name already exists
        existing = get_version_by_name(args.name)
        if existing:
            version_id = existing['id']
            print(f"Found existing version '{args.name}' ({version_id})")
            print("Using this version for pipeline run.")

        else:
            # Load or create configuration
            if args.config:
                config = load_config_file(args.config)
                print(f"Loaded configuration from {args.config}")
            else:
                config = get_default_config()
                print("Using default configuration")

            # Check if a version with this config already exists
            existing_config = find_version_by_config(config)
            if existing_config:
                print(f"\nWarning: A version with this configuration already exists:")
                print(f"  Name: {existing_config['name']}")
                print(f"  ID: {existing_config['id']}")
                print(f"\nWould you like to:")
                print(f"  1) Use the existing version")
                print(f"  2) Create a new version with the same config but different name")
                print(f"  3) Abort")

                choice = input("\nEnter choice (1/2/3): ").strip()

                if choice == "1":
                    version_id = existing_config['id']
                    print(f"\nUsing existing version: {existing_config['name']}")
                elif choice == "2":
                    print(f"\nCreating new version '{args.name}' with duplicate config...")
                    version_id = create_version(args.name, args.description, config)
                    print(f"Created version: {version_id}")
                else:
                    print("Aborted.")
                    sys.exit(0)
            else:
                # Create new version
                print(f"\nCreating new version '{args.name}'...")
                version_id = create_version(args.name, args.description, config)
                print(f"Created version: {version_id}")

    # Show configuration
    version = get_version(version_id)
    print(f"\nVersion Configuration:")
    print(f"  Name: {version['name']}")
    print(f"  Description: {version['description'] or '(none)'}")
    print(f"  Created: {version['created_at']}")
    print(f"  Random seed: {version['configuration'].get('random_seed', 42)}")

    print(f"\nPipeline Status:")
    status = version['pipeline_status']
    print(f"  Embeddings: {'✓' if status.get('embeddings') else '○'}")
    print(f"  Topics: {'✓' if status.get('topics') else '○'}")
    print(f"  Clustering: {'✓' if status.get('clustering') else '○'}")

    # Confirm before running
    print(f"\n{'='*60}")
    proceed = input("Proceed with pipeline? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("Aborted.")
        sys.exit(0)

    # Run pipeline steps
    success = True

    if not args.skip_embeddings:
        if not run_step("01_generate_embeddings.py", version_id):
            print("\n❌ Embedding generation failed")
            success = False
            sys.exit(1)

    if not args.skip_topics and success:
        if not run_step("02_discover_topics.py", version_id):
            print("\n❌ Topic discovery failed")
            success = False
            sys.exit(1)

    if not args.skip_clustering and success:
        if not run_step("03_cluster_events.py", version_id):
            print("\n❌ Event clustering failed")
            success = False
            sys.exit(1)

    # Final status
    print(f"\n{'='*60}")
    if success:
        version = get_version(version_id)
        if version['is_complete']:
            print("✅ Pipeline completed successfully!")
        else:
            print("⚠️  Pipeline run complete, but some steps were skipped")
    else:
        print("❌ Pipeline failed")
        sys.exit(1)

    print(f"\nVersion: {version['name']} ({version_id})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
