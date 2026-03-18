#!/usr/bin/env python3
"""Extract quotes from articles using LLM structured output."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.quotes import extract_quotes_from_articles
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Extract quotes from articles")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the quote_extraction result version"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of articles to process per batch (default: 50)"
    )
    args = parser.parse_args()

    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "quote_extraction":
        print(f"Error: Version {args.version_id} is not a quote_extraction version (type: {version['analysis_type']})")
        sys.exit(1)

    version_config = get_version_config(args.version_id)
    qe_config = version_config.get("quote_extraction", {})

    print("=" * 60)
    print("Quote Extraction Pipeline")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print(f"LLM Provider: {qe_config.get('llm_provider', 'openai')}")
    print(f"LLM Model: {qe_config.get('llm_model', 'gpt-4o-mini')}")
    print()

    counts = extract_quotes_from_articles(
        version_id=args.version_id,
        config=version_config,
        batch_size=args.batch_size
    )

    print()
    print("=" * 60)
    print("Results:")
    print(f"  Successful: {counts['successful']}")
    print(f"  Failed:     {counts['failed']}")
    print(f"  Skipped:    {counts['skipped']} (already processed)")
    print("=" * 60)

    update_pipeline_status(args.version_id, "quote_extraction", True)
    print(f"\n✓ Quote extraction step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
