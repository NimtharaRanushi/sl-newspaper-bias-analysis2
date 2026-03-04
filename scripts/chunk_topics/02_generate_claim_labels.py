#!/usr/bin/env python3
"""Generate LLM claim labels for existing chunk topics.

Use this to add claim labels to an already-completed chunk topic version
without re-running the full discovery pipeline.
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db import get_db
from src.llm import get_llm
from src.prompts import load_prompt
from src.versions import get_version


def main():
    parser = argparse.ArgumentParser(description="Generate LLM claim labels for chunk topics")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the chunk_topics result version"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate labels even if description already exists"
    )
    args = parser.parse_args()

    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)
    if version["analysis_type"] != "chunk_topics":
        print(f"Error: Version is type '{version['analysis_type']}', expected 'chunk_topics'")
        sys.exit(1)

    print("=" * 60)
    print("Generate Claim Labels for Chunk Topics")
    print("=" * 60)
    print(f"Version: {version['name']} ({args.version_id})")
    print(f"Force: {args.force}")
    print()

    llm = get_llm()
    system_prompt = load_prompt("chunk_topics/claim_label_system.md")

    with get_db() as db:
        topics = db.get_chunk_topics_with_counts(args.version_id, min_count=1)
        print(f"Found {len(topics)} non-outlier topics")

        success_count = 0
        skip_count = 0

        for topic in topics:
            db_id = topic["id"]
            tid = topic["topic_id"]

            # Skip if already has a claim label (unless --force)
            if not args.force and topic.get("description"):
                try:
                    existing = json.loads(topic["description"]) if isinstance(topic["description"], str) else topic["description"]
                    if existing.get("claim"):
                        skip_count += 1
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

            # Get top 25 chunks by confidence
            chunks_for_topic = db.get_chunks_by_topic(db_id, args.version_id, limit=25)
            if not chunks_for_topic:
                print(f"  [T{tid}] No chunks found, skipping")
                continue

            # Build keywords section
            keywords = topic.get("keywords") or []
            kw_section = "\n".join(f"  - {kw}" for kw in keywords[:20])

            # Build chunks section with full text and metadata
            chunk_lines = []
            for i, chunk in enumerate(chunks_for_topic, 1):
                source = chunk.get("source_id", "unknown")
                date = chunk["date_posted"].strftime("%Y-%m-%d") if chunk.get("date_posted") else "unknown"
                text = chunk["chunk_text"].replace("\n", " ").strip()
                chunk_lines.append(f"  {i}. [{source}, {date}]\n     {text}")
            chunks_section = "\n".join(chunk_lines)

            prompt = load_prompt(
                "chunk_topics/claim_label_user.md",
                topic_id=tid,
                chunk_count=topic["chunk_count"],
                keyword_count=min(20, len(keywords)),
                keywords_section=kw_section,
                chunks_shown=len(chunks_for_topic),
                chunks_section=chunks_section,
            )

            try:
                response = llm.generate(prompt, system_prompt=system_prompt, json_mode=True)
                result = json.loads(response.content)
                db.update_chunk_topic_description(db_id, json.dumps(result))
                print(f"  [T{tid}] {result.get('claim', '?')}")
                success_count += 1
            except Exception as e:
                fallback = {
                    "claim": " ".join(keywords[:3]) if keywords else topic["name"],
                    "description": f"Topic covering: {', '.join(keywords[:5])}"
                }
                db.update_chunk_topic_description(db_id, json.dumps(fallback))
                print(f"  [T{tid}] (fallback) {fallback['claim']} — {e}")
                success_count += 1

    print()
    print(f"Labelled: {success_count}")
    if skip_count:
        print(f"Skipped (already labelled): {skip_count}")
    print("Done.")


if __name__ == "__main__":
    main()
