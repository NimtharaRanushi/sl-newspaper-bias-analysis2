#!/usr/bin/env python3
"""Discover chunk-level topics using BERTopic.

Chunks articles into 5-sentence windows, embeds them, fits a global BERTopic
model, and stores results for dashboard browsing.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Set environment variables for single-threaded execution (reproducibility)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import spacy

from src.db import get_db
from src.db._articles import ditwah_filters
from src.llm import get_embeddings_client
from src.config import load_config
from src.topics import TopicModeler, label_topics_from_keywords
from src.versions import (
    get_version, get_version_config, update_pipeline_status,
    save_model_to_version
)


def chunk_articles(articles, chunk_size=5, min_chunk_sentences=2):
    """Chunk articles into sentence windows.

    Args:
        articles: List of article dicts with id, content
        chunk_size: Number of sentences per chunk
        min_chunk_sentences: Minimum sentences for final chunk (otherwise merge)

    Returns:
        List of chunk dicts
    """
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])

    chunk_records = []

    for article in articles:
        content = article["content"]
        if not content or not content.strip():
            continue

        doc = nlp(content)
        sentences = list(doc.sents)
        if not sentences:
            continue

        article_chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk_sents = sentences[i:i + chunk_size]

            # Merge final chunk into previous if too short
            if len(chunk_sents) < min_chunk_sentences and article_chunks:
                prev = article_chunks[-1]
                prev["end_char"] = chunk_sents[-1].end_char
                continue

            article_chunks.append({
                "article_id": str(article["id"]),
                "chunk_index": len(article_chunks),
                "start_char": chunk_sents[0].start_char,
                "end_char": chunk_sents[-1].end_char,
            })

        chunk_records.extend(article_chunks)

    return chunk_records


def main():
    parser = argparse.ArgumentParser(description="Discover chunk-level topics using BERTopic")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the chunk_topics result version"
    )
    parser.add_argument(
        "--skip-claims",
        action="store_true",
        help="Skip LLM claim label generation"
    )
    args = parser.parse_args()

    # Validate version
    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "chunk_topics":
        print(f"Error: Version {args.version_id} is not a chunk_topics version (type: {version['analysis_type']})")
        sys.exit(1)

    version_config = get_version_config(args.version_id)

    print("=" * 60)
    print("Chunk-Level Topic Discovery")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print()

    # Extract config sections
    chunking_config = version_config.get("chunking", {})
    topic_config = version_config.get("topics", {})
    embeddings_config = version_config.get("embeddings", {})

    chunk_size = chunking_config.get("chunk_size", 5)
    min_chunk_sentences = chunking_config.get("min_chunk_sentences", 2)
    embedding_model = embeddings_config.get("model", "all-mpnet-base-v2")

    # Step 1: Load articles
    print("Loading articles...")
    with get_db() as db:
        articles = db.get_articles(filters=ditwah_filters())
    print(f"  Loaded {len(articles)} articles")

    # Step 2: Chunk articles
    print(f"\nChunking articles ({chunk_size}-sentence windows)...")
    chunks = chunk_articles(articles, chunk_size=chunk_size, min_chunk_sentences=min_chunk_sentences)
    print(f"  Created {len(chunks)} chunks from {len(set(c['article_id'] for c in chunks))} articles")

    # Step 3: Store chunks
    print("\nStoring chunks in database...")
    with get_db() as db:
        db.store_chunks(chunks, args.version_id)
    print(f"  Stored {len(chunks)} chunks")

    # Step 4: Reload chunks from DB to get UUIDs
    print("\nReloading chunks with UUIDs...")
    with get_db() as db:
        db_chunks = db.get_chunks_for_version(args.version_id)
    print(f"  Loaded {len(db_chunks)} chunks from database")

    # Step 5: Embed chunk texts
    print(f"\nEmbedding {len(db_chunks)} chunks with {embedding_model}...")
    config = load_config()
    emb_config = config["embeddings"].copy()
    emb_config["model"] = embedding_model
    emb_client = get_embeddings_client(config=emb_config)

    chunk_texts = [c["chunk_text"] for c in db_chunks]
    chunk_embeddings = np.array(emb_client.embed(chunk_texts, batch_size=64))
    print(f"  Embeddings shape: {chunk_embeddings.shape}")

    # Step 6: Fit BERTopic
    print("\nFitting BERTopic model...")
    modeler = TopicModeler(
        min_topic_size=topic_config.get("min_topic_size", 10),
        embedding_model=embedding_model,
        stop_words=topic_config.get("stop_words"),
        umap_params=topic_config.get("umap"),
        hdbscan_params=topic_config.get("hdbscan"),
        vectorizer_params=topic_config.get("vectorizer"),
    )

    topics, probs = modeler.fit(chunk_texts, chunk_embeddings)

    # Step 7: Label topics
    labeled_topics = label_topics_from_keywords(modeler)

    # Rename article_count -> chunk_count
    for t in labeled_topics:
        t["chunk_count"] = t.pop("article_count")

    # Step 8: Store topics
    print("\nStoring chunk topics...")
    with get_db() as db:
        db.store_chunk_topics(labeled_topics, args.version_id)

    # Step 9: Store assignments
    print("Storing chunk-topic assignments...")
    assignments = []
    for i, db_chunk in enumerate(db_chunks):
        chunk_id = str(db_chunk["id"])
        topic_id = topics[i]
        confidence = float(probs[i]) if probs is not None else 0.0

        if topic_id == -1:
            continue

        assignments.append({
            "chunk_id": chunk_id,
            "topic_id": topic_id,
            "confidence": confidence,
        })

    with get_db() as db:
        db.store_chunk_topic_assignments(assignments, args.version_id)
    print(f"  Stored {len(assignments)} assignments (excluding outliers)")

    # Step 10: Save model
    print("\nSaving BERTopic model to database...")
    import tempfile
    with tempfile.TemporaryDirectory(prefix=f"chunk_bertopic_{args.version_id[:8]}_") as temp_dir:
        temp_model_path = f"{temp_dir}/model"
        modeler.save(temp_model_path)
        save_model_to_version(args.version_id, temp_model_path)

    # Step 11: Generate LLM claim labels
    if not args.skip_claims:
        try:
            import json
            from src.llm import get_llm
            from src.prompts import load_prompt

            print("\nGenerating LLM claim labels...")
            llm = get_llm()
            system_prompt = load_prompt("chunk_topics/claim_label_system.md")

            non_outlier_topics = [t for t in labeled_topics if t["topic_id"] != -1]
            success_count = 0

            with get_db() as db:
                # Get DB IDs for topics
                schema = db.config["schema"]
                with db.cursor() as cur:
                    cur.execute(
                        f"SELECT id, topic_id FROM {schema}.chunk_topics WHERE result_version_id = %s",
                        (args.version_id,)
                    )
                    topic_db_ids = {row["topic_id"]: row["id"] for row in cur.fetchall()}

                for topic in non_outlier_topics:
                    tid = topic["topic_id"]
                    db_id = topic_db_ids.get(tid)
                    if not db_id:
                        continue

                    # Get top 25 chunks by confidence
                    chunks_for_topic = db.get_chunks_by_topic(db_id, args.version_id, limit=25)
                    if not chunks_for_topic:
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
                        # Fallback to keyword-based label
                        fallback = {
                            "claim": " ".join(keywords[:3]) if keywords else topic["name"],
                            "description": f"Topic covering: {', '.join(keywords[:5])}"
                        }
                        db.update_chunk_topic_description(db_id, json.dumps(fallback))
                        print(f"  [T{tid}] (fallback) {fallback['claim']} — {e}")
                        success_count += 1

            print(f"  Labelled {success_count}/{len(non_outlier_topics)} topics")

        except Exception as e:
            print(f"\nWarning: Could not generate claim labels: {e}")
            print("  Pipeline results are still valid — claims can be added later.")
    else:
        print("\nSkipping LLM claim label generation (--skip-claims)")

    # Step 12: Update pipeline status
    update_pipeline_status(args.version_id, "chunk_topics", True)

    # Summary
    n_topics = len([t for t in labeled_topics if t["topic_id"] != -1])
    n_outliers = sum(1 for t in topics if t == -1)

    print("\n" + "=" * 60)
    print("Chunk-Level Topic Discovery Complete")
    print("=" * 60)
    print(f"  Total chunks: {len(db_chunks)}")
    print(f"  Topics discovered: {n_topics}")
    print(f"  Outlier chunks: {n_outliers} ({100*n_outliers/len(db_chunks):.1f}%)")
    print(f"  Assigned chunks: {len(assignments)}")
    print()

    # Print top topics
    print("Top 20 Topics:")
    for topic in sorted(labeled_topics, key=lambda x: x["chunk_count"], reverse=True)[:20]:
        if topic["topic_id"] == -1:
            continue
        print(f"  [T{topic['topic_id']}] {topic['name']} ({topic['chunk_count']} chunks)")

    print(f"\n✓ chunk_topics step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
