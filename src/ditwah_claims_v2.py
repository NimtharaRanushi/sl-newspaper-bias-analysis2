"""
Ditwah Claims Analysis - Version 2

New approach:
1. Generate claims FROM each individual article (not all articles at once)
2. Deduplicate similar claims using LLM
3. Filter claims with < 5 articles
4. Ensure all claims have both sentiment AND stance data
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from uuid import UUID
from collections import defaultdict

from src.db import get_db

logger = logging.getLogger(__name__)


def filter_ditwah_articles() -> List[Dict]:
    """
    Get all articles marked as Ditwah-related.

    Returns:
        List of article dicts with id, title, content, source_id, date_posted
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
                ORDER BY date_posted, id
            """)
            articles = cur.fetchall()

    logger.info(f"Found {len(articles)} Ditwah articles")
    return [dict(a) for a in articles]


def generate_claims_from_single_article(llm, article: Dict, config: Dict) -> List[Dict]:
    """
    Generate 1-3 claims from a single article using LLM.

    Args:
        llm: LLM client instance
        article: Article dict with title, content
        config: Generation config with categories

    Returns:
        List of claim dicts with claim_text and claim_category
    """
    categories = config.get('categories', [
        'casualties_and_displacement',
        'infrastructure_damage',
        'economic_impact',
        'government_response',
        'international_response',
        'humanitarian_aid'
    ])

    prompt = f"""You are analyzing a news article about Cyclone Ditwah in Sri Lanka.

Extract 1-3 specific, verifiable factual claims from this article. Each claim should be:
- A concrete factual statement (not opinion or speculation)
- Specific enough to verify against other articles
- Newsworthy and significant
- Self-contained (understandable without reading the article)

Article Title: {article['title']}

Article Content:
{article['content'][:2000]}

Categories: {', '.join(categories)}

Return ONLY a JSON array of claims in this format:
[
  {{"claim_text": "Specific factual claim here", "claim_category": "category_name"}},
  {{"claim_text": "Another specific claim", "claim_category": "category_name"}}
]

Extract 1-3 claims. Return valid JSON only."""

    try:
        response = llm.generate(prompt=prompt, json_mode=True)
        claims = json.loads(response.content)

        if not isinstance(claims, list):
            logger.warning(f"LLM returned non-list for article {article['id']}")
            return []

        # Validate claims
        valid_claims = []
        for claim in claims:
            if isinstance(claim, dict) and 'claim_text' in claim and 'claim_category' in claim:
                # Add article ID to track which article generated this claim
                claim['source_article_id'] = str(article['id'])
                valid_claims.append(claim)

        return valid_claims

    except Exception as e:
        logger.error(f"Error generating claims from article {article['id']}: {e}")
        return []


def deduplicate_claims_with_llm(llm, all_claims: List[Dict], config: Dict) -> List[Dict]:
    """
    Deduplicate similar claims using LLM to merge duplicates.

    Args:
        llm: LLM client instance
        all_claims: List of all claims from all articles
        config: Configuration dict

    Returns:
        List of deduplicated claims with article_ids list
    """
    logger.info(f"Deduplicating {len(all_claims)} claims...")

    # Group claims by category first for efficiency
    by_category = defaultdict(list)
    for claim in all_claims:
        by_category[claim['claim_category']].append(claim)

    deduplicated = []

    for category, claims in by_category.items():
        logger.info(f"Processing {len(claims)} claims in category: {category}")

        # Build claim groups using LLM
        claim_groups = []  # Each group is a list of similar claims

        for claim in claims:
            # Check if this claim is similar to any existing group
            merged = False

            for group in claim_groups:
                # Compare with first claim in group
                representative = group[0]

                # Ask LLM if claims are about the same fact
                prompt = f"""Are these two claims describing the SAME factual event or statement?

Claim 1: {representative['claim_text']}
Claim 2: {claim['claim_text']}

Answer with ONLY "YES" or "NO".

If they describe the same core fact (even with different wording), answer YES.
If they describe different facts, answer NO."""

                try:
                    response = llm.generate(prompt=prompt, json_mode=False)
                    answer = response.content.strip().upper()

                    if 'YES' in answer:
                        group.append(claim)
                        merged = True
                        break

                except Exception as e:
                    logger.error(f"Error comparing claims: {e}")
                    continue

            if not merged:
                # Start new group
                claim_groups.append([claim])

        # Merge each group into single claim
        for group in claim_groups:
            # Collect all article IDs
            article_ids = list(set([c['source_article_id'] for c in group]))

            # Use the most common or first claim text
            merged_claim = {
                'claim_text': group[0]['claim_text'],
                'claim_category': category,
                'article_ids': article_ids,
                'article_count': len(article_ids)
            }

            deduplicated.append(merged_claim)

    logger.info(f"Deduplicated to {len(deduplicated)} unique claims")
    return deduplicated


def filter_claims_by_article_count(claims: List[Dict], min_articles: int = 5) -> List[Dict]:
    """
    Filter out claims with fewer than min_articles.

    Args:
        claims: List of deduplicated claims
        min_articles: Minimum article count threshold

    Returns:
        Filtered list of claims
    """
    filtered = [c for c in claims if c['article_count'] >= min_articles]

    logger.info(f"Filtered {len(claims)} claims to {len(filtered)} claims with >={min_articles} articles")
    return filtered


def analyze_claim_sentiment(claim: Dict, config: Dict) -> List[Dict]:
    """
    Fetch sentiment scores for articles in this claim.

    Args:
        claim: Claim dict with article_ids list
        config: Sentiment config

    Returns:
        List of sentiment records
    """
    sentiment_records = []
    primary_model = config.get('primary_model', 'roberta')

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            for article_id in claim['article_ids']:
                # Fetch existing sentiment
                cur.execute(f"""
                    SELECT
                        a.id,
                        a.source_id,
                        sa.overall_sentiment,
                        sa.model_name
                    FROM {schema}.news_articles a
                    INNER JOIN {schema}.sentiment_analyses sa ON a.id = sa.article_id
                    WHERE a.id = %s AND sa.model_type = %s
                    LIMIT 1
                """, (article_id, primary_model))

                result = cur.fetchone()
                if result:
                    sentiment_records.append({
                        'article_id': str(result['id']),
                        'source_id': result['source_id'],
                        'sentiment_score': result['overall_sentiment'],
                        'sentiment_model': result['model_name']
                    })

    return sentiment_records


def analyze_claim_stance_batch(llm, claim_text: str, articles: List[Dict], config: Dict) -> List[Dict]:
    """
    Analyze stance for a batch of articles using LLM.

    Args:
        llm: LLM client
        claim_text: The claim text
        articles: List of article dicts (id, title, content, source_id)
        config: Stance config

    Returns:
        List of stance records
    """
    if not articles:
        return []

    # Prepare article summaries for LLM
    article_summaries = []
    for i, article in enumerate(articles):
        article_summaries.append(f"""
Article {i+1} [ID: {article['id']}]
Title: {article['title']}
Content: {article['content'][:800]}...""")

    prompt = f"""Analyze how each article relates to this claim:

CLAIM: "{claim_text}"

For each article below, determine the article's stance toward this claim:
- STRONGLY_AGREE (+1.0): Article explicitly supports/confirms the claim
- AGREE (+0.5): Article generally supports the claim
- NEUTRAL (0.0): Article mentions the claim without clear support/opposition
- DISAGREE (-0.5): Article contradicts or questions the claim
- STRONGLY_DISAGREE (-1.0): Article explicitly contradicts the claim

{chr(10).join(article_summaries)}

Return JSON array with stance for each article:
[
  {{
    "article_id": "article_id_here",
    "stance_score": 0.5,
    "stance_label": "agree",
    "confidence": 0.8,
    "reasoning": "Brief explanation",
    "supporting_quotes": ["Quote 1", "Quote 2"]
  }}
]

Return valid JSON only."""

    try:
        response = llm.generate(prompt=prompt, json_mode=True)
        stances = json.loads(response.content)

        if not isinstance(stances, list):
            return []

        # Add source_id from articles
        article_map = {str(a['id']): a['source_id'] for a in articles}
        for stance in stances:
            if 'article_id' in stance:
                stance['source_id'] = article_map.get(stance['article_id'], 'unknown')

        return stances

    except Exception as e:
        logger.error(f"Error analyzing stance: {e}")
        return []


def store_claims_v2(version_id: UUID, claims: List[Dict], llm_provider: str, llm_model: str) -> List[Tuple[UUID, Dict]]:
    """
    Store deduplicated claims with article counts.

    Returns:
        List of tuples: (claim_id, claim_dict)
    """
    with get_db() as db:
        schema = db.config["schema"]
        results = []

        with db.cursor() as cur:
            for i, claim in enumerate(claims):
                cur.execute(f"""
                    INSERT INTO {schema}.ditwah_claims (
                        result_version_id,
                        claim_text,
                        claim_category,
                        claim_order,
                        article_count,
                        llm_provider,
                        llm_model
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (result_version_id, claim_text) DO UPDATE
                    SET claim_category = EXCLUDED.claim_category,
                        article_count = EXCLUDED.article_count
                    RETURNING id
                """, (
                    str(version_id),
                    claim['claim_text'],
                    claim['claim_category'],
                    i + 1,
                    claim['article_count'],
                    llm_provider,
                    llm_model
                ))

                claim_id = cur.fetchone()['id']
                results.append((claim_id, claim))

        logger.info(f"Stored {len(results)} claims")
        return results


def store_claim_sentiment(claim_id: UUID, sentiment_records: List[Dict]) -> int:
    """Store sentiment records."""
    if not sentiment_records:
        return 0

    with get_db() as db:
        schema = db.config["schema"]
        count = 0

        with db.cursor() as cur:
            for record in sentiment_records:
                cur.execute(f"""
                    INSERT INTO {schema}.claim_sentiment (
                        claim_id,
                        article_id,
                        source_id,
                        sentiment_score,
                        sentiment_model
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (claim_id, article_id) DO UPDATE
                    SET sentiment_score = EXCLUDED.sentiment_score
                """, (
                    str(claim_id),
                    record['article_id'],
                    record['source_id'],
                    record['sentiment_score'],
                    record['sentiment_model']
                ))
                count += 1

        return count


def store_claim_stance(claim_id: UUID, stance_records: List[Dict]) -> int:
    """Store stance records."""
    if not stance_records:
        return 0

    with get_db() as db:
        schema = db.config["schema"]
        count = 0

        with db.cursor() as cur:
            for record in stance_records:
                cur.execute(f"""
                    INSERT INTO {schema}.claim_stance (
                        claim_id,
                        article_id,
                        source_id,
                        stance_score,
                        stance_label,
                        confidence,
                        reasoning,
                        supporting_quotes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (claim_id, article_id) DO UPDATE
                    SET stance_score = EXCLUDED.stance_score,
                        stance_label = EXCLUDED.stance_label,
                        confidence = EXCLUDED.confidence
                """, (
                    str(claim_id),
                    record['article_id'],
                    record['source_id'],
                    record.get('stance_score', 0.0),
                    record.get('stance_label', 'neutral'),
                    record.get('confidence', 0.5),
                    record.get('reasoning', ''),
                    json.dumps(record.get('supporting_quotes', []))
                ))
                count += 1

        return count


def generate_claims_pipeline_v2(version_id: UUID, config: Dict) -> Dict[str, Any]:
    """
    New pipeline for Ditwah claims analysis.

    Steps:
    1. Get all Ditwah articles
    2. Generate 1-3 claims FROM each article using LLM
    3. Deduplicate similar claims using LLM
    4. Filter claims with < 5 articles
    5. For each remaining claim:
       a. Fetch sentiment from existing sentiment_analyses
       b. Analyze stance with LLM (batched)
       c. Store to database
    6. Only keep claims with BOTH sentiment AND stance data

    Args:
        version_id: Result version ID
        config: Configuration dict

    Returns:
        Summary dict with counts
    """
    from src.llm import get_llm

    logger.info(f"Starting Ditwah claims pipeline V2 for version {version_id}")

    # Step 1: Get all Ditwah articles
    articles = filter_ditwah_articles()
    if not articles:
        logger.error("No Ditwah articles found")
        return {'error': 'No Ditwah articles found'}

    logger.info(f"Processing {len(articles)} Ditwah articles")

    # Step 2: Generate claims from each article
    llm_config = config.get('llm', {})
    llm = get_llm(llm_config)
    generation_config = config.get('generation', {})

    all_claims = []
    articles_with_claims = 0

    for i, article in enumerate(articles):
        if i % 100 == 0:
            logger.info(f"Processing article {i+1}/{len(articles)}...")

        claims = generate_claims_from_single_article(llm, article, generation_config)
        if claims:
            all_claims.extend(claims)
            articles_with_claims += 1

    logger.info(f"✅ Generated {len(all_claims)} claims from {articles_with_claims}/{len(articles)} articles")

    if not all_claims:
        logger.error("No claims generated from any article")
        return {'error': 'No claims generated'}

    # Step 3: Deduplicate claims
    deduplicated_claims = deduplicate_claims_with_llm(llm, all_claims, config)

    # Step 4: Filter by article count
    min_articles = generation_config.get('min_articles', 5)
    filtered_claims = filter_claims_by_article_count(deduplicated_claims, min_articles)

    if not filtered_claims:
        logger.error(f"No claims with >= {min_articles} articles")
        return {'error': f'No claims with >= {min_articles} articles'}

    logger.info(f"✅ {len(filtered_claims)} claims passed article count filter")

    # Step 5: Store claims and analyze sentiment/stance
    llm_provider = llm_config.get('provider', 'mistral')
    llm_model = llm_config.get('model', 'mistral-large-latest')

    claim_results = store_claims_v2(version_id, filtered_claims, llm_provider, llm_model)

    sentiment_config = config.get('sentiment', {})
    stance_config = config.get('stance', {})
    batch_size = stance_config.get('batch_size', 5)

    total_sentiment = 0
    total_stance = 0
    claims_with_both = 0

    for claim_id, claim in claim_results:
        claim_text = claim['claim_text']
        logger.info(f"Analyzing claim: {claim_text[:60]}...")

        # Analyze sentiment
        sentiment_records = analyze_claim_sentiment(claim, sentiment_config)
        if sentiment_records:
            count = store_claim_sentiment(claim_id, sentiment_records)
            total_sentiment += count
            logger.info(f"  Sentiment: {count} records")
        else:
            logger.warning(f"  No sentiment data found")
            continue

        # Analyze stance in batches
        article_ids = claim['article_ids']
        stance_records = []

        # Fetch full article data for stance analysis
        with get_db() as db:
            schema = db.config["schema"]
            with db.cursor() as cur:
                cur.execute(f"""
                    SELECT id, title, content, source_id
                    FROM {schema}.news_articles
                    WHERE id = ANY(%s)
                """, (article_ids,))
                full_articles = [dict(a) for a in cur.fetchall()]

        # Process in batches
        for i in range(0, len(full_articles), batch_size):
            batch = full_articles[i:i+batch_size]
            batch_stances = analyze_claim_stance_batch(llm, claim_text, batch, stance_config)
            stance_records.extend(batch_stances)

        if stance_records:
            count = store_claim_stance(claim_id, stance_records)
            total_stance += count
            logger.info(f"  Stance: {count} records")
            claims_with_both += 1
        else:
            logger.warning(f"  No stance data generated")

    logger.info(f"✅ Pipeline complete!")
    logger.info(f"   Claims with both sentiment & stance: {claims_with_both}/{len(claim_results)}")
    logger.info(f"   Total sentiment records: {total_sentiment}")
    logger.info(f"   Total stance records: {total_stance}")

    return {
        'total_articles': len(articles),
        'articles_with_claims': articles_with_claims,
        'raw_claims': len(all_claims),
        'deduplicated_claims': len(deduplicated_claims),
        'filtered_claims': len(filtered_claims),
        'claims_with_data': claims_with_both,
        'sentiment_records': total_sentiment,
        'stance_records': total_stance
    }
