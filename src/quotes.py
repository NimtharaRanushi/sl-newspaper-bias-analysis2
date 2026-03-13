"""Quote extraction for news articles using LLM structured output."""

import logging
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from src.llm import get_llm, OpenAILLM
from src.db import get_db, ditwah_filters

logger = logging.getLogger(__name__)


class QuoteType(str, Enum):
    DIRECT = "direct"
    SENTENCE_END = "sentence_end"
    PARAPHRASE = "paraphrase"
    INDIRECT = "indirect"
    SPLIT = "split"


class Quote(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str = Field(..., description="The quoted text exactly as it appears in the article")
    source: Optional[str] = Field(None, description="Name of the speaker or source")
    cue: Optional[str] = Field(None, description="Signal word indicating speech (e.g. said, stated, declared)")
    quote_type: QuoteType = Field(..., alias="type", description="One of: direct, sentence_end, paraphrase, indirect, split")


class ArticleWithQuotes(BaseModel):
    quotes: List[Quote] = Field(..., description="All quotes extracted from the article, in document order")


SYSTEM_PROMPT = "You are an expert at extracting direct and indirect quotes from news articles. Always respond with valid JSON."

EXTRACTION_PROMPT = """Extract all quotes from the following English news article.

Return a JSON object with one field "quotes" containing a list of quote objects.
Each quote object has:
- "source": name of the speaker or source (person or organization), or null
- "cue": the signal word indicating speech (e.g. said, stated, declared), or null
- "content": the quoted text exactly as it appears
- "type": the quote type based on definitions below

Only include quotes with an explicit source attribution. Ignore ironic/hypothetical
quotation marks, slogans, vague sources, or quotes without a clear cue or speaker.
Extract only quotes that appear verbatim in the source text. Do not add
interpretations, additions, or shortened versions.

### QUOTE TYPE DEFINITIONS:

1. direct
   The cue introduces a literal quote, usually preceded by a colon.
   Example: "Looking ahead, Smith said: 'We will invest in infrastructure.'"

2. sentence_end
   The quote appears at the start of the sentence, followed by cue and source.
   Example: "'Unprecedented nonsense,' said the minister."

3. paraphrase
   The statement is summarized or partially reproduced, possibly in quotes.
   Example: "Jones said the company had made a 'critical error'."

4. indirect
   No quotation marks. Content is summarized or described.
   Example: "He condemned the decision, saying it treated citizens unfairly."

5. split
   The quote is interrupted by the cue or additional explanation.
   Example: "'We will invest,' said Lee, 'and it is necessary.'"

Article:
{article_content}"""


def extract_quotes_for_article(llm, article: Dict[str, Any]) -> List[Quote]:
    """Extract quotes from a single article.

    Returns list of Quote objects, or empty list on failure.
    """
    content = f"{article['title']}\n\n{article['content']}"
    prompt = EXTRACTION_PROMPT.format(article_content=content)

    try:
        result = llm.generate_structured(prompt, ArticleWithQuotes, SYSTEM_PROMPT)
        return result.quotes
    except Exception as e:
        logger.warning(f"Quote extraction failed for article {article.get('id')}: {e}")
        return []


def extract_quotes_from_articles(
    version_id: str,
    config: Dict[str, Any],
    batch_size: int = 50
) -> Dict[str, int]:
    """Extract quotes from all Ditwah articles for a version.

    Args:
        version_id: UUID of the quote_extraction result version
        config: Version configuration dict (quote_extraction key)
        batch_size: Articles to process per DB commit batch

    Returns:
        Dict with keys: successful, failed, skipped
    """
    qe_config = config.get("quote_extraction", {})
    provider = qe_config.get("llm_provider", "openai")
    model = qe_config.get("llm_model", "gpt-4o-mini")
    temperature = qe_config.get("temperature", 0.0)

    llm_config = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": 4096
    }
    llm = get_llm(llm_config)

    # Load all Ditwah articles
    with get_db() as db:
        articles = db.get_articles(filters=ditwah_filters())
        already_done = db.get_articles_with_quotes(version_id)

    done_ids = set(already_done)
    articles = [a for a in articles if str(a['id']) not in done_ids]

    print(f"Articles to process: {len(articles)} (skipped {len(done_ids)} already done)")

    counts = {"successful": 0, "failed": 0, "skipped": len(done_ids)}

    for i, article in enumerate(articles):
        quotes = extract_quotes_for_article(llm, article)

        quote_dicts = []
        for order, q in enumerate(quotes):
            quote_dicts.append({
                "quote_content": q.content,
                "quote_source": q.source,
                "cue": q.cue,
                "quote_type": q.quote_type.value,
                "quote_order": order
            })

        try:
            with get_db() as db:
                db.store_quotes(str(article['id']), version_id, quote_dicts)
            counts["successful"] += 1
        except Exception as e:
            logger.error(f"Failed to store quotes for article {article.get('id')}: {e}")
            counts["failed"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(articles)} articles...")

    return counts
