"""Text preprocessing utilities for topic modeling.

Provides a shared token pattern, domain stop-word list, and text cleaner
used across LDA, SBM, and GloVe topic models.
"""

import re

# ---------------------------------------------------------------------------
# Token pattern — only pure alphabetic tokens of 3+ characters.
# This silently drops: numbers ("450", "2026"), alphanumerics ("30th"),
# abbreviations ("rs", "mr", "dr"), single letters and 2-char noise.
# ---------------------------------------------------------------------------
TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"

# ---------------------------------------------------------------------------
# Domain stop words (extend sklearn's ENGLISH_STOP_WORDS)
# ---------------------------------------------------------------------------
DOMAIN_STOP_WORDS = [
    # ── Reporting verbs (pervasive in news, not topic-discriminative) ────────
    "said", "say", "says", "saying", "told", "tell", "tells",
    "added", "add", "noted", "note", "stated", "state", "states",
    "reported", "report", "reports", "announced", "announce",
    "confirmed", "confirm", "explained", "explain", "mentioned",
    "described", "describe", "pointed", "highlighted", "emphasized",
    "indicated", "revealed", "reveal", "according", "stated",

    # ── Discourse / connector words ──────────────────────────────────────────
    "addition", "also", "however", "therefore", "furthermore", "moreover",
    "meanwhile", "including", "included", "include", "based", "following",
    "followed", "result", "results", "thus", "hence", "indeed",
    "despite", "although", "though", "yet", "still", "even",

    # ── Generic auxiliary / modal verbs ─────────────────────────────────────
    "would", "could", "should", "must", "shall", "might",
    "make", "made", "making", "take", "taken", "taking",
    "come", "came", "coming", "goes", "went", "going",
    "need", "needs", "use", "used", "using", "give", "given", "giving",
    "provide", "provided", "providing", "ensure", "ensuring",
    "put", "puts", "putting", "set", "sets", "setting",
    "hold", "holds", "held", "holding", "keep", "keeps", "kept",

    # ── Generic nouns (too broad to discriminate topics) ─────────────────────
    "people", "person", "public", "number", "numbers",
    "way", "ways", "fact", "issue", "issues", "matter", "matters",
    "time", "times", "year", "years", "month", "months",
    "day", "days", "week", "weeks", "today", "yesterday",
    "morning", "night", "evening",
    "percent", "total", "overall", "level", "levels", "rate", "rates",
    "area", "areas", "region", "regions", "place", "places",
    "part", "parts", "section", "group", "groups", "type", "types",
    "work", "works", "working", "meeting", "meetings", "event", "events",
    "situation", "conditions", "condition", "case", "cases",
    "action", "actions", "step", "steps", "measure", "measures",
    "process", "processes", "plan", "plans",

    # ── Generic adjectives / adverbs ─────────────────────────────────────────
    "just", "like", "let", "far", "new", "old", "good", "bad",
    "first", "second", "third", "last", "next", "latest",
    "high", "low", "large", "small", "long", "short", "major", "minor",
    "many", "much", "more", "most", "less", "least", "several",
    "various", "different", "similar", "important", "key", "main",
    "current", "recent", "previous", "future", "possible",
    "available", "necessary", "full", "general", "special",

    # ── Sri Lanka–specific noise ─────────────────────────────────────────────
    "sri", "lanka", "lankan", "srilanka",
    "colombo",          # appears in nearly every article
    "north", "south", "east", "west",
    "northern", "southern", "eastern", "western", "central",

    # ── News titles / honorifics ─────────────────────────────────────────────
    "minister", "ministry",  # keep only if truly discriminative — comment out if needed
    "president", "secretary",

    # ── Numbers as words ─────────────────────────────────────────────────────
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million", "billion",
]

# ---------------------------------------------------------------------------
# Pre-compiled patterns for text cleaning
# ---------------------------------------------------------------------------
_RE_URL     = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL   = re.compile(r"\S+@\S+")
_RE_NUMERIC = re.compile(r"\b\d[\d,./%-]*\b")        # standalone numbers / prices
_RE_SPACE   = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean raw article text before vectorization.

    Steps:
      1. Remove URLs and email addresses.
      2. Remove standalone numeric tokens (prices, dates, percentages, …).
      3. Collapse multiple whitespace characters.

    The token pattern (``TOKEN_PATTERN``) applied by the vectorizer then
    further drops 2-char tokens, mixed alphanumeric strings, and punctuation.
    """
    if not text:
        return ""
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_NUMERIC.sub(" ", text)
    text = _RE_SPACE.sub(" ", text)
    return text.strip()
