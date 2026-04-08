"""Ditwah Cyclone Chatbot — RAG-powered Q&A."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import streamlit as st

from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style
from src.db import get_db
from src.llm import get_llm, EmbeddingClient

# ============================================================================
# Page config
# ============================================================================

apply_page_style()

# ============================================================================
# Cached resources
# ============================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str):
    """Load a sentence-transformer model. EmbeddingGemma uses 'retrieval' task."""
    is_gemma = "embeddinggemma" in model_name.lower()
    return EmbeddingClient(
        provider="local",
        model=model_name,
        task="retrieval" if is_gemma else None,
    )


@st.cache_resource(show_spinner="Connecting to LLM...")
def load_llm():
    return get_llm()


# ============================================================================
# Data loaders
# ============================================================================

@st.cache_data(ttl=600)
def get_available_embedding_models() -> list:
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT e.embedding_model, COUNT(*) AS article_count
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles n ON e.article_id = n.id
                WHERE n.is_ditwah_cyclone = 1
                GROUP BY e.embedding_model
                ORDER BY article_count DESC
            """)
            return cur.fetchall()


@st.cache_data(ttl=600)
def get_total_ditwah_count() -> int:
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) AS count
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
            """)
            return cur.fetchone()["count"]


def _embedding_to_pg_str(embedding: list) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


def search_ditwah_articles(question_embedding: list, embedding_model: str, min_similarity: float = 0.6) -> list:
    """Semantic search: Ditwah articles above a cosine similarity threshold (hard cap 50)."""
    pg_vec = _embedding_to_pg_str(question_embedding)
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT n.id, n.title, n.content, n.source_id, n.date_posted, n.url,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles n ON e.article_id = n.id
                WHERE n.is_ditwah_cyclone = 1
                  AND e.embedding_model = %s
                  AND 1 - (e.embedding <=> %s::vector) >= %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT 50
                """,
                (pg_vec, embedding_model, pg_vec, min_similarity, pg_vec),
            )
            return cur.fetchall()


def generate_answer(question: str, articles: list, llm) -> str:
    """LLM generates a grounded answer from the top-10 retrieved article excerpts."""
    context_parts = []
    for i, art in enumerate(articles[:10], 1):
        source_name = SOURCE_NAMES.get(art["source_id"], art["source_id"])
        date_str = art["date_posted"].strftime("%d %B %Y") if art["date_posted"] else "Unknown date"
        snippet = (art["content"] or "").strip()[:800]
        context_parts.append(
            f"[Article {i}] {source_name} — {date_str}\n"
            f"Title: {art['title']}\n"
            f"Excerpt: {snippet}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a knowledgeable analyst who answers questions about Cyclone Ditwah "
        "based solely on Sri Lankan newspaper articles provided to you. "
        "Answer factually and concisely. Cite the source newspapers when relevant. "
        "If the articles do not contain enough information, say so clearly."
    )
    prompt = (
        f"Using only the following Sri Lankan newspaper article excerpts about Cyclone Ditwah, "
        f"answer this question:\n\nQuestion: {question}\n\nArticles:\n{context}\n\nAnswer:"
    )
    return llm.generate(prompt, system_prompt=system_prompt).content.strip()


# ============================================================================
# Main UI
# ============================================================================

emb_models = get_available_embedding_models()
if not emb_models:
    st.error("No Ditwah embeddings found.")
    st.stop()

model_options = {
    m["embedding_model"]: m["embedding_model"]
    for m in emb_models
}

col_title, col_settings = st.columns([5, 1])
with col_title:
    st.title("💬 Ditwah Cyclone Chatbot")
    st.caption(
        "Ask any question about Cyclone Ditwah. Get an answer grounded in newspaper articles."
    )
with col_settings:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("⚙️"):
        selected_label = st.selectbox(
            "Embedding model",
            list(model_options.keys()),
            index=0,
            help="Question and articles are embedded with the same model.",
        )
        selected_model = model_options[selected_label]

# --- Chat input ---
question = st.chat_input("Ask a question about Cyclone Ditwah…")

# When a new question is submitted, run the RAG pipeline and store in session_state
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Embedding question…"):
            embed_client = load_embedding_model(selected_model)
            q_embedding = embed_client.embed_single(question)

        with st.spinner("Searching articles…"):
            articles = search_ditwah_articles(q_embedding, selected_model)

        if not articles:
            st.warning("No relevant Ditwah articles found. Try rephrasing your question.")
            st.stop()

        with st.spinner("Generating answer…"):
            llm = load_llm()
            answer = generate_answer(question, articles, llm)

        # Persist in session state so reruns don't lose results
        st.session_state["chatbot_question"] = question
        st.session_state["chatbot_answer"] = answer
        st.session_state["chatbot_articles"] = [dict(a) for a in articles]
        st.session_state["chatbot_article_page"] = 0

        st.markdown(answer)

# If we have a stored result (either just generated or from previous run), show everything
if "chatbot_question" in st.session_state:
    stored_question = st.session_state["chatbot_question"]
    stored_answer   = st.session_state["chatbot_answer"]
    stored_articles = st.session_state["chatbot_articles"]

    # If this is a rerun and not a fresh question, re-show the answer block
    if question is None:
        with st.chat_message("user"):
            st.markdown(stored_question)
        with st.chat_message("assistant"):
            st.markdown(stored_answer)

    total = get_total_ditwah_count()

    n = len(stored_articles)

    st.divider()

    # ---- Article count + cards ----
    st.markdown(
        f"**{n} related article{'s' if n != 1 else ''} found** "
        f"(out of {total} total Ditwah articles)"
    )

    with st.expander("View source articles", expanded=False):
        excerpt_len = 400

        PAGE_SIZE = 5
        total_pages = max(1, math.ceil(n / PAGE_SIZE))
        page = st.session_state.get("chatbot_article_page", 0)
        page_articles = stored_articles[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

        for rank, art in enumerate(page_articles, page * PAGE_SIZE + 1):
            source_name  = SOURCE_NAMES.get(art["source_id"], art["source_id"])
            source_color = SOURCE_COLORS.get(source_name, "#888888")
            date_str = (
                art["date_posted"].strftime("%d %B %Y")
                if art.get("date_posted") else "Unknown date"
            )
            sim_pct = round(float(art["similarity"]) * 100, 1)
            raw     = (art.get("content") or "").strip()
            excerpt = raw[:excerpt_len] + ("…" if len(raw) > excerpt_len else "")

            if rank > page * PAGE_SIZE + 1:
                st.divider()
            st.markdown(
                f"**#{rank}** &nbsp;"
                f'<span style="background:{source_color};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.8em;">'
                f"{source_name}</span> &nbsp; "
                f'<span style="color:#888;font-size:0.9em;">{sim_pct}% match</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{art['title']}**")
            st.caption(date_str)
            st.markdown(excerpt)
            col_link, col_btn = st.columns([1, 1])
            with col_link:
                if art.get("url"):
                    st.link_button("View full article", art["url"])
                else:
                    st.caption("No URL available")
            with col_btn:
                if st.button("Open in Article Insights →", key=f"insights_{art['id']}"):
                    st.switch_page("pages/10_Article_Insights.py", query_params={"article_id": str(art["id"])})

        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("← Prev", disabled=(page == 0), key="art_prev"):
                st.session_state["chatbot_article_page"] = page - 1
                st.rerun()
        with col_info:
            st.caption(f"Page {page + 1} of {total_pages} ({n} articles)")
        with col_next:
            if st.button("Next →", disabled=(page >= total_pages - 1), key="art_next"):
                st.session_state["chatbot_article_page"] = page + 1
                st.rerun()

# Welcome state (no question asked yet)
elif "chatbot_question" not in st.session_state:
    st.info(
        "Ask a question in the chat box below — for example:\n\n"
        "- *How many people were displaced by Cyclone Ditwah?*\n"
        "- *What was the government's response to the cyclone?*\n"
        "- *Which areas were most affected by flooding?*\n"
        "- *What international aid was provided?*"
    )
