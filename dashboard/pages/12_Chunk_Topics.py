"""Chunk-Level Topic Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import streamlit as st
import pandas as pd

from data.loaders import (
    load_chunk_topics, load_chunk_topic_by_source, load_chunks_by_topic,
    load_chunk_topic_stats, load_chunk_outlet_totals, load_bertopic_model,
    load_outlier_chunks
)
from components.source_mapping import SOURCE_NAMES
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Chunk-Level Topic Analysis")

# Create version button and selector
render_create_version_button('chunk_topics')
version_id = render_version_selector('chunk_topics')

if not version_id:
    st.stop()

# Load data
topics = load_chunk_topics(version_id)
if not topics:
    st.warning("No chunk topics found for this version. Run the pipeline first.")
    st.code(f"PYTHONHASHSEED=42 python3 scripts/chunk_topics/01_discover_chunk_topics.py --version-id {version_id}")
    st.stop()

stats = load_chunk_topic_stats(version_id)
topic_source_data = load_chunk_topic_by_source(version_id)
outlet_totals = load_chunk_outlet_totals(version_id)

# --- Summary Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Chunks", f"{stats.get('total_chunks', 0):,}")
with col2:
    st.metric("Topics Found", stats.get("total_topics", 0))
with col3:
    outlier_pct = (
        stats["outlier_chunks"] / stats["total_chunks"] * 100
        if stats.get("total_chunks", 0) > 0 else 0
    )
    st.metric("Outlier %", f"{outlier_pct:.1f}%")
with col4:
    st.metric("Articles", f"{stats.get('total_articles', 0):,}")

st.divider()

# --- Browse Chunks by Topic (PRIMARY FEATURE) ---
st.subheader("Browse Chunks by Topic")

# Topic dropdown — show claim label if available, fall back to keyword name
def _topic_label(t):
    desc = t.get("description")
    if desc:
        try:
            parsed = json.loads(desc) if isinstance(desc, str) else desc
            claim = parsed.get("claim")
            if claim:
                return f"[T{t['topic_id']}] {claim} ({t['chunk_count']} chunks)"
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return f"[T{t['topic_id']}] {t['name']} ({t['chunk_count']} chunks)"

topic_options = {_topic_label(t): t for t in topics}

selected_label = st.selectbox(
    "Select a topic",
    options=list(topic_options.keys()),
    key="chunk_topic_selector"
)

selected_topic = topic_options[selected_label]

# Source filter
OUTLETS = ["dailynews_en", "themorning_en", "ft_en", "island_en"]
source_options = ["All"] + [SOURCE_NAMES.get(o, o) for o in OUTLETS]
source_filter = st.selectbox("Filter by source", source_options, key="chunk_source_filter")

# Resolve source filter to source_id
source_id_filter = None
if source_filter != "All":
    source_id_filter = next(
        (sid for sid, name in SOURCE_NAMES.items() if name == source_filter), None
    )

# Load and display chunks
chunks = load_chunks_by_topic(
    version_id, selected_topic["id"],
    source_id=source_id_filter, limit=50
)

if chunks:
    st.caption(f"Showing {len(chunks)} chunks (sorted by confidence)")

    for chunk in chunks:
        source_name = SOURCE_NAMES.get(chunk["source_id"], chunk["source_id"])
        date_str = chunk["date_posted"].strftime("%Y-%m-%d") if chunk["date_posted"] else ""

        with st.container(border=True):
            # Header row
            col1, col2 = st.columns([3, 1])
            with col1:
                if chunk.get("url"):
                    st.markdown(f"[{chunk['title']}]({chunk['url']})")
                else:
                    st.markdown(f"**{chunk['title']}**")
            with col2:
                st.caption(f"{source_name} &bull; {date_str} &bull; Chunk {chunk['chunk_index']}")

            # Chunk text
            st.markdown(
                f"<div style='padding: 10px; border-radius: 5px; "
                f"font-size: 0.9em; line-height: 1.5;'>{chunk['chunk_text']}</div>",
                unsafe_allow_html=True,
            )
else:
    st.info("No chunks found for this topic with the current filter.")

# --- Coverage Analysis ---
st.divider()
st.subheader("Coverage Analysis")
st.markdown(
    "Which topics show the largest coverage differences between outlets? "
)
st.markdown(
    "**Spread** = max proportion − min proportion across outlets."
)

if topic_source_data and outlet_totals:
    ts_df = pd.DataFrame(topic_source_data)
    all_outlets = [o for o in OUTLETS if o in outlet_totals]
    outlet_names = [SOURCE_NAMES[o] for o in all_outlets]

    top_topic_names = [t["name"] for t in topics]
    ts_filtered = ts_df[ts_df["topic_name"].isin(top_topic_names)]

    bias_rows = []
    for t in topics:
        tname = t["name"]
        proportions = {}

        for o in all_outlets:
            count = ts_filtered[
                (ts_filtered["topic_name"] == tname) & (ts_filtered["source_id"] == o)
            ]["count"].sum()
            total = outlet_totals.get(o, 1)
            proportions[o] = (count / total) * 100

        vals = list(proportions.values())
        spread = max(vals) - min(vals)
        max_outlet = SOURCE_NAMES[max(proportions, key=proportions.get)]
        min_outlet = SOURCE_NAMES[min(proportions, key=proportions.get)]

        # Use claim label if available, fall back to keywords
        label = None
        desc = t.get("description")
        if desc:
            try:
                parsed = json.loads(desc) if isinstance(desc, str) else desc
                label = parsed.get("claim") or parsed.get("aspect")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        if not label:
            keywords = t.get("keywords") or []
            label = ", ".join(keywords[:5]).title() if keywords else tname

        row = {"Topic": f"T{t['topic_id']}", "Label": label}
        for o in all_outlets:
            row[SOURCE_NAMES[o]] = round(proportions[o], 1)
        row["Spread"] = round(spread, 1)
        row["Most"] = max_outlet
        row["Least"] = min_outlet
        bias_rows.append(row)

    bias_df = pd.DataFrame(bias_rows).sort_values("Spread", ascending=False)

    display_cols = ["Topic", "Label"] + outlet_names + ["Spread", "Most", "Least"]
    st.dataframe(
        bias_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spread": st.column_config.NumberColumn(format="%.1f pp"),
            **{name: st.column_config.NumberColumn(format="%.1f%%") for name in outlet_names},
        },
    )

st.divider()
st.subheader("Omission Detection")
st.markdown(
    "Topics present in some outlets but absent from others "
    "(threshold: at least 5 chunks in one outlet)."
)

if topic_source_data:
    ts_df = pd.DataFrame(topic_source_data)
    all_outlets = [o for o in OUTLETS if o in (outlet_totals or {})]

    all_topic_ids = ts_df["topic_id"].unique()
    omission_rows = []

    for _, t in enumerate(topics):
        tid = t["topic_id"]
        tname = t["name"]
        counts = {}
        for o in all_outlets:
            match = ts_df[(ts_df["topic_id"] == tid) & (ts_df["source_id"] == o)]
            counts[o] = int(match["count"].sum()) if len(match) > 0 else 0

        present = [o for o, c in counts.items() if c > 0]
        absent = [o for o, c in counts.items() if c == 0]

        if absent and max(counts.values()) >= 5:
            # Use LLM-generated claim if available, fall back to keywords
            claim_text = None
            desc = t.get("description")
            if desc:
                try:
                    parsed = json.loads(desc) if isinstance(desc, str) else desc
                    claim_text = parsed.get("claim")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            if not claim_text:
                keywords = t.get("keywords") or []
                claim_text = ", ".join(keywords[:5])
            omission_rows.append({
                "Topic": f"T{tid}",
                "Claim": claim_text,
                "Total": sum(counts.values()),
                **{SOURCE_NAMES[o]: counts[o] for o in all_outlets},
                "Present In": ", ".join(SOURCE_NAMES[o] for o in present),
                "Absent From": ", ".join(SOURCE_NAMES[o] for o in absent),
            })

    if omission_rows:
        omission_df = pd.DataFrame(omission_rows).sort_values("Total", ascending=False)
        st.caption(f"Found {len(omission_df)} topics with omissions")
        st.dataframe(omission_df, use_container_width=True, hide_index=True)
    else:
        st.info("No omissions detected with current thresholds.")

# # --- Statistical Significance ---
# st.divider()
# st.subheader("Statistical Significance")
# st.markdown("Chi-squared test with Bonferroni correction for uneven topic distributions.")

# if topic_source_data and outlet_totals:
#     from scipy.stats import chi2_contingency

#     ts_df = pd.DataFrame(topic_source_data)
#     all_outlets = [o for o in OUTLETS if o in outlet_totals]
#     outlet_total_chunks = {o: outlet_totals[o] for o in all_outlets}

#     n_tests = len(topics)
#     alpha_bonferroni = 0.05 / n_tests if n_tests > 0 else 0.05

#     sig_rows = []
#     for t in topics:
#         tid = t["topic_id"]
#         counts = []
#         for o in all_outlets:
#             match = ts_df[(ts_df["topic_id"] == tid) & (ts_df["source_id"] == o)]
#             counts.append(int(match["count"].sum()) if len(match) > 0 else 0)

#         if sum(counts) < 5 or all(c == counts[0] for c in counts):
#             continue

#         other_counts = [outlet_total_chunks.get(o, 0) - c for o, c in zip(all_outlets, counts)]
#         table = np.array([counts, other_counts])

#         if (table < 0).any():
#             continue

#         try:
#             chi2, p, dof, expected = chi2_contingency(table)
#         except ValueError:
#             continue

#         if p < alpha_bonferroni:
#             keywords = t.get("keywords") or []
#             sig_rows.append({
#                 "Topic": f"T{tid}",
#                 "Keywords": ", ".join(keywords[:5]),
#                 "chi2": round(chi2, 1),
#                 "p-value": f"{p:.2e}",
#                 "Total": sum(counts),
#                 **{SOURCE_NAMES[o]: c for o, c in zip(all_outlets, counts)},
#             })

#     if sig_rows:
#         sig_df = pd.DataFrame(sig_rows).sort_values("chi2", ascending=False)
#         st.caption(
#             f"Significant topics (Bonferroni alpha={alpha_bonferroni:.6f}): "
#             f"{len(sig_df)}/{n_tests}"
#         )
#         st.dataframe(sig_df, use_container_width=True, hide_index=True)
#     else:
#         st.info("No statistically significant topics found.")

# --- BERTopic Visualizations ---
st.divider()
st.subheader("Topic Model Visualizations")

topic_model = load_bertopic_model(version_id)
if topic_model:
    viz_option = st.selectbox(
        "Select visualization",
        [
            "Topic Similarity Heatmap",
            "Hierarchical Topic Clustering",
            "Topic Bar Charts",
        ],
        key="chunk_viz_selector"
    )

    try:
        n_topics = len(topics)
        top_n = min(n_topics, 30)

        if viz_option == "Topic Similarity Heatmap":
            st.caption("Similarity between topics based on c-TF-IDF vectors")
            fig = topic_model.visualize_heatmap(top_n_topics=top_n)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Hierarchical Topic Clustering":
            st.caption("How topics group into broader categories")
            fig = topic_model.visualize_hierarchy(top_n_topics=top_n)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Topic Bar Charts":
            st.caption("Top keywords per topic")
            top_ids = [t["topic_id"] for t in topics[:20]]
            fig = topic_model.visualize_barchart(top_n_topics=20, topics=top_ids)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating visualization: {e}")
else:
    st.info("BERTopic model not found in database for this version.")

# --- Outlier Chunks (not in any topic) ---
st.divider()
st.subheader("Outlier Chunks")
st.markdown(
    f"Chunks not assigned to any topic cluster "
    f"(**{stats.get('outlier_chunks', 0):,}** chunks, "
    f"{outlier_pct:.1f}% of total)."
)

outlier_source_options = ["All"] + [SOURCE_NAMES.get(o, o) for o in OUTLETS]
outlier_source_filter = st.selectbox(
    "Filter by source", outlier_source_options, key="outlier_source_filter"
)

outlier_source_id = None
if outlier_source_filter != "All":
    outlier_source_id = next(
        (sid for sid, name in SOURCE_NAMES.items() if name == outlier_source_filter), None
    )

outlier_chunks = load_outlier_chunks(version_id, source_id=outlier_source_id, limit=50)

if outlier_chunks:
    st.caption(f"Showing {len(outlier_chunks)} outlier chunks (sorted by date)")

    for chunk in outlier_chunks:
        source_name = SOURCE_NAMES.get(chunk["source_id"], chunk["source_id"])
        date_str = chunk["date_posted"].strftime("%Y-%m-%d") if chunk["date_posted"] else ""

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                if chunk.get("url"):
                    st.markdown(f"[{chunk['title']}]({chunk['url']})")
                else:
                    st.markdown(f"**{chunk['title']}**")
            with col2:
                st.caption(f"{source_name} &bull; {date_str} &bull; Chunk {chunk['chunk_index']}")

            st.markdown(
                f"<div style='padding: 10px; border-radius: 5px; "
                f"font-size: 0.9em; line-height: 1.5;'>{chunk['chunk_text']}</div>",
                unsafe_allow_html=True,
            )
else:
    st.info("No outlier chunks found with the current filter.")
