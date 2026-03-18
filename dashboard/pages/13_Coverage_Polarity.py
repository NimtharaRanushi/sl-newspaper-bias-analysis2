"""Coverage Polarity Page — how differently do outlets frame the same event?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data.loaders import (
    load_available_models,
    load_event_polarity,
    load_event_articles_with_sentiment,
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector
from components.styling import apply_page_style

apply_page_style()

st.title("Coverage Polarity")
st.caption(
    "Identifies events where outlets diverge in sentiment — "
    "one portraying the story positively, another negatively."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
version_id = render_version_selector("clustering")
if not version_id:
    st.stop()

available_models = load_available_models()
if not available_models:
    st.warning("No sentiment data found. Run a sentiment analysis script first.")
    st.stop()

MODEL_DISPLAY = {
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT",
    "finbert": "FinBERT",
    "vader": "VADER",
    "textblob": "TextBlob",
    "local": "Local (RoBERTa)",
}

model_list = [m["model_type"] for m in available_models]
selected_model = st.selectbox(
    "Sentiment model",
    options=model_list,
    format_func=lambda x: MODEL_DISPLAY.get(x, x.upper()),
)

min_spread = st.slider(
    "Minimum polarity spread",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Only show events where the gap between the most positive and most negative source is at least this large.",
)

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
raw = load_event_polarity(version_id, selected_model)
if not raw:
    st.warning("No event polarity data found for this version and model.")
    st.info(
        "Make sure you have:\n"
        "1. Run event clustering: `python3 scripts/clustering/02_cluster_events.py --version-id <id>`\n"
        "2. Run sentiment analysis: `python3 scripts/sentiment/01_analyze_sentiment.py`"
    )
    st.stop()

df = pd.DataFrame(raw)
df["source_name"] = df["source_id"].map(SOURCE_NAMES).fillna(df["source_id"])

# Per-event stats: spread = max - min avg_sentiment across sources
event_stats = (
    df.groupby(["cluster_id", "cluster_name", "date_start", "date_end"])
    .agg(
        max_sentiment=("avg_sentiment", "max"),
        min_sentiment=("avg_sentiment", "min"),
        sources_covered=("source_name", "nunique"),
        total_articles=("article_count", "sum"),
    )
    .reset_index()
)
event_stats["polarity_spread"] = (
    event_stats["max_sentiment"] - event_stats["min_sentiment"]
).round(3)
event_stats = event_stats[event_stats["polarity_spread"] >= min_spread].sort_values(
    "polarity_spread", ascending=False
)

if event_stats.empty:
    st.info(
        f"No events with polarity spread ≥ {min_spread:.1f}. "
        "Try lowering the minimum spread slider."
    )
    st.stop()

# ── Section 1: Divergence leaderboard ────────────────────────────────────────
st.markdown("### Polarity Divergence Leaderboard")
st.caption(
    "Events ranked by sentiment spread across sources. "
    "High spread = outlets disagree strongly on tone."
)

leaderboard_df = event_stats[
    ["cluster_name", "date_start", "date_end", "sources_covered", "total_articles", "polarity_spread"]
].copy()
leaderboard_df.columns = ["Event", "From", "To", "Sources", "Articles", "Spread"]
leaderboard_df["From"] = pd.to_datetime(leaderboard_df["From"]).dt.strftime("%Y-%m-%d")
leaderboard_df["To"] = pd.to_datetime(leaderboard_df["To"]).dt.strftime("%Y-%m-%d")
leaderboard_df["Spread"] = leaderboard_df["Spread"].round(2)

st.dataframe(leaderboard_df.head(20).reset_index(drop=True), use_container_width=True, height=320)

# ── Section 2: Source pairing matrix ─────────────────────────────────────────
st.markdown("### Source Pairing Matrix")
st.caption(
    "Average sentiment difference between each pair of sources across all shared events. "
    "Positive = row source is more positive than column source."
)

# Pivot: event × source — restrict to events that passed the spread filter
filtered_cluster_ids = event_stats["cluster_id"].tolist()
pivot = df[df["cluster_id"].isin(filtered_cluster_ids)].pivot_table(
    index="cluster_id", columns="source_name", values="avg_sentiment", aggfunc="mean"
)
pivot = pivot.dropna(thresh=2)  # only events with ≥2 sources

sources = pivot.columns.tolist()
if len(sources) < 2:
    st.info("Not enough sources covering shared events to build a pairing matrix.")
else:
    pair_matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)
    for s1 in sources:
        for s2 in sources:
            if s1 == s2:
                pair_matrix.loc[s1, s2] = 0.0
            else:
                shared = pivot[[s1, s2]].dropna()
                if not shared.empty:
                    pair_matrix.loc[s1, s2] = round((shared[s1] - shared[s2]).mean(), 3)

    fig_matrix = go.Figure(
        data=go.Heatmap(
            z=pair_matrix.values.astype(float),
            x=pair_matrix.columns.tolist(),
            y=pair_matrix.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            zmin=-3,
            zmax=3,
            colorbar=dict(title="Avg Δ sentiment"),
            text=pair_matrix.values.round(2).astype(str),
            texttemplate="%{text}",
        )
    )
    fig_matrix.update_layout(
        height=400,
        xaxis_title="Column source",
        yaxis_title="Row source",
    )
    st.plotly_chart(fig_matrix, use_container_width=True)
    st.caption(
        "Read as: *row source* tends to be **[value]** sentiment points more positive "
        "than *column source* on events they both cover."
    )

# ── Section 3: Event deep-dive ────────────────────────────────────────────────
st.divider()
st.markdown("### Event Deep-Dive")

event_options = {
    f"{row['cluster_name']}  (spread {row['polarity_spread']:.2f}, "
    f"{row['sources_covered']} sources, {row['total_articles']} articles)": row["cluster_id"]
    for _, row in event_stats.iterrows()
}

selected_label = st.selectbox("Select an event", options=list(event_options.keys()))
selected_cluster_id = event_options[selected_label]

# Per-source sentiment bar
event_sources = df[df["cluster_id"] == selected_cluster_id].copy()
event_sources = event_sources.sort_values("avg_sentiment")

fig_bars = go.Figure()
fig_bars.add_trace(
    go.Bar(
        x=event_sources["source_name"],
        y=event_sources["avg_sentiment"].round(3),
        error_y=dict(
            type="data",
            array=event_sources["stddev_sentiment"].fillna(0).round(3),
            visible=True,
        ),
        marker_color=[
            "#2ca02c" if v > 0.5 else "#d62728" if v < -0.5 else "#7f7f7f"
            for v in event_sources["avg_sentiment"]
        ],
        text=event_sources["avg_sentiment"].round(2),
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Avg sentiment: %{y:.2f}<br>"
            "Articles: " + event_sources["article_count"].astype(str) + "<extra></extra>"
        ),
    )
)
fig_bars.update_layout(
    yaxis_title="Average Sentiment (-5 to +5)",
    xaxis_title="News Source",
    height=380,
    yaxis_range=[-5.5, 5.5],
)
fig_bars.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
st.plotly_chart(fig_bars, use_container_width=True)

# Article list per source
st.markdown("#### Articles by source")
articles_raw = load_event_articles_with_sentiment(selected_cluster_id, selected_model)

if articles_raw:
    art_df = pd.DataFrame(articles_raw)
    art_df["source_name"] = art_df["source_id"].map(SOURCE_NAMES).fillna(art_df["source_id"])
    art_df["date_posted"] = pd.to_datetime(art_df["date_posted"]).dt.strftime("%Y-%m-%d")
    art_df["overall_sentiment"] = art_df["overall_sentiment"].round(2)

    # Colour label
    def _label(score):
        if score > 0.5:
            return "🟢 Positive"
        if score < -0.5:
            return "🔴 Negative"
        return "⚪ Neutral"

    art_df["tone"] = art_df["overall_sentiment"].apply(_label)

    for source in sorted(art_df["source_name"].unique()):
        src_df = art_df[art_df["source_name"] == source][
            ["title", "date_posted", "overall_sentiment", "tone", "url"]
        ].copy()
        src_df.columns = ["Title", "Date", "Score", "Tone", "URL"]
        with st.expander(f"**{source}** — {len(src_df)} article(s)", expanded=True):
            st.dataframe(
                src_df[["Title", "Date", "Score", "Tone"]],
                use_container_width=True,
                hide_index=True,
            )
