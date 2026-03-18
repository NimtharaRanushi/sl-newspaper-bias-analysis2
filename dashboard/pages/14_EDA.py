"""Exploratory Data Analysis — sentiment density plots and gamma-weighted topic sentiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

from data.loaders import load_available_models, load_sentiment_density, load_weighted_topic_sentiment
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style
from components.version_selector import render_version_selector
from src.versions import list_versions

st.set_page_config(
    page_title="EDA - Sri Lanka Media Bias Detector",
    page_icon="📈",
    layout="wide",
)

apply_page_style()

st.title("Exploratory Data Analysis")
st.caption("Sentiment score distributions per newspaper outlet, Ditwah Cyclone coverage (Nov 22 – Dec 31 2025).")

# ── Shared controls ─────────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "roberta": "RoBERTa",
    "distilbert": "DistilBERT",
    "finbert": "FinBERT",
    "vader": "VADER",
    "textblob": "TextBlob",
    "sentimentr": "sentimentr (R)",
    "local": "Local (RoBERTa)",
}

available_models = load_available_models()
if not available_models:
    st.warning("No sentiment data found. Run a sentiment analysis script first.")
    st.stop()

model_list = [m["model_type"] for m in available_models]
default_idx = model_list.index("sentimentr") if "sentimentr" in model_list else 0

col_model, col_score_type = st.columns([1, 1])

with col_model:
    selected_model = st.selectbox(
        "Sentiment model",
        options=model_list,
        index=default_idx,
        format_func=lambda x: MODEL_DISPLAY.get(x, x.upper()),
    )

with col_score_type:
    score_type = st.radio(
        "Score type",
        options=["Overall Article", "Headline Only"],
        horizontal=True,
    )

score_col = "overall_sentiment" if score_type == "Overall Article" else "headline_sentiment"
weighted_col = "weighted_overall" if score_type == "Overall Article" else "weighted_headline"
avg_col = "avg_overall" if score_type == "Overall Article" else "avg_headline"

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Outlet sentiment density
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Outlet Sentiment Distribution")

rows = load_sentiment_density(selected_model)
if not rows:
    st.warning(f"No data found for model '{selected_model}'. Run the sentiment pipeline first.")
else:
    df = pd.DataFrame(rows, columns=["source_id", "overall_sentiment", "headline_sentiment"])
    df["source_name"] = df["source_id"].map(SOURCE_NAMES).fillna(df["source_id"])

    x_range = np.linspace(-5, 5, 500)
    fig = go.Figure()

    for source_name, group in df.groupby("source_name"):
        scores = group[score_col].dropna().values
        if len(scores) < 10:
            continue
        kde = gaussian_kde(scores, bw_method=0.4)
        y_vals = kde(x_range)
        hex_color = SOURCE_COLORS.get(source_name, "#888888")
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_vals, name=source_name,
            mode="lines", fill="tozeroy",
            line=dict(color=hex_color, width=2),
            fillcolor=f"rgba({r},{g},{b},0.12)",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="Neutral", annotation_position="top")
    fig.update_layout(
        title=f"Sentiment Density by Outlet ({MODEL_DISPLAY.get(selected_model, selected_model)} — {score_type})",
        xaxis_title="Sentiment Score", yaxis_title="Density",
        xaxis=dict(range=[-5, 5]), legend=dict(title="Outlet"),
        height=460, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats table
    stats_rows = []
    for source_name, group in df.groupby("source_name"):
        scores = group[score_col].dropna()
        if len(scores) == 0:
            continue
        stats_rows.append({
            "Outlet": source_name, "N": len(scores),
            "Mean": round(scores.mean(), 3), "Median": round(scores.median(), 3),
            "Std Dev": round(scores.std(), 3), "Min": round(scores.min(), 3),
            "Max": round(scores.max(), 3),
        })
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows).set_index("Outlet"), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Gamma-Weighted Topic Sentiment
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Gamma-Weighted Topic Sentiment")

with st.expander("About this method", expanded=False):
    st.markdown("""
    **Gamma-weighting** uses each document's *topic assignment confidence* (γ) as a weight
    when computing the sentiment score for a topic.

    Since there is one sentiment score per document but potentially uncertain topic membership,
    a document is weighted by how strongly it belongs to a topic:

    > **Topic sentiment** = Σ (sentiment_i × γ_i) / Σ γ_i

    A document with γ = 0.99 (strongly assigned) contributes almost its full sentiment to
    the topic. A document with γ = 0.01 (weakly assigned) has minimal impact. This produces
    a more accurate measure of *tonality* towards each topic than a simple unweighted average.

    *Reference: "The Media Bias Detector: A Framework for Annotating and Analyzing the
    News at Scale" (UPenn, 2025).*
    """)

# Topic version selector + label style
topic_versions = list_versions(analysis_type="topics")
if not topic_versions:
    st.info("No topic versions found. Run a topic modeling script first.")
else:
    version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v["id"]
        for v in topic_versions
        if v.get("pipeline_status", {}).get("topics")
    }

    if not version_options:
        st.info("No completed topic versions found. Run a topic pipeline script first.")
    else:
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            selected_label = st.selectbox(
                "Topic version",
                options=list(version_options.keys()),
                key="eda_topic_version",
            )
        with ctrl_col2:
            label_style = st.radio(
                "Topic label",
                options=["Multi-word", "One-word"],
                horizontal=True,
                key="eda_label_style",
            )
        selected_version_id = version_options[selected_label]

        topic_rows = load_weighted_topic_sentiment(selected_model, selected_version_id)

        if not topic_rows:
            st.warning("No articles found with both topic assignments and sentiment scores for this combination.")
        else:
            df_topics = pd.DataFrame(topic_rows, columns=[
                "topic_id", "topic_name",
                "weighted_overall", "weighted_headline",
                "avg_overall", "avg_headline",
                "stddev_overall", "min_sentiment", "max_sentiment",
                "n_articles",
            ])

            # Apply label style — one-word takes only the first keyword
            if label_style == "One-word":
                df_topics["display_name"] = (
                    df_topics["topic_name"]
                    .str.split(",").str[0]
                    .str.strip()
                    .str.title()
                )
            else:
                df_topics["display_name"] = df_topics["topic_name"]

            # Use the selected score column
            df_topics["weighted_sentiment"] = df_topics[weighted_col].astype(float)
            df_topics["avg_sentiment"] = df_topics[avg_col].astype(float)
            df_topics = df_topics.sort_values("weighted_sentiment")

            # ── Bar chart ──────────────────────────────────────────────────────

            def _sentiment_color(val):
                if val < -0.3:
                    return "#d62728"   # red
                elif val > 0.3:
                    return "#2ca02c"   # green
                else:
                    return "#7f7f7f"   # grey

            bar_colors = [_sentiment_color(v) for v in df_topics["weighted_sentiment"]]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=df_topics["weighted_sentiment"],
                y=df_topics["display_name"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.3f}" for v in df_topics["weighted_sentiment"]],
                textposition="outside",
                customdata=np.stack([
                    df_topics["n_articles"],
                    df_topics["avg_sentiment"],
                    df_topics["topic_name"],          # full name always in hover
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    "Weighted Sentiment: %{x:.3f}<br>"
                    "Unweighted Avg: %{customdata[1]:.3f}<br>"
                    "Articles: %{customdata[0]}<extra></extra>"
                ),
            ))

            fig_bar.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_bar.update_layout(
                title=f"Gamma-Weighted Topic Sentiment — {label_style} labels "
                      f"({MODEL_DISPLAY.get(selected_model, selected_model)} — {score_type})",
                xaxis_title="Weighted Sentiment Score",
                yaxis_title=None,
                height=max(400, len(df_topics) * 28 + 100),
                margin=dict(l=10, r=80, t=50, b=40),
                xaxis=dict(range=[
                    min(-0.5, df_topics["weighted_sentiment"].min() - 0.1),
                    max(0.5, df_topics["weighted_sentiment"].max() + 0.2),
                ]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Descriptive statistics table ───────────────────────────────────

            st.subheader("Topic Sentiment — Descriptive Statistics")

            display_df = df_topics[[
                "display_name", "topic_name", "n_articles",
                "weighted_sentiment", "avg_sentiment",
                "stddev_overall", "min_sentiment", "max_sentiment",
            ]].rename(columns={
                "display_name": "Topic",
                "topic_name": "Full Label",
                "n_articles": "N Articles",
                "weighted_sentiment": "Weighted Mean (γ)",
                "avg_sentiment": "Unweighted Mean",
                "stddev_overall": "Std Dev",
                "min_sentiment": "Min",
                "max_sentiment": "Max",
            }).sort_values("Weighted Mean (γ)")

            # Hide "Full Label" column when already in multi-word mode
            cols_to_show = display_df.columns.tolist()
            if label_style == "Multi-word":
                cols_to_show = [c for c in cols_to_show if c != "Full Label"]
            display_df = display_df[cols_to_show]

            for col in ["Weighted Mean (γ)", "Unweighted Mean", "Std Dev", "Min", "Max"]:
                display_df[col] = display_df[col].round(3)

            st.dataframe(
                display_df.set_index("Topic"),
                use_container_width=True,
                column_config={
                    "Weighted Mean (γ)": st.column_config.ProgressColumn(
                        "Weighted Mean (γ)",
                        help="Gamma-weighted average sentiment (−5 to +5)",
                        format="%.3f",
                        min_value=-5,
                        max_value=5,
                    ),
                },
            )

            # ── Scatter: weighted vs unweighted ────────────────────────────────

            with st.expander("Weighted vs Unweighted Sentiment Comparison"):
                fig_scatter = px.scatter(
                    df_topics,
                    x="avg_sentiment",
                    y="weighted_sentiment",
                    text="display_name",
                    size="n_articles",
                    color="weighted_sentiment",
                    color_continuous_scale="RdYlGn",
                    labels={
                        "avg_sentiment": "Unweighted Mean",
                        "weighted_sentiment": "Gamma-Weighted Mean",
                        "n_articles": "Articles",
                    },
                    title="Impact of Gamma-Weighting on Topic Sentiment",
                )
                fig_scatter.update_traces(textposition="top center", textfont_size=10)
                # Diagonal reference line y=x
                lim = max(
                    abs(df_topics["avg_sentiment"].max()),
                    abs(df_topics["weighted_sentiment"].max()),
                    0.5,
                ) + 0.1
                fig_scatter.add_shape(
                    type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                    line=dict(dash="dash", color="gray", width=1),
                )
                fig_scatter.update_layout(height=520, coloraxis_showscale=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption(
                    "Points above the diagonal: gamma-weighting increased the topic's polarity "
                    "(highly confident documents skew sentiment). Points below: weighting reduced it."
                )
