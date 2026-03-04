"""Events Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network

from data.loaders import load_similarity_edges, load_article_metadata
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style
from src.clustering import compute_clusters_from_edges

st.set_page_config(
    page_title="Sri Lanka Media Bias Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_page_style()

st.title("Event Clustering Analysis")

# Create version button at the top
render_create_version_button('clustering')

# Version selector
version_id = render_version_selector('clustering')

if not version_id:
    st.stop()

st.markdown("---")

# ── Runtime controls ──────────────────────────────────────────────────────────

col_thresh, col_window = st.columns(2)
with col_thresh:
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        help="Minimum cosine similarity for two articles to be linked. "
             "Only values ≥ the version's storage threshold are meaningful."
    )
with col_window:
    date_window = st.slider(
        "Date window (days)",
        min_value=1,
        max_value=30,
        value=3,
        help="Articles more than this many days apart cannot be in the same event cluster."
    )

# ── Load data ─────────────────────────────────────────────────────────────────

edges = load_similarity_edges(version_id, threshold, date_window)
meta = load_article_metadata(version_id)

if not meta:
    st.warning("No article embeddings found for this version. Run the pipeline first.")
    st.code(f"python3 scripts/clustering/02_cluster_events.py --version-id {version_id}")
    st.stop()

if not edges:
    st.warning(
        "No article pairs found at this threshold and date window. "
        "Try lowering the similarity threshold or widening the date window. "
        "If the pipeline hasn't been run yet: "
        f"`python3 scripts/clustering/02_cluster_events.py --version-id {version_id}`"
    )
    st.stop()

# Build article metadata lookup
meta_by_id = {str(r["id"]): r for r in meta}

# Compute clusters (connected components) from the filtered edges
components_list, G = compute_clusters_from_edges(edges, min_cluster_size=2)


def build_cluster_info(components_list, meta_by_id, G):
    """Build cluster metadata list from connected components."""
    clusters = []
    for comp in components_list:
        articles = [meta_by_id[aid] for aid in comp if aid in meta_by_id]
        if not articles:
            continue

        # Highest-degree node = representative (most connections within component)
        degrees = {aid: G.degree(aid) for aid in comp if aid in G}
        rep_id = max(degrees, key=degrees.get) if degrees else next(iter(comp))
        rep_title = meta_by_id.get(rep_id, {}).get("title", "Unnamed")

        sources = list({a["source_id"] for a in articles})
        dates = [a["date_posted"] for a in articles if a.get("date_posted")]

        clusters.append({
            "article_ids": list(comp),
            "articles": articles,
            "name": rep_title,
            "article_count": len(articles),
            "sources_count": len(sources),
            "sources": sources,
            "date_start": min(dates) if dates else None,
            "date_end": max(dates) if dates else None,
        })
    return clusters


cluster_info = build_cluster_info(components_list, meta_by_id, G)

st.caption(
    f"**{len(cluster_info)}** event clusters · "
    f"**{len(edges)}** article pairs · "
    f"threshold **{threshold}** · window **±{date_window}d**"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_explorer, tab_matrix, tab_network = st.tabs(["Event Explorer", "Coverage Matrix", "Network"])

# ── Tab 1: Event Explorer ─────────────────────────────────────────────────────

with tab_explorer:
    multi_source = [c for c in cluster_info if c["sources_count"] > 1]

    if not multi_source:
        st.warning("No multi-source event clusters at this threshold/date window.")
        st.stop()

    event_options = {
        f"{c['name'][:80]}… ({c['article_count']} articles, {c['sources_count']} sources)": i
        for i, c in enumerate(multi_source)
    }

    selected_label = st.selectbox("Select an event to explore", options=list(event_options.keys()))

    if selected_label:
        cluster = multi_source[event_options[selected_label]]
        articles_df = pd.DataFrame(cluster["articles"])
        articles_df["source_name"] = articles_df["source_id"].map(SOURCE_NAMES)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Coverage by Source**")
            source_counts = articles_df["source_name"].value_counts()
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                color=source_counts.index,
                color_discrete_map=SOURCE_COLORS,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Articles in this Event**")
            display_df = articles_df[["title", "source_name", "date_posted"]].copy()
            display_df.columns = ["Title", "Source", "Date"]
            st.dataframe(display_df, use_container_width=True, height=300)

# ── Tab 2: Coverage Matrix ────────────────────────────────────────────────────

with tab_matrix:
    st.caption(
        "Which outlets covered which events? "
        "Identify systematic gaps — events widely covered by others but absent from one outlet."
    )

    if not cluster_info:
        st.warning("No event clusters found at this threshold/date window.")
        st.stop()

    # Build long-format dataframe from cluster_info
    rows = []
    for ci, c in enumerate(cluster_info):
        for art in c["articles"]:
            rows.append({
                "cluster_idx": ci,
                "cluster_name": c["name"],
                "date_start": c["date_start"],
                "date_end": c["date_end"],
                "article_count": c["article_count"],
                "sources_count": c["sources_count"],
                "source_id": art["source_id"],
            })

    long_df = pd.DataFrame(rows)

    all_sources = [s for s in SOURCE_NAMES if s in long_df["source_id"].unique()]
    all_source_names = [SOURCE_NAMES[s] for s in all_sources]

    if not all_sources:
        st.error("No recognised sources found in clustering results.")
        st.stop()

    long_df["article_count_row"] = 1
    pivot = long_df.pivot_table(
        index="cluster_idx",
        columns="source_id",
        values="article_count_row",
        aggfunc="sum",
        fill_value=0,
    )
    pivot.columns.name = None
    for s in all_sources:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[all_sources]

    meta_cols = (
        long_df[["cluster_idx", "cluster_name", "date_start", "date_end", "sources_count", "article_count"]]
        .drop_duplicates("cluster_idx")
        .set_index("cluster_idx")
    )
    pivot = pivot.join(meta_cols).sort_values("date_start")

    min_other = st.slider(
        "Minimum outlets that must have covered an event for it to appear",
        min_value=1,
        max_value=max(1, len(all_sources) - 1),
        value=min(2, len(all_sources) - 1),
        help="Raises the bar for what counts as a 'newsworthy' event worth comparing across outlets.",
        key="matrix_min_other",
    )

    dynamic_source_count = (pivot[all_sources] > 0).sum(axis=1)
    matrix_df = pivot[dynamic_source_count >= min_other].copy()

    if matrix_df.empty:
        st.warning(f"No events covered by ≥{min_other} outlets. Try lowering the filter.")
        st.stop()

    st.caption(f"Showing {len(matrix_df)} events covered by ≥{min_other} outlets.")

    # Summary stats
    st.subheader("Coverage Summary")
    n_events = len(matrix_df)
    outlet_stats = []
    for src_id in all_sources:
        covered = int((matrix_df[src_id] > 0).sum())
        outlet_stats.append({
            "outlet": SOURCE_NAMES[src_id],
            "source_id": src_id,
            "covered": covered,
            "missed": n_events - covered,
            "rate": covered / n_events if n_events > 0 else 0,
        })
    stats_df = pd.DataFrame(outlet_stats).sort_values("rate", ascending=False)

    metric_cols = st.columns(len(all_sources) + 1)
    with metric_cols[0]:
        st.metric("Events", n_events)
    for i, row in enumerate(stats_df.itertuples()):
        with metric_cols[i + 1]:
            st.metric(row.outlet, f"{row.rate:.0%}", f"−{row.missed} missed", delta_color="inverse")

    fig_bar = px.bar(
        stats_df,
        x="outlet",
        y="rate",
        color="outlet",
        color_discrete_map=SOURCE_COLORS,
        text=stats_df["rate"].map(lambda x: f"{x:.0%}"),
        labels={"rate": "Coverage rate", "outlet": ""},
        title="Coverage rate per outlet",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_yaxes(range=[0, 1.15], tickformat=".0%")
    fig_bar.update_layout(showlegend=False, height=340)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    st.subheader("Coverage Matrix")

    def make_label(row):
        date = str(row["date_start"])[:10] if pd.notna(row["date_start"]) else "?"
        name = row["cluster_name"] or "Unnamed"
        if len(name) > 60:
            name = name[:57] + "…"
        return f"{date} | {name}"

    event_labels = matrix_df.apply(make_label, axis=1).tolist()
    presence = (matrix_df[all_sources] > 0).astype(int)
    counts = matrix_df[all_sources]
    heatmap_height = max(400, len(matrix_df) * 20)

    htab_binary, htab_counts = st.tabs(["Presence", "Article counts"])

    with htab_binary:
        fig_b = go.Figure(go.Heatmap(
            z=presence.values,
            x=all_source_names,
            y=event_labels,
            colorscale=[[0, "#f0f0f0"], [1, "#2196F3"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}: %{customdata} articles<extra></extra>",
            customdata=counts.values,
        ))
        fig_b.update_layout(
            height=heatmap_height,
            margin=dict(l=340, r=20, t=40, b=40),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_b, use_container_width=True)

    with htab_counts:
        fig_c = go.Figure(go.Heatmap(
            z=counts.values,
            x=all_source_names,
            y=event_labels,
            colorscale="Blues",
            hovertemplate="%{y}<br>%{x}: %{z} articles<extra></extra>",
        ))
        fig_c.update_layout(
            height=heatmap_height,
            margin=dict(l=340, r=20, t=40, b=40),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    # Missed events per outlet
    st.subheader("Missed Events by Outlet")
    st.caption(
        f"Events covered by ≥{min_other} other outlets but absent from the selected outlet. "
        "Sorted by how many other outlets covered the event — top rows are the strongest signals."
    )

    outlet_tabs = st.tabs(all_source_names)
    for src_id, src_name, otab in zip(all_sources, all_source_names, outlet_tabs):
        with otab:
            other_sources = [s for s in all_sources if s != src_id]
            other_coverage = (matrix_df[other_sources] > 0).sum(axis=1)
            missed_mask = (matrix_df[src_id] == 0) & (other_coverage >= min_other)
            missed = matrix_df[missed_mask].copy()

            if missed.empty:
                st.success(f"{src_name} covered all events in this matrix.")
                continue

            missed["_other_outlets"] = (missed[other_sources] > 0).sum(axis=1)
            missed = missed.sort_values(["_other_outlets", "article_count"], ascending=False)
            st.metric(f"Events missed by {src_name}", len(missed))

            rows_out = []
            for _, ev in missed.iterrows():
                covering = [f"{SOURCE_NAMES[s]} ({int(ev[s])})" for s in other_sources if ev[s] > 0]
                date_start = str(ev["date_start"])[:10] if pd.notna(ev["date_start"]) else "?"
                date_end = str(ev["date_end"])[:10] if pd.notna(ev["date_end"]) else "?"
                rows_out.append({
                    "Event": ev["cluster_name"] or "Unnamed",
                    "Dates": f"{date_start} – {date_end}",
                    "Covered by (articles)": ", ".join(covering),
                    "Outlets covering": int(ev["_other_outlets"]),
                })

            st.dataframe(
                pd.DataFrame(rows_out),
                use_container_width=True,
                hide_index=True,
                column_config={"Outlets covering": st.column_config.NumberColumn(format="%d")},
            )

# ── Tab 3: Network ────────────────────────────────────────────────────────────

with tab_network:
    st.caption(
        "Force-directed graph of article similarity. "
        "Node color = source outlet. Edge width ∝ similarity score. "
        "Click a node to highlight its connected cluster."
    )

    # Build source_id → hex color mapping
    source_id_to_color = {
        sid: SOURCE_COLORS.get(SOURCE_NAMES.get(sid, sid), "#888888")
        for sid in SOURCE_NAMES
    }

    # Collect node IDs that appear in at least one edge
    node_ids_in_edges = set()
    for edge in edges:
        node_ids_in_edges.add(str(edge["article_id_a"]))
        node_ids_in_edges.add(str(edge["article_id_b"]))

    if not node_ids_in_edges:
        st.warning("No edges to display at the current threshold and date window.")
    else:
        n_nodes = len(node_ids_in_edges)
        n_edges = len(edges)

        if n_edges > 5000:
            st.warning(
                f"Large graph: {n_nodes} nodes, {n_edges} edges. "
                "Rendering may be slow. Consider raising the similarity threshold."
            )

        # Search box — find a node by title
        title_to_id = {
            meta_by_id[nid].get("title", nid): nid
            for nid in node_ids_in_edges
            if nid in meta_by_id
        }
        search_options = [""] + sorted(title_to_id.keys())
        selected_title = st.selectbox(
            "Search for an article (highlights it in the graph)",
            options=search_options,
            format_func=lambda t: "— none —" if t == "" else t,
        )
        highlight_id = title_to_id.get(selected_title) if selected_title else None

        net = Network(height="720px", width="100%", notebook=True)
        net.set_options("""
        {
          "configure": { "enabled": true, "filter": "physics" },
          "layout": { "improvedLayout": false },
          "physics": {
            "stabilization": { "enabled": true, "iterations": 100, "fit": true },
            "barnesHut": { "gravitationalConstant": -8000, "springConstant": 0.04 },
            "minVelocity": 5
          }
        }
        """)

        for article_id in node_ids_in_edges:
            art = meta_by_id.get(article_id, {})
            source_id = art.get("source_id", "")
            source_name = SOURCE_NAMES.get(source_id, source_id)
            color = source_id_to_color.get(source_id, "#888888")
            date_str = str(art.get("date_posted", ""))[:10]
            title_text = art.get("title", article_id)
            tooltip = f"{title_text}\n{source_name}\n{date_str}"
            short_label = title_text[:50] + "…" if len(title_text) > 50 else title_text
            if highlight_id and article_id == highlight_id:
                node_color = {"background": "#FFD700", "border": "#FF8C00"}
                node_size = 40
            else:
                node_color = color
                node_size = 10
            net.add_node(article_id, label=short_label, title=tooltip, color=node_color, size=node_size)

        for edge in edges:
            id_a = str(edge["article_id_a"])
            id_b = str(edge["article_id_b"])
            score = float(edge["similarity_score"])
            net.add_edge(id_a, id_b, value=score, color="#cccccc")


        html = net.generate_html()

        if highlight_id:
            focus_js = f"""
            <script>
            document.addEventListener("DOMContentLoaded", function() {{
                var checkReady = setInterval(function() {{
                    if (typeof network !== "undefined") {{
                        clearInterval(checkReady);
                        network.on("stabilizationIterationsDone", function() {{
                            network.focus("{highlight_id}", {{
                                scale: 2.0,
                                animation: {{ duration: 800, easingFunction: "easeInOutQuad" }}
                            }});
                            network.selectNodes(["{highlight_id}"]);
                        }});
                    }}
                }}, 100);
            }});
            </script>
            """
            html = html.replace("</body>", focus_js + "</body>")

        components.html(html, height=720, scrolling=False)

        # Legend
        legend_cols = st.columns(len(SOURCE_NAMES))
        for i, (sid, sname) in enumerate(SOURCE_NAMES.items()):
            color = SOURCE_COLORS.get(sname, "#888888")
            with legend_cols[i]:
                st.markdown(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color};border-radius:50%;margin-right:4px"></span>{sname}',
                    unsafe_allow_html=True,
                )
