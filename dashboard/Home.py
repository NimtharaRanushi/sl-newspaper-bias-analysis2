"""Sri Lanka Media Bias Dashboard - Home Page."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from data.loaders import load_overview_stats
from components.source_mapping import SOURCE_NAMES
from components.styling import apply_page_style


st.set_page_config(
    page_title="Sri Lanka Media Bias Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def show_home():
    apply_page_style()

    st.title("Sri Lanka Media Bias Detector")
    st.markdown("Analyzing coverage patterns across Sri Lankan English newspapers")

    stats = load_overview_stats()

    st.header("Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Articles", f"{stats['total_articles']:,}")
    with col2:
        st.metric("Ditwah Cyclone Articles", f"{stats['ditwah_articles']:,}")
    with col3:
        if stats['date_range']['min_date']:
            st.caption(f"**Date range:** {stats['date_range']['min_date']} to {stats['date_range']['max_date']}")

    st.divider()

    # Show sources summary
    st.subheader("Sources Analyzed")
    cols = st.columns(4)
    for idx, (source_id, source_name) in enumerate(SOURCE_NAMES.items()):
        source_count = next(
            (s['count'] for s in stats['by_source'] if s['source_id'] == source_id),
            0
        )
        ditwah_count = next(
            (s['count'] for s in stats['ditwah_by_source'] if s['source_id'] == source_id),
            0
        )
        with cols[idx % 4]:
            st.metric(source_name, f"{source_count:,} articles")
            st.caption(f"Cyclone Ditwah: {ditwah_count:,} articles")


pg = st.navigation({
    "": [
        st.Page(show_home, title="Home", default=True),
        st.Page("pages/10_Article_Insights.py", title="Article Insights"),
        st.Page("pages/12_Chatbot.py", title="Chatbot"),
    ],
    "Deep Dive": [
        st.Page("pages/deep_dive_ditwah.py", title="Ditwah Cyclone"),
    ],
    "Analysis": [
        st.Page("pages/1_Coverage.py", title="Coverage"),
        st.Page("pages/2_Topics.py", title="Topics"),
        st.Page("pages/3_Events.py", title="Events"),
        st.Page("pages/4_Sentiment.py", title="Sentiment"),
        st.Page("pages/5_Summaries.py", title="Summaries"),
        st.Page("pages/6_Word_Frequency.py", title="Word Frequency"),
        st.Page("pages/7_Named_Entities.py", title="Named Entities"),
        st.Page("pages/8_Ditwah_Claims.py", title="Ditwah Claims"),
        st.Page("pages/9_Stance.py", title="Stance"),
        st.Page("pages/11_Entity_Stance.py", title="Entity Stance"),
        st.Page("pages/13_Chunk_Topics.py", title="Chunk Topics"),
    ],
})
pg.run()
