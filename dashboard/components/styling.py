"""Shared styling for the dashboard."""

import streamlit as st

SIDEBAR_CSS = """
<style>
    [data-testid="stSidebar"] {
        min-width: 180px;
        max-width: 180px;
    }

    /* Align section headers with nav link items */
    [data-testid="stSidebarNavSectionHeader"] p {
        padding-left: 0.75rem;
    }
</style>
"""


def apply_page_style():
    """Apply shared CSS styling. Call at the top of every page."""
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
