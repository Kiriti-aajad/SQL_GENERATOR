"""
Result table component placeholder
"""

import streamlit as st
import pandas as pd

def result_table(df: pd.DataFrame):
    """
    Display a pandas DataFrame in a scrollable Streamlit table.

    Args:
        df (pd.DataFrame): The query result data to display.
    """
    if df.empty:
        st.info("No results to display.")
        return

    st.dataframe(df, use_container_width=True)
