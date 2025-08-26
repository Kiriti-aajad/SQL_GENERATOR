"""
SQL Editor component placeholder
"""

import streamlit as st

def sql_editor(sql_text: str) -> str:
    """
    Display an editable SQL code block and return the updated SQL.

    Args:
        sql_text (str): The initial SQL code to display.

    Returns:
        str: The edited SQL code entered by the user.
    """
    edited_sql = st.text_area("Edit SQL", value=sql_text, height=250, key="sql_editor")
    return edited_sql
