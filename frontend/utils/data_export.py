"""
Data export utilities
"""

import pandas as pd
import io

def to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to CSV bytes.

    Args:
        df (pd.DataFrame): Data to convert.

    Returns:
        bytes: CSV data in bytes for download.
    """
    return df.to_csv(index=False).encode('utf-8')


def to_excel(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to Excel bytes.

    Args:
        df (pd.DataFrame): Data to convert.

    Returns:
        bytes: Excel data in bytes for download.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()
