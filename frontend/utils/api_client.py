"""
API client utils for backend communication
"""

import requests

def call_generate_sql(api_url: str, payload: dict, token: str = None) -> dict:
    """
    Call the FastAPI /generate-sql endpoint with the given payload.

    Args:
        api_url (str): Base URL of the FastAPI backend (e.g. http://localhost:8000)
        payload (dict): JSON payload to send (query, options, etc.)
        token (str, optional): Authorization token if required.

    Returns:
        dict: JSON response from the backend parsed into a Python dictionary.

    Raises:
        requests.HTTPError: if the HTTP request returned an unsuccessful status code.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    url = f"{api_url.rstrip('/')}/generate-sql"
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()
