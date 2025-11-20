from ddgs import DDGS
from typing import List, Dict


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information related to the query using DuckDuckGo
    Args:
        query: the search query
        max_results: maximum number of results to return (default: 5)
    Returns:
        results: list of dictionaries containing title, url, and snippet for each result
    """
    try:
        with DDGS() as ddgs:
            results = []
            # Search for text results
            for r in ddgs.text(query, region="us-en", max_results=max_results):
                results.append({
                    "title": r.get("title", "N/A"),
                    "url": r.get("href", "N/A"),
                    "snippet": r.get("body", "N/A")
                })
            return results
    except Exception as e:
        # Return error message if search fails
        return [{
            "title": "Search Error",
            "url": "N/A",
            "snippet": f"Failed to perform web search: {str(e)}"
        }]