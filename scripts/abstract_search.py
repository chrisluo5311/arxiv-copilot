from AWS.search_knn import search_papers

def search_abstracts(query, top_k=3):
    """
    Search for abstracts based on a query
    Args:
        query: the query
        top_k: the number of top results
    Returns:
        results: the top results
    """
    results = search_papers(query, top_k=top_k)
    formatted = []
    for r in results:
        formatted.append({
            "id": r.get("paper_id"),
            "title": r.get("title"),
            "abstract": r.get("abstract"),
            "authors": ", ".join(r.get("authors", [])),
            "categories": " ".join(r.get("categories", [])),
            "year": r.get("year"),
        })
    return formatted
