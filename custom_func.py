# Define a function to search arXiv paper abstracts related to a query
search_abstracts_function = {
    "type": "function",
    "function": {
        "name": "search_abstracts",
        "description": "Search the local arXiv database for academic paper abstracts. Use this ONLY when looking for scientific papers, research, or academic literature. Do NOT use this for general knowledge or current events.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The keyword to search in the arXiv database."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of top results to return",
                    "default": 3
                }
            },
            "required": ["query"]

        }
    }
}
download_arxiv_pdf_function = {
    "type": "function",
    "function" : {
        "name": "download_arxiv_pdf",
        "description": "Download the full PDF text of a paper from arXiv. Use this ONLY when you have a specific arXiv ID and need to read the full content of that paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "The arXiv paper ID (e.g., '2310.12345')"}
            },
            "required": ["arxiv_id"]
        }
    }
}

web_search_function = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public internet for general information, news, recent events, or broad topics not limited to academic papers. Use this when 'search_abstracts' is not appropriate or yields no results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for the web."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of search results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
