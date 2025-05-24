# Define a function to search arXiv paper abstracts related to a query
search_abstracts_function = [
    {
        "type": "function",
        "function": {
            "name": "search_abstracts",
            "description": "Search arXiv paper abstracts related to a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to search related research papers"
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
]