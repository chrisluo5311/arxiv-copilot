def build_system_prompt(tools_enabled: list) -> str:
    base_prompt = (
        "You have access to external tools to help you search for relevant papers and analyze their full content. "
        "When the user asks about a specific paper (e.g., by title or topic), follow these steps:\n"
    )

    # search_abstracts and download_arxiv_pdf are always available
    search_tools = ["`search_abstracts` (to find related arXiv papers)"]
    
    # web_search is optional
    if "web_search" in tools_enabled:
        search_tools.append("`web_search` (to find relevant information from the web)")
    
    base_prompt += (
        "If the user didn't provide a specific `arxiv_id` or asks about a specific topic or paper title, "
        f"you should first use {' or '.join(search_tools)} to find relevant information. "
        "If you find a relevant paper and need to read its full content, extract the `arxiv_id` and then call `download_arxiv_pdf`.\n\n"
        "However, if the user's question is general, about recent events, or not specific to a scientific paper, use `web_search` directly to answer. "
        "You do NOT always need to download a PDF if the search results are sufficient to answer the user's question."
    )

    return base_prompt.strip()
