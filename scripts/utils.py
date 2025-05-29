def build_system_prompt(tools_enabled: list) -> str:
    base_prompt = (
        "You have access to external tools to help you search for relevant papers and analyze their full content. "
        "When the user asks about a specific paper (e.g., by title or topic), follow these steps:\n"
    )

    if "download_arxiv_pdf" in tools_enabled:
        base_prompt += (
            "If the user didn't provide a specific `arxiv_id`"
            "You should first use the `search_abstracts` function to find related arXiv papers. "
            "You can extract the correct `arxiv_id` from those results and then call the `download_arxiv_pdf` function to retrieve "
            "and analyze the full paper.\n\n"
            "Use `search_abstracts` to find IDs. Then use `download_arxiv_pdf` to retrieve context."
        )

    return base_prompt.strip()
