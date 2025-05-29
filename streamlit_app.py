import os
import torch
import streamlit as st
from dotenv import load_dotenv
from custom_func import search_abstracts_function, download_arxiv_pdf_function
from scripts.rag_generate import load_and_chunk, retrieve_top_chunks, standalone_answer
from scripts.download_pdf import download_arxiv_pdf
from scripts.parse_pdf_llama import parse_pdf_with_llamaparse
from scripts.abstract_search import search_abstracts
from scripts.utils import build_system_prompt
from scripts.prompt_template import default_prompt
from streamlit_option_menu import option_menu
from openai import OpenAI
import json

# disable torch classes
torch.classes.__path__ = []
# Load environment variables
load_dotenv()
# absolute path to the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__)) # /Users/luojidong/程式/arxiv-copilot
all_function_tools = [search_abstracts_function, download_arxiv_pdf_function]

# Sidebar navigation
with st.sidebar:
    mode = option_menu(
        menu_title="ArXiv Copilot",
        options=["Fast Retrieval", "File Q&A", "Chatbot"],
        default_index=0,
    )

# Side bar
# 1. OpenAI api key
openai_api_key = st.sidebar.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
# 2. Model selection
model_selection = st.sidebar.selectbox(
    "Select Model",
    options=["gpt-3.5-turbo", "gpt-4o", "gpt-4.1-nano"],
)
# 3. Show system prompt
show_system_prompt = st.sidebar.checkbox("🔧 Show system prompt", value=False)
# 4. Custom system prompt (editable for debugging)
st.sidebar.subheader("🧠 System Prompt")
system_prompt_input = st.sidebar.text_area(
    label="✏️ Edit System Prompt",
    value=default_prompt,
    height=250,
    key="system_prompt_input"
)
# Optional button to reset chat with new prompt
if st.sidebar.button("🔄 Reset Chat with New Prompt"):
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt_input},
        {"role": "assistant", "content": "System prompt updated. How can I help you?"}
    ]
    st.session_state["show_toast"] = True
    st.rerun()
# Show toast if the flag is set, then clear it
if st.session_state.get("show_toast"):
    st.toast('Reset successfully!', icon='🎉')
    del st.session_state["show_toast"]
# 5. Show available tools
st.sidebar.subheader("🛠️ Available Tools")
use_abstracts = st.sidebar.checkbox("📖 Summarize PDF", value=True)
available_tools = []
if use_abstracts:
    available_tools.append("summarize_pdf")

# 6. Clear chat history
is_reset_chat = st.sidebar.button("🗑️ Reset Chat")
if is_reset_chat:
    st.session_state.pop("messages",None)
    st.rerun()

# ------------------- File Q&A Mode ------------------- #
if mode == "File Q&A":
    st.title("File Q&A (arXiv Copilot)")
    st.caption("Answer questions about arXiv papers")

    arxiv_id = st.text_input("Enter arXiv ID", value="0704.0001")
    question = st.text_input("Ask a question about the paper")
    top_k = st.slider("🔍 Top-k Chunks", min_value=1, max_value=10, value=5)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif question and arxiv_id:
        # Check if the PDF is already downloaded
        # If not, download it
        pdf_path = f"{current_dir}/pdfs/{arxiv_id}.pdf"
        if not os.path.exists(pdf_path):
            st.warning("Path not found. Creating Path for PDF...")
            os.makedirs("./pdfs", exist_ok=True)
            os.makedirs("./pdf_chunks", exist_ok=True)
        if not download_arxiv_pdf(arxiv_id):
            st.error("❌ Failed to download PDF from arXiv.")
            st.stop()
        try:
            with st.spinner("Parsing paper with LlamaParse..."):
                parse_pdf_with_llamaparse(arxiv_id)
            st.success("✅ PDF parsed successfully!")
        except Exception as e:
            st.error(f"❌ Parsing failed: {str(e)}")
            st.stop()
        try:
            chunks = load_and_chunk(arxiv_id)
            top_chunks = retrieve_top_chunks(question, chunks, top_k=top_k)
            answer = standalone_answer(question, top_chunks, model_selection, openai_api_key)

            with st.expander("📚 Top Retrieved Chunks"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)
                    st.markdown("---")

            st.write("### Answer")
            st.write(answer)
        except FileNotFoundError:
            st.error("Parsed file not found. Make sure the PDF has been parsed into /pdf_chunks/")

# ------------------- Chatbot Mode ------------------- #
elif mode == "Chatbot":
    st.title("Chatbot")
    st.caption("🚀 A general-purpose chatbot powered by OpenAI")

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    # Initialize system prompt and first message
    if "messages" not in st.session_state:
        tool_prompt = build_system_prompt(available_tools)
        combined_prompt = f"{tool_prompt}\n\n{system_prompt_input}"
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": combined_prompt
            },
            {
                "role": "assistant",
                "content": "How can I help you?"
            }
        ]
    # If show prompt => Display the system prompt & first message
    if show_system_prompt:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
    else:
        # just display the assistant message
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                st.chat_message(msg["role"]).write(msg["content"])

    # User input is assigned to the prompt variable
    if prompt := st.chat_input("Ask me anything about arXiv papers!"):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        # append user input to the message history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # display user message
        st.chat_message("user").write(prompt)

        with st.status("💬 Thinking...", expanded=True) as status:
            st.write("🧠 Sending query to GPT...")

            # generate response
            response = client.chat.completions.create(
                model=model_selection,  # model
                messages=st.session_state.messages, # history messages
                tools=all_function_tools,  # search abstract tools
                tool_choice="auto",  # auto decide whether to call the function
            )

            max_tool_calls = 5
            call_count = 0
            # chain of tool calls
            while response.choices[0].finish_reason == "tool_calls":
                if call_count >= max_tool_calls:
                    st.error("⚠️ Too many tool calls — possible infinite loop. Please refine your question.")
                    break
                call_count += 1
                tool_calls = response.choices[0].message.tool_calls
                print(f"🔧 Tool calls: {tool_calls}")
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    if function_name == "search_abstracts":
                        st.write(f"🛠️ Calling `{function_name}`...")
                        results = search_abstracts(**arguments)

                        # Format the top-k search results
                        top_k_search_result = "\n\n".join(
                            [f"**Title**: {r.get('title','N/A')}\n\n"
                            f"**ID**: `{r.get('id','N/A')}`\n\n"
                            f"**Authors**: {r.get('authors','N/A')}\n\n"
                            f"**Abstract**: {r.get('abstract','N/A')}"
                            for r in results]
                        )

                        st.write("📚 Tool results received. Re-querying GPT...")

                        # Log tool response
                        st.session_state.messages.append({
                            "role": "function",
                            "name": function_name,
                            "content": top_k_search_result
                        })

                    elif function_name == "download_arxiv_pdf":
                        st.write(f"🛠️ Calling `{function_name}`...")
                        arxiv_id = arguments["arxiv_id"]
                        if f"pdf_context_{arxiv_id}" in st.session_state:
                            st.write(f"🧠 Using cached parsed result for `{arxiv_id}`.")
                            # Log tool result from cache
                            chunk_context = st.session_state[f"pdf_context_{arxiv_id}"]
                            st.session_state.messages.append({
                                "role": "function",
                                "name": "download_arxiv_pdf",
                                "content": (
                                    f"📥 (Cached) Result from `download_pdf`: Relevant chunks from `{arxiv_id}`:\n\n"
                                    f"{chunk_context}"
                                )
                            })
                            continue
                        if not download_arxiv_pdf(**arguments):
                            st.error(f"❌ Failed to download PDF:{arxiv_id} from arXiv.")
                            st.stop()
                        st.write(f"📥 PDF:{arxiv_id} downloaded successfully!")
                        try:
                            st.write("⏳ Parsing paper with LlamaParse...")
                            parse_pdf_with_llamaparse(arxiv_id)
                            st.write(f"✅ PDF:{arxiv_id} parsed successfully!")

                            # Load and chunk the parsed text
                            chunks = load_and_chunk(arxiv_id)
                            if not chunks:
                                st.error(f"No chunks generated from parsed file for arXiv ID: {arxiv_id}")
                                st.stop()
                            top_chunks = retrieve_top_chunks(prompt, chunks)
                            # Log top chunks to session (tool result)
                            chunk_context = "\n\n".join(top_chunks)
                            st.session_state[f"pdf_context_{arxiv_id}"] = chunk_context
                            st.session_state.messages.append({
                                "role": "function",
                                "name": "download_arxiv_pdf",
                                "content": (
                                    f"📥 Result from `download_pdf`: Here are relevant chunks extracted from the paper `{arxiv_id}`:\n\n"
                                    f"{chunk_context}"
                                )
                            })
                        except FileNotFoundError:
                            st.error("Parsed file not found. Make sure the PDF has been parsed into /pdf_chunks/")

                    response = client.chat.completions.create(
                        model=model_selection,
                        messages=st.session_state.messages,
                        tools=all_function_tools,
                        tool_choice="auto"
                    )

            final_msg = response.choices[0].message.content
            # Append the assistant's response to the session state
            st.session_state.messages.append({"role": "assistant", "content": final_msg})
            # Display assistant message
            st.chat_message("assistant").write(final_msg)
            status.update(label="✅ Answer generated.", state="complete", expanded=True)

# ------------------- Fast Retrieval Mode ------------------- #
elif mode == "Fast Retrieval":
    st.title("Fast Retrieval")
    st.caption("Search for arXiv papers by abstract")

    query = st.text_input("Enter a topic or keyword to search for arXiv papers")
    top_k = st.slider("🔍 Top-k Results", min_value=1, max_value=10, value=5)

    # Fast retrieval using FAISS, no need AI models
    if query:
        results = search_abstracts(query, top_k=top_k)
        cnt = 1
        for r in results:
            st.markdown(f"### Result {cnt}:")
            st.markdown(f"**Title:** {r['title']}")
            st.markdown(f"**ID:** {r['id']}")
            st.markdown(f"**Authors:** {r['authors']}")
            st.markdown(f"**Abstract:** {r['abstract']}")
            st.markdown("---")
            cnt += 1