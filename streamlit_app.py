import os
import torch
import streamlit as st
from dotenv import load_dotenv
from custom_func import search_abstracts_function, download_arxiv_pdf_function
from scripts.rag_generate import load_and_chunk, retrieve_top_chunks, standalone_answer
from scripts.download_pdf import download_arxiv_pdf
from scripts.parse_pdf_llama import parse_pdf_with_llamaparse, parse_file_with_llamaparse
from scripts.abstract_search import search_abstracts
from scripts.utils import build_system_prompt
from scripts.prompt_template import default_prompt
from streamlit_option_menu import option_menu
from openai import OpenAI
import json
import base64
from PIL import Image

# ------------------ Configuration ------------------ #
# disable torch classes
torch.classes.__path__ = []
# Load environment variables
load_dotenv()
# absolute path to the directory where the script is located
# e.g., current_dir: "/Users/luojidong/程式/arxiv-copilot"
current_dir = os.path.dirname(os.path.abspath(__file__))
all_function_tools = [search_abstracts_function, download_arxiv_pdf_function]

# Define the list of supported file types
image_exts = ["jpg", "jpeg", "png"]
text_exts = ["pdf", "csv", "txt"]
uploaded_file_save_path = os.path.join(current_dir, "scripts", "user_upload")

# ------------------ Helper function ------------------ #
def resize_image(image_path, max_dim=224):
    img = Image.open(image_path)
    img.thumbnail((max_dim, max_dim))
    img.save(image_path)

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return ""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

def process_uploaded_file(files, query):
    """
    Process the uploaded file and return its content.
    """
    for uploaded_file in files:
        filename = uploaded_file.name
        ext = filename.split(".")[-1].lower()
        full_filename = uploaded_file_save_path+"/"+filename

        with open(full_filename, "wb") as f:
            f.write(uploaded_file.read())

        if ext in ["jpg", "jpeg", "png"]:
            resize_image(full_filename)
            base64_img = encode_image_to_base64(full_filename)
            return base64_img, ext
        elif ext in ["pdf", "csv", "txt"]:
            parse_file_with_llamaparse(filename)
            upload_file_chunks = load_and_chunk(filename)
            top_uploaded_chunks =retrieve_top_chunks(query, upload_file_chunks, 3)
            return top_uploaded_chunks, None
    return "", None


# ------------------ Sidebar navigation ------------------ #
with st.sidebar:
    mode = option_menu(
        menu_title="ArXiv Copilot",
        options=["Fast Retrieval", "File Q&A", "Chatbot"],
        default_index=0,
    )

# ------------------ Sidebar ------------------ #
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
use_abstracts = st.sidebar.checkbox("📖 Download PDF", value=True)
available_tools = []
if use_abstracts:
    available_tools.append("download_arxiv_pdf")

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
            answer = standalone_answer(question, top_chunks, model_selection, None, openai_api_key)

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
        combined_prompt = f"{system_prompt_input}\n\n{tool_prompt}"
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
    if show_system_prompt:
        # Show everything in the chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
    else:
        for msg in st.session_state.messages:
            # Skip system, developer, and function messages
            if msg["role"] != "system" and msg["role"] != "developer" and msg["role"] != "function":
                st.chat_message(msg["role"]).write(msg["content"])

    # User input is assigned to the prompt variable
    if prompt := st.chat_input("Ask me anything about arXiv papers!",
                                accept_file=True,
                                file_type=["jpg", "jpeg", "png", "pdf", "csv", "txt"]):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # ------ Process user input and uploaded files ------ #
        user_text = prompt.get("text", "")
        files = prompt.get("files", [])
        uploaded_file_content = ""
        file_ext = None
        if files:
            uploaded_file_content, file_ext = process_uploaded_file(files, user_text)

        # If image is uploaded, clear older messages
        if file_ext in image_exts:
            st.session_state.messages = st.session_state.messages[:2]  # Keep only system and greeting

        # append user input to the message history
        if isinstance(uploaded_file_content, str) and uploaded_file_content.startswith("data:image/"):
            # This is a base64 image already in data URI format
            st.session_state.messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text or "Analyze this image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/{file_ext};base64,{uploaded_file_content}"}}
                ]
            })
        else:
            # This is textual context (e.g., chunked text from PDF/txt/csv)
            combined_text = f"{user_text}\n\n{uploaded_file_content}" if uploaded_file_content else user_text
            st.session_state.messages.append({"role": "user", "content": combined_text})

        # display user message
        st.chat_message("user").write(user_text)

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

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
                # ------------------ Tool calls ------------------ #
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
                            top_chunks = retrieve_top_chunks(user_text, chunks)
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