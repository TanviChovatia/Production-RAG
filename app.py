from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chat_store import append_message, create_chat, delete_chat, list_chats, load_chat
from config import settings
from document_loader import load_document
from rag_chain import build_rag_system

load_dotenv()

st.set_page_config(page_title=settings.app_name, layout="wide")
st.title("Production RAG Application")
st.caption("Ask short questions about your uploaded PDF")


# -------------------------
# Session state
# -------------------------
if "rag" not in st.session_state:
    st.session_state.rag = None

if "active_doc_name" not in st.session_state:
    st.session_state.active_doc_name = None

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def load_selected_chat(chat_id: str) -> None:
    chat_data = load_chat(chat_id)
    st.session_state.chat_id = chat_id
    st.session_state.messages = chat_data.get("messages", [])


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    st.divider()
    st.header("Chat History")

    if st.button("➕ Start New Chat", use_container_width=True):
        new_chat_id = create_chat(document_name=st.session_state.active_doc_name)
        load_selected_chat(new_chat_id)
        st.rerun()

    chats = list_chats()
    if chats:
        for chat in chats:
            title = chat.get("title", "Untitled Chat")

            col1, col2 = st.columns([5, 1])

            with col1:
                if st.button(title, key=f"open_{chat['chat_id']}", use_container_width=True):
                    load_selected_chat(chat["chat_id"])
                    st.rerun()

            with col2:
                if st.button("✕", key=f"delete_{chat['chat_id']}", use_container_width=True):
                    delete_chat(chat["chat_id"])

                    if st.session_state.chat_id == chat["chat_id"]:
                        st.session_state.chat_id = None
                        st.session_state.messages = []

                    st.rerun()


# -------------------------
# Load PDF safely
# -------------------------
if uploaded_file is not None and uploaded_file.name != st.session_state.active_doc_name:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getbuffer())

        with st.spinner("Loading, chunking, indexing, and preparing retrieval..."):
            docs = load_document(tmp_path)
            collection_name = uploaded_file.name.replace(".", "_").replace(" ", "_")

            # IMPORTANT:
            # Do NOT delete the Chroma directory here.
            # On Windows, chroma.sqlite3 may still be locked by a previous process.
            st.session_state.rag = None
            st.session_state.rag = build_rag_system(docs, collection_name=collection_name)
            st.session_state.active_doc_name = uploaded_file.name

            if st.session_state.chat_id is None:
                st.session_state.chat_id = create_chat(document_name=uploaded_file.name)
                st.session_state.messages = []

    st.success(f"Ready: {uploaded_file.name}")


# -------------------------
# Current PDF info
# -------------------------
if st.session_state.active_doc_name:
    st.info(f"Current PDF: {st.session_state.active_doc_name}")


# -------------------------
# Render chat history
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("metrics_text"):
                st.caption(msg["metrics_text"])

            if msg.get("sources"):
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.markdown(source["title"])
                        st.write(source["snippet"])

            if msg.get("metadata"):
                with st.expander("Trace / metadata"):
                    st.json(msg["metadata"])


# -------------------------
# Chat input
# -------------------------
question = st.chat_input("Ask a question about the uploaded PDF")

if question:
    if st.session_state.rag is None:
        st.warning("Upload a PDF first.")
    else:
        if st.session_state.chat_id is None:
            st.session_state.chat_id = create_chat(document_name=st.session_state.active_doc_name)

        user_msg = {"role": "user", "content": question}
        st.session_state.messages.append(user_msg)
        append_message(st.session_state.chat_id, "user", question)

        with st.chat_message("user"):
            st.markdown(question)

        final_question = (
            f"{question}\n\n"
            "Instruction: Answer briefly in 3-4 lines only. "
            "Keep it short, clear, and grounded in the sources."
        )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving sources and generating answer..."):
                response = st.session_state.rag.ask(final_question, session_id=st.session_state.chat_id)

            st.markdown(response.answer)

            metrics_text = (
                f"latency={response.metadata['latency_seconds']}s | "
                f"retrieval={response.metadata['retrieval_latency_seconds']}s | "
                f"generation={response.metadata['generation_latency_seconds']}s"
            )
            st.caption(metrics_text)

            source_blocks = []
            with st.expander("Sources"):
                for idx, doc in enumerate(response.source_documents, start=1):
                    page = doc.metadata.get("page", "?")
                    if isinstance(page, int):
                        page += 1

                    title = f"**S{idx}** - `{doc.metadata.get('source')}` - page {page}"
                    snippet = doc.page_content[:700] + ("..." if len(doc.page_content) > 700 else "")

                    st.markdown(title)
                    st.write(snippet)

                    source_blocks.append(
                        {
                            "title": title,
                            "snippet": snippet,
                        }
                    )

            with st.expander("Trace / metadata"):
                st.json(response.metadata)

        assistant_msg = {
            "role": "assistant",
            "content": response.answer,
            "metrics_text": metrics_text,
            "sources": source_blocks,
            "metadata": response.metadata,
        }

        st.session_state.messages.append(assistant_msg)
        append_message(
            st.session_state.chat_id,
            "assistant",
            response.answer,
            metrics_text=metrics_text,
            sources=source_blocks,
            metadata=response.metadata,
        )