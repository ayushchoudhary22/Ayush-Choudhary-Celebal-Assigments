import os
import shutil
import streamlit as st
from vectorstore import SimpleVectorStore, chunk_document, get_fallback_embedding
from chatbot import get_embedding, generate_llm_answer

# Page Config
st.set_page_config(
    page_title="DocuMind AI - Streamlit RAG System",
    page_icon="🤖",
    layout="wide"
)

# App directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORE_PATH = os.path.join(BASE_DIR, "vector_store.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize vector store
@st.cache_resource
def get_vector_store():
    return SimpleVectorStore(storage_path=STORE_PATH)

store = get_vector_store()

# Sidebar Configuration
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8653/8653200.png", width=80)
st.sidebar.title("DocuMind Settings")

# 1. API Credentials
st.sidebar.subheader("🔑 Credentials")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="If empty, local Demo Mode will match context offline.")

# 2. Engine Settings
st.sidebar.subheader("⚙️ RAG Configurations")
top_k = st.sidebar.slider("Retrieve Context (Top K)", min_value=1, max_value=10, value=4)
chunk_size = st.sidebar.number_input("Chunk Size (characters)", min_value=200, max_value=2000, value=600, step=50)
chunk_overlap = st.sidebar.number_input("Chunk Overlap (characters)", min_value=0, max_value=500, value=80, step=10)

# Load existing documents info
def list_local_files():
    if not os.path.exists(DATA_DIR):
        return []
    files = []
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(('.pdf', '.txt')):
            path = os.path.join(DATA_DIR, f)
            files.append({
                "filename": f,
                "size_kb": os.path.getsize(path) / 1024
            })
    return files

local_files = list_local_files()
stats = store.get_indexed_documents() if hasattr(store, 'get_indexed_documents') else {"total_chunks": len(store.chunks), "documents": []}
indexed_filenames = [doc["filename"] for doc in stats.get("documents", [])]

# Stats Footer in Sidebar
st.sidebar.subheader("📊 Database Stats")
st.sidebar.write(f"Total Chunks Indexed: `{len(store.chunks)}`")
st.sidebar.write(f"Indexed Documents: `{len(indexed_filenames)}`")

# Header
st.title("DocuMind AI: Document Question Answering 🤖📄")
st.markdown("This Retrieval-Augmented Generation (RAG) chatbot extracts knowledge from your PDFs, indexes them, and answers your queries with citations.")

# Tabs
tab_chat, tab_docs, tab_lab = st.tabs(["💬 Chat Assistant", "📂 Manage Documents", "🔬 Retrieval Inspector"])

# Tab 1: Chat Assistant
with tab_chat:
    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! I am ready to answer questions about your indexed documents. Type a question below."}
        ]

    # Clear chat helper
    if st.button("Clear Conversation History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! History cleared. How can I help you today?"}
        ]

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chunks" in msg and msg["chunks"]:
                with st.expander("📚 View Citations"):
                    for i, chunk in enumerate(msg["chunks"]):
                        st.markdown(f"**[{i+1}] {chunk['source']} (Page {chunk['page']})**")
                        st.caption(f"Similarity Score: {chunk['similarity_score']:.2f}")
                        st.info(chunk["text"])

    # User query input
    if prompt := st.chat_input("Ask a question about your indexed files..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process response
        if not store.chunks:
            with st.chat_message("assistant"):
                st.warning("No documents have been indexed yet. Please go to the **Manage Documents** tab to index files.")
            st.session_state.messages.append({"role": "assistant", "content": "Warning: No documents indexed."})
        else:
            with st.spinner("Searching document index and synthesizing response..."):
                try:
                    # 1. Generate query embedding
                    q_emb = get_embedding(prompt, api_key)
                    # 2. Similarity search (using hybrid search)
                    matching_chunks = store.hybrid_search(prompt, q_emb, top_k=top_k)
                    # 3. Generate answer
                    answer = generate_llm_answer(prompt, matching_chunks, api_key)
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if matching_chunks:
                            with st.expander("📚 View Citations"):
                                for i, chunk in enumerate(matching_chunks):
                                    st.markdown(f"**[{i+1}] {chunk['source']} (Page {chunk['page']})**")
                                    st.caption(f"Similarity Score: {chunk['similarity_score']:.2f}")
                                    st.info(chunk["text"])
                    
                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "chunks": matching_chunks
                    })
                except Exception as e:
                    st.error(f"Error querying RAG engine: {e}")

# Tab 2: Manage Documents
with tab_docs:
    st.subheader("Upload PDF or TXT Documents")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded '{uploaded_file.name}' successfully! Ingest it below.")
        local_files = list_local_files()  # Refresh file list

    st.subheader("Available Files in Database")
    if not local_files:
        st.info("No files available. Drop a file in the uploader above to get started.")
    else:
        for f in local_files:
            col_name, col_size, col_status, col_actions = st.columns([4, 2, 2, 2])
            with col_name:
                st.markdown(f"📄 **{f['filename']}**")
            with col_size:
                st.text(f"{f['size_kb']:.1f} KB")
            
            is_indexed = f["filename"] in indexed_filenames
            with col_status:
                if is_indexed:
                    st.success("Indexed")
                else:
                    st.warning("Unindexed")
            
            with col_actions:
                col_idx, col_del = st.columns(2)
                with col_idx:
                    if st.button("Index", key=f"idx_{f['filename']}"):
                        with st.spinner(f"Indexing {f['filename']}..."):
                            try:
                                filepath = os.path.join(DATA_DIR, f["filename"])
                                # Delete old chunks first
                                store.delete_document(f["filename"])
                                chunks = chunk_document(filepath, chunk_size, chunk_overlap)
                                if chunks:
                                    texts = [c["text"] for c in chunks]
                                    embs = [get_fallback_embedding(t) for t in texts]
                                    store.add_documents(chunks, embs)
                                    st.success(f"Indexed {f['filename']}!")
                                    st.rerun()
                                else:
                                    st.error("Document is empty or unreadable.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                with col_del:
                    if st.button("Delete", key=f"del_{f['filename']}"):
                        filepath = os.path.join(DATA_DIR, f["filename"])
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        store.delete_document(f["filename"])
                        st.info(f"Removed {f['filename']}.")
                        st.rerun()

# Tab 3: Retrieval Inspector (Search Lab)
with tab_lab:
    st.subheader("Inspect Similarity Matching & Cosine Scores")
    st.markdown("Enter a query below to see raw matching chunks and similarity percentages from the vector store, without calling any LLM.")
    
    lab_query = st.text_input("Enter search query:")
    if st.button("Query Vector Database"):
        if not lab_query:
            st.warning("Please enter a query.")
        elif not store.chunks:
            st.warning("No documents indexed.")
        else:
            with st.spinner("Retrieving matches..."):
                q_emb = get_embedding(lab_query, api_key)
                results = store.hybrid_search(lab_query, q_emb, top_k=top_k)
                
                st.write(f"Found {len(results)} matches:")
                for i, match in enumerate(results):
                    st.markdown(f"### Match #{i+1}")
                    col_info, col_score = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"**Source:** `{match['source']}` | **Page:** `{match['page']}`")
                    with col_score:
                        score = match['similarity_score']
                        st.metric("Cosine Match", f"{score*100:.1f}%")
                        st.progress(max(0.0, min(1.0, score)))
                    
                    st.text_area("Chunk Content:", value=match["text"], height=120, disabled=True, key=f"inspect_{i}")
                    st.markdown("---")
