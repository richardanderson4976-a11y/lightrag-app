import streamlit as st
import os
import tempfile
from pathlib import Path
import asyncio
from datetime import datetime

st.set_page_config(
    page_title="LightRAG with Gemini",
    page_icon="🔦",
    layout="wide"
)

# Initialize session state
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def initialize_lightrag(api_key):
    """Initialize LightRAG with Gemini"""
    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete, openai_embedding
        
        working_dir = "./rag_storage"
        os.makedirs(working_dir, exist_ok=True)
        
        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete(
                "gemini-2.0-flash-exp",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                **kwargs
            )

        async def embed_func(texts):
            return await openai_embedding(
                texts,
                model="text-embedding-004",
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            embedding_func=embed_func,
        )
        
        return rag
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None

def load_document(rag, file_content, filename):
    """Load document into RAG"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag.ainsert(tmp_path))
        loop.close()
        
        os.unlink(tmp_path)
        return True
    except Exception as e:
        st.error(f"Load failed: {str(e)}")
        return False

def query_rag(rag, query, mode="hybrid"):
    """Query the RAG system"""
    try:
        from lightrag import QueryParam
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            rag.aquery(query, param=QueryParam(mode=mode))
        )
        loop.close()
        return result
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.title("🔦 LightRAG Settings")
    
    # API Key - supports both secrets and manual input
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Using API key from secrets")
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key from https://aistudio.google.com/app/apikey"
        )
    
    if api_key and st.session_state.rag_instance is None:
        with st.spinner("Initializing LightRAG..."):
            st.session_state.rag_instance = initialize_lightrag(api_key)
            if st.session_state.rag_instance:
                st.success("✅ LightRAG Ready!")
    
    st.divider()
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['txt', 'md', 'pdf', 'docx'],
        help="Upload text files, PDFs, or Word documents"
    )
    
    if uploaded_files and st.session_state.rag_instance:
        if st.button("📥 Load Documents", type="primary"):
            progress_bar = st.progress(0)
            success_count = 0
            
            for idx, file in enumerate(uploaded_files):
                with st.status(f"Loading {file.name}...", expanded=True) as status:
                    content = file.read()
                    if load_document(st.session_state.rag_instance, content, file.name):
                        st.write(f"✅ {file.name}")
                        success_count += 1
                    else:
                        st.write(f"❌ {file.name}")
                    status.update(label=f"Loaded {idx + 1}/{len(uploaded_files)}", state="running")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.session_state.documents_loaded = True
            st.success(f"✅ Successfully loaded {success_count}/{len(uploaded_files)} documents!")
    
    st.divider()
    
    # Query mode
    st.subheader("🔍 Query Mode")
    query_mode = st.selectbox(
        "Select mode",
        ["hybrid", "local", "global", "naive"],
        help="""
        • Hybrid: Best overall results (recommended)
        • Local: Focus on specific entities
        • Global: High-level summaries
        • Naive: Simple vector search
        """
    )
    
    st.divider()
    
    # Stats
    if st.session_state.documents_loaded:
        st.metric("📊 Documents", len(uploaded_files) if uploaded_files else 0)
        st.metric("💬 Messages", len(st.session_state.messages))
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main area
st.title("💬 Chat with Your Documents")

if not api_key:
    st.info("👈 **Get Started:** Enter your Gemini API key in the sidebar")
    st.markdown("""
    ### How to get a Gemini API key:
    1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
    2. Click "Create API Key"
    3. Copy the key and paste it in the sidebar
    
    **It's free!** 🎉
    """)
    
elif not st.session_state.rag_instance:
    st.warning("⚠️ Failed to initialize. Please check your API key.")
    
elif not st.session_state.documents_loaded:
    st.info("👈 **Next Step:** Upload documents in the sidebar to start chatting")
    st.markdown("""
    ### Supported formats:
    - 📄 Text files (.txt, .md)
    - 📕 PDF documents (.pdf)
    - 📘 Word documents (.docx)
    
    Upload multiple files at once!
    """)
    
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"🕐 {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕐 {timestamp}")
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = query_rag(
                    st.session_state.rag_instance,
                    prompt,
                    mode=query_mode
                )
                
                if response:
                    st.markdown(response)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.caption(f"🕐 {timestamp}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp
                    })
                else:
                    st.error("Failed to get response. Please try again.")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔦 Powered by LightRAG")
with col2:
    st.caption("🤖 Gemini 2.0 Flash")
with col3:
    st.caption("📊 text-embedding-004")
