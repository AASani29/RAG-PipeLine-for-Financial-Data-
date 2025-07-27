import streamlit as st
import tempfile
import os
import time
import uuid
import traceback

# Page configuration
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your RAG implementation
try:
    # Try to import from your notebook conversion
    from rag_implementation import AdvancedRAGPipeline, EnhancedRAGPipeline, BasicRAGPipeline
    st.success("✅ Successfully imported RAG implementation")
    IMPLEMENTATION_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Could not import RAG implementation: {e}")
    st.info("Please make sure rag_implementation.py is in the same folder with all your classes")
    IMPLEMENTATION_AVAILABLE = False

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "pipeline_type" not in st.session_state:
    st.session_state.pipeline_type = "Advanced"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        margin-right: 20%;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
    }
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

@st.cache_resource
def get_pipeline_class(pipeline_type):
    """Get the appropriate pipeline class"""
    if not IMPLEMENTATION_AVAILABLE:
        return None
    
    if pipeline_type == "Advanced":
        return AdvancedRAGPipeline
    elif pipeline_type == "Enhanced":
        return EnhancedRAGPipeline
    else:
        return BasicRAGPipeline

def initialize_rag(file_path, pipeline_type="Advanced"):
    """Initialize RAG pipeline with uploaded document"""
    if not IMPLEMENTATION_AVAILABLE:
        st.error("RAG implementation not available. Please check your rag_implementation.py file.")
        return None
    
    try:
        with st.spinner(f"🔄 Processing document with {pipeline_type} RAG Pipeline..."):
            pipeline_class = get_pipeline_class(pipeline_type)
            if pipeline_class is None:
                st.error("Pipeline class not available")
                return None
            
            pipeline = pipeline_class()
            pipeline.setup(file_path)
            return pipeline
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

def format_response_time(seconds):
    """Format response time for display"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def display_chat_history():
    """Display the chat history"""
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show metadata if available
                if "metadata" in message:
                    with st.expander("📊 Response Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Response Time", format_response_time(message['metadata']['query_time']))
                        with col2:
                            st.metric("Sources Found", message['metadata']['sources'])
                        with col3:
                            confidence = message['metadata'].get('confidence', 0)
                            st.metric("Max Confidence", f"{confidence:.3f}")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("📄 Document Management")
    
    # Pipeline selection
    if IMPLEMENTATION_AVAILABLE:
        st.subheader("🔧 Pipeline Settings")
        pipeline_type = st.selectbox(
            "Choose RAG Pipeline:",
            ["Advanced", "Enhanced", "Basic"],
            index=0,
            help="Advanced: Full features, Enhanced: Multi-modal, Basic: Simple RAG"
        )
        st.session_state.pipeline_type = pipeline_type
        
        # Pipeline description
        descriptions = {
            "Advanced": "🚀 Full-featured with query optimization, multi-scale retrieval, and iterative processing",
            "Enhanced": "⚡ Multi-modal processing with text and table extraction",
            "Basic": "🎯 Simple and fast RAG pipeline"
        }
        st.info(descriptions[pipeline_type])
    
    # File upload section
    st.subheader("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to chat with (max 50MB)",
        key="pdf_uploader"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"📁 Selected: {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size / 1024 / 1024:.1f} MB")
        
        if st.button("🚀 Process Document", type="primary", disabled=not IMPLEMENTATION_AVAILABLE):
            if not IMPLEMENTATION_AVAILABLE:
                st.error("Cannot process - RAG implementation not loaded")
            else:
                # Save and process the uploaded file
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.session_state.rag_pipeline = initialize_rag(file_path, pipeline_type)
                    if st.session_state.rag_pipeline:
                        st.session_state.document_loaded = True
                        st.session_state.chat_history = []  # Clear previous chat
                        st.session_state.document_name = uploaded_file.name
                        st.success("✅ Document processed successfully!")
                        # Clean up temp file
                        os.unlink(file_path)
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")
    
    # Document status and controls
    if st.session_state.document_loaded:
        st.success("📚 Document Ready")
        st.info(f"📄 **Current Document:**\n{st.session_state.document_name}")
        st.info(f"🔧 **Pipeline:** {st.session_state.pipeline_type}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 New Document"):
                st.session_state.document_loaded = False
                st.session_state.rag_pipeline = None
                st.session_state.chat_history = []
                st.session_state.document_name = ""
                st.rerun()
    
    # Statistics
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📊 Chat Statistics")
        
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Your Questions", user_messages)
        
        # Calculate average response time
        response_times = [
            m.get("metadata", {}).get("query_time", 0) 
            for m in st.session_state.chat_history 
            if m.get("role") == "assistant" and "metadata" in m
        ]
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            st.metric("Avg Response Time", format_response_time(avg_time))

# Main content area
st.markdown('<div class="main-header"><h1>🤖 RAG Document Chatbot</h1><p>Upload a PDF document and have an intelligent conversation with it!</p></div>', unsafe_allow_html=True)

# Check if implementation is available
if not IMPLEMENTATION_AVAILABLE:
    st.error("🚨 RAG Implementation Not Found")
    st.markdown("""
    **To fix this issue:**
    
    1. **Extract your classes from the Jupyter notebook:**
       ```bash
       jupyter nbconvert --to script Code.ipynb
       ```
    
    2. **Create `rag_implementation.py`** with all your classes
    
    3. **Make sure these classes are included:**
       - `AdvancedRAGPipeline`
       - `EnhancedRAGPipeline` 
       - `BasicRAGPipeline`
       - And all supporting classes
    
    4. **Place the file in the same folder as this app**
    """)
    st.stop()

# Main chat interface
if not st.session_state.document_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 👋 Welcome to RAG Document Chatbot!
        
        **How it works:**
        1. 📤 **Upload** a PDF document using the sidebar
        2. 🔧 **Choose** your preferred RAG pipeline
        3. 🤖 **Wait** for processing (we'll extract and index content)
        4. 💬 **Chat** with your document using natural language
        5. 🎯 **Get** accurate, context-aware answers
        
        **Perfect for:**
        - 📊 Financial reports and earnings statements
        - 📑 Research papers and academic documents
        - 📋 Legal documents and contracts
        - 📈 Business plans and strategy documents
        - 📚 Educational materials and textbooks
        
        **Powered by advanced AI:**
        - 🧠 Sentence transformers for semantic understanding
        - 🔍 Vector similarity search with FAISS
        - 📝 Multi-modal processing (text + tables)
        - ⚡ Query optimization and enhancement
        """)
    
    st.info("👈 Please upload a PDF document in the sidebar to start chatting.")
    
    # Show sample questions
    st.subheader("💡 Sample Questions You Can Ask:")
    sample_questions = [
        "What are the key highlights of this document?",
        "What was the revenue/income in the latest quarter?", 
        "How did performance compare to the previous year?",
        "What are the main risks mentioned?",
        "Summarize the financial results",
        "What are the future plans or outlook?"
    ]
    
    for question in sample_questions:
        st.markdown(f"- *{question}*")

else:
    # Chat interface
    st.subheader(f"💬 Chat with {st.session_state.document_name}")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask a question about your document...", key="chat_input")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Get response from RAG pipeline
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                result = st.session_state.rag_pipeline.query(user_input)
                end_time = time.time()
                
                response = result['answer']
                query_time = result.get('query_time', end_time - start_time)
                search_results = result.get('search_results', [])
                sources_count = len(search_results)
                
                # Calculate confidence score
                if search_results:
                    max_confidence = max([r.get('score', 0) for r in search_results])
                else:
                    max_confidence = 0
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": time.time(),
                    "metadata": {
                        "query_time": query_time,
                        "sources": sources_count,
                        "confidence": max_confidence,
                        "pipeline_type": st.session_state.pipeline_type
                    }
                })
                
                st.rerun()
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": time.time(),
                    "error": True
                })
                st.error(f"Error: {e}")
                st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit and Advanced RAG Technology</p>
        <p><small>Session ID: {}</small></p>
    </div>
    """.format(st.session_state.session_id[:8] + "..."), unsafe_allow_html=True)

# Debug information (only show in development)
if st.sidebar.checkbox("🔧 Debug Info", value=False):
    st.sidebar.subheader("Debug Information")
    st.sidebar.json({
        "Implementation Available": IMPLEMENTATION_AVAILABLE,
        "Document Loaded": st.session_state.document_loaded,
        "Pipeline Type": st.session_state.pipeline_type,
        "Chat History Length": len(st.session_state.chat_history),
        "Session ID": st.session_state.session_id
    })