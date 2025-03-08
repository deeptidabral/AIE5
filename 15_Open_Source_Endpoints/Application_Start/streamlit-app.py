import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

# Import LangChain components
from langchain.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import tqdm

# Load environment variables
load_dotenv()

# Environment variables
HF_LLM_ENDPOINT = os.environ.get("HF_LLM_ENDPOINT")
HF_EMBED_ENDPOINT = os.environ.get("HF_EMBED_ENDPOINT")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Set page config
st.set_page_config(page_title="Paul Graham Essay Bot", page_icon="📚", layout="wide")

# App title and description
st.title("Paul Graham Essay Bot")
st.markdown("Ask questions about Paul Graham's essays")

# Sidebar with app information
with st.sidebar:
    st.title("About")
    st.info("This app uses a Hugging Face model to answer questions about Paul Graham's essays.")
    st.markdown("---")
    st.subheader("Environment Setup")
    if st.checkbox("Show Environment Variables"):
        st.write(f"LLM Endpoint: {HF_LLM_ENDPOINT}")
        st.write(f"Embedding Endpoint: {HF_EMBED_ENDPOINT}")
        st.write("HF Token: ********" if HF_TOKEN else "HF Token not set")

# Initialize session state to keep track of messages and retriever
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def initialize_retriever():
    """Initialize the retriever with vector database."""
    with st.spinner("Loading and indexing documents..."):
        # Load documents
        document_loader = TextLoader("./data/paul_graham_essays.txt")
        documents = document_loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
        split_documents = text_splitter.split_documents(documents)
        
        # Initialize HuggingFace embeddings
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Using a standard model that works without token
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(split_documents, hf_embeddings)
        
        # Return retriever
        return vectorstore.as_retriever()

def initialize_rag_chain(retriever):
    """Initialize the RAG chain with the retriever."""
    # RAG prompt template
    RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    
    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # HuggingFace LLM endpoint
    hf_llm = HuggingFaceEndpoint(
        endpoint_url=HF_LLM_ENDPOINT,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.3,
        repetition_penalty=1.15
    )
    
    # Build LCEL RAG chain
    lcel_rag_chain = (
        {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
        | rag_prompt 
        | hf_llm
    )
    
    return lcel_rag_chain

# Initialize resources on first load
if st.session_state.retriever is None:
    try:
        st.session_state.retriever = initialize_retriever()
        st.session_state.rag_chain = initialize_rag_chain(st.session_state.retriever)
        st.success("App initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Paul Graham's essays"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_placeholder = st.empty()
            
            # Call the RAG chain
            response = st.session_state.rag_chain.invoke({"query": prompt})
            
            # Display response
            response_placeholder.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})