import streamlit as st
import os
from pypdf import PdfReader
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import List

# --- Configuration for Free Deployment ---
# This app uses st.secrets to securely load the API key.
# On Streamlit Community Cloud, you must set the secret "GEMINI_API_KEY".

# --- Environment Setup and Constants ---

def initialize_gemini():
    """Initializes the Gemini client components."""
    # Try to load the API key from Streamlit secrets or environment variables
    # We use __api_key as a placeholder for the user's secret key
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()
        
    # Initialize the LLM (Generator) and the Embeddings model
    # gemini-2.5-flash is fast and suitable for Q&A tasks
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    
    # Use the specific embedding model for vector creation
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key
    )
    
    return llm, embeddings

# --- RAG Pipeline Functions ---

def get_pdf_text(pdf_docs: List[BytesIO]) -> str:
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits the raw text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(text_chunks: List[str], embeddings) -> FAISS:
    """
    Converts text chunks into vectors and stores them in a FAISS in-memory index.
    This function is memoized with st.cache_resource to run only once per document set.
    """
    st.info("Creating vector store... This may take a moment.")
    
    # Create vectors (embeddings) for all chunks
    # Then create a FAISS index from these vectors and the text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    st.success("Vector Store Created Successfully!")
    return vector_store

def get_conversational_chain(llm):
    """
    Defines the prompt and structure for the Q&A chain.
    The chain is responsible for taking the context (retrieved documents) and 
    generating the final answer.
    """
    # This prompt instructs the LLM to act as a Q&A expert using only the provided context.
    prompt_template = """
    You are an expert Question Answering system. Your task is to answer the user's question 
    only based on the provided context. Do not use external knowledge. 
    If the answer is not found in the provided context, state clearly that the information is 
    not available in the documents. 
    Provide a detailed and well-structured answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Create the prompt and the RAG chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # We use load_qa_chain to combine the context and question using the LLM
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question: str, vector_store: FAISS, qa_chain):
    """
    Handles the user question by performing retrieval and generation.
    """
    # 1. Retrieval: Find the most relevant documents in the vector store
    docs = vector_store.similarity_search(user_question)

    # 2. Generation: Pass the question and retrieved docs to the LLM chain
    response = qa_chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Display the result to the user
    with st.chat_message("assistant", avatar="🤖"):
        st.write(response["output_text"])
        
    # Optional: Display the sources found
    with st.expander("Source Documents Used"):
        if docs:
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}** (Source: Page {doc.metadata.get('page', 'N/A')}):")
                st.text(doc.page_content[:200] + "...") # Show snippet
        else:
            st.write("No relevant documents found for this query.")

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Free Gemini RAG Doc-QA", layout="wide")
    st.title("📄 Free RAG Doc-QA App with Gemini & Streamlit")
    st.caption("Upload PDFs and ask questions based ONLY on the document content.")

    # Initialize the LLM and Embeddings models
    try:
        llm, embeddings = initialize_gemini()
    except Exception as e:
        # Error message handled within initialize_gemini
        return 

    # --- Sidebar for Data Upload and Processing ---
    with st.sidebar:
        st.header("1. Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files here and click 'Process'", 
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create the vector store and save it in session state
                    # The get_vector_store is decorated with @st.cache_resource
                    # so the processing is fast on subsequent runs.
                    st.session_state.vector_store = get_vector_store(text_chunks, embeddings)
                    st.session_state.qa_chain = get_conversational_chain(llm)
                    st.session_state.processing_complete = True
            else:
                st.warning("Please upload at least one PDF file.")

        st.markdown("---")
        st.markdown("🚀 **Deployment Stack (Fully Free)**")
        st.markdown("- **LLM:** Gemini 2.5 Flash (via API Free Tier)")
        st.markdown("- **Framework:** Streamlit (UI & Backend)")
        st.markdown("- **RAG:** LangChain, FAISS (In-Memory Vector Store)")
        st.markdown("- **Hosting:** Streamlit Community Cloud (Free)")

    # --- Main Chat Interface ---
    if st.session_state.get("processing_complete"):
        
        # Initialize chat history if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.write(message["content"])

        # Accept user input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to chat history and display
            st.session_state.messages.append({"role": "user", "avatar": "👤", "content": user_question})
            with st.chat_message("user", avatar="👤"):
                st.write(user_question)

            # Process and handle the question
            handle_user_input(
                user_question, 
                st.session_state.vector_store, 
                st.session_state.qa_chain
            )
            # Add assistant response to history (handled inside handle_user_input)
            
    else:
        st.info("Please upload your PDF documents in the sidebar and click 'Process' to begin.")

if __name__ == "__main__":
    main()
