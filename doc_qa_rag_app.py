import streamlit as st
import os
from pypdf import PdfReader
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from typing import List, Union

# --- Configuration for Free Deployment ---

def get_api_key() -> Union[str, None]:
    """Retrieves the GEMINI_API_KEY securely from Streamlit secrets or environment."""
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()
    return api_key

@st.cache_resource
def get_gemini_llm(api_key: str):
    """Initializes and caches the ChatGoogleGenerativeAI model."""
    st.info("Initializing Gemini Chat Model...")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

@st.cache_resource
def get_gemini_embeddings(api_key: str):
    """Initializes and caches the GoogleGenerativeAIEmbeddings model."""
    st.info("Initializing Gemini Embeddings Model...")
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key
    )

# --- RAG Pipeline Functions ---

def get_pdf_text(pdf_docs: List[BytesIO]) -> str:
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Preserve page numbers as metadata
        for i, page in enumerate(pdf_reader.pages):
            page_content = page.extract_text()
            # A simple way to tag page metadata to the chunk text
            text += f"\n\n---PAGE {i+1}---\n\n{page_content}"
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
def get_vector_store(text_chunks: List[str], api_key: str) -> FAISS:
    """
    Converts text chunks into vectors and stores them in a FAISS in-memory index.
    Accepts only hashable primitives (text_chunks list and api_key string).
    """
    st.info("Creating vector store... This may take a moment.")
    
    # Retrieve the cached embeddings model internally
    embeddings = get_gemini_embeddings(api_key) 
    
    # Create vectors (embeddings) for all chunks and create a FAISS index
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    st.success("Vector Store Created Successfully!")
    return vector_store

def get_conversational_chain(llm):
    """
    Defines the prompt and structure for the Q&A chain.
    """
    prompt_template = """
    You are an expert Question Answering system. Your task is to answer the user's question 
    only based on the provided context. Do not use external knowledge. 
    If the answer is not found in the provided context, state clearly that the information is 
    not available in the documents. 
    If possible, mention the page number(s) (e.g., "---PAGE X---") from the context 
    that informed your answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question: str, vector_store: FAISS, qa_chain):
    """
    Handles the user question by performing retrieval and generation.
    """
    # 1. Retrieval: Find the top 4 most relevant documents in the vector store
    # We increase k to 4 to provide more context to the LLM
    docs = vector_store.similarity_search(user_question, k=4)

    # 2. Generation: Pass the question and retrieved docs to the LLM chain
    response = qa_chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Display the result to the user
    with st.chat_message("assistant", avatar="🤖"):
        st.write(response["output_text"])
        
    # Optional: Display the sources found (the chunks of text)
    with st.expander("Source Context Used"):
        if docs:
            for i, doc in enumerate(docs):
                # The page number is now embedded in the text chunk itself
                source_identifier = doc.page_content.split("---PAGE")[1].split("---")[0].strip() if "---PAGE" in doc.page_content else "N/A"
                st.markdown(f"**Chunk {i+1}** (Source Page: {source_identifier}):")
                # Show snippet of the retrieved text
                st.text(doc.page_content[:250].strip() + "...") 
        else:
            st.write("No relevant documents found for this query.")

# --- Main Streamlit Application ---

def main():
    st.set_page_config(page_title="Free Gemini RAG Doc-QA", layout="wide")
    st.title("📄 Free RAG Doc-QA App with Gemini & Streamlit")
    st.caption("Upload PDFs and ask questions based ONLY on the document content.")

    # 1. Initialization and API Key Retrieval
    try:
        api_key = get_api_key()
        llm = get_gemini_llm(api_key)
        # Note: We don't need to call get_gemini_embeddings here, it will be called inside get_vector_store
    except Exception:
        # Error message handled within get_api_key
        return 

    # --- Sidebar for Data Upload and Processing ---
    with st.sidebar:
        st.header("1. Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files here and click 'Process'", 
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process Documents", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    
                    # FIXED: Passing the hashable api_key string instead of the unhashable embeddings object
                    st.session_state.vector_store = get_vector_store(text_chunks, api_key)
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
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.write(message["content"])

        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Display user message
            st.session_state.messages.append({"role": "user", "avatar": "👤", "content": user_question})
            with st.chat_message("user", avatar="👤"):
                st.write(user_question)

            # Process and handle the question
            handle_user_input(
                user_question, 
                st.session_state.vector_store, 
                st.session_state.qa_chain
            )
            # Note: The assistant response is added within handle_user_input
            
    else:
        st.info("Please upload your PDF documents in the sidebar and click 'Process' to begin.")

if __name__ == "__main__":
    main()
