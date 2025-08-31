# app.py
import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from PyPDF2 import PdfReader
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from ctransformers import AutoModelForCausalLM as CTransformersModel
from typing import Any

# Page setup
st.set_page_config(page_title="MyAi.advisor", page_icon="🧠")
st.title("🧠 MyAi.advisor")
st.markdown("AI-powered knowledge assistant for enterprise teams")

# Load documents from all supported formats
def load_docs():
    docs = []
    data_path = "data"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")
    
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        file_ext = file.lower().split(".")[-1]
        
        # .txt
        if file_ext == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
        
        # .docx
        elif file_ext == "docx":
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())
        
        # .pdf
        elif file_ext == "pdf":
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                docs.append({"page_content": text, "metadata": {"source": file}})
            except Exception as e:
                st.warning(f"Could not read PDF: {file} | Error: {str(e)}")
        
        # .pptx
        elif file_ext == "pptx":
            try:
                prs = Presentation(file_path)
                text = ""
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
                docs.append({"page_content": text, "metadata": {"source": file}})
            except Exception as e:
                st.warning(f"Could not read PPTX: {file} | Error: {str(e)}")
    
    if not docs:
        raise ValueError("No documents loaded. Check your data folder and file formats.")
    
    return docs

# Split text into chunks
@st.cache_resource
def get_vectorstore():
    docs = load_docs()
    # Convert dict to LangChain Document if needed
    from langchain.schema import Document
    formatted_docs = [
        Document(page_content=d["page_content"], metadata=d.get("metadata", {})) 
        if isinstance(d, dict) else d 
        for d in docs
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(formatted_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(texts, embedding=embeddings)

# Custom LLM wrapper for ctransformers
class CTransformersLLM(LLM):
    model: Any

    @property
    def _llm_type(self) -> str:
        return "ctransformers"

    def _call(self, prompt: str, **kwargs) -> str:
        return self.model(prompt)

# Load LLM locally
@st.cache_resource
def get_llm():
    model = CTransformersModel.from_pretrained(
        r"D:\Documents\MY PROJECTS\Personal Projects\MyAi.advisor\model\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama",
        max_new_tokens=200,
        temperature=0.3,
        context_length=2048
    )
    return CTransformersLLM(model=model)

# Get the QA chain
@st.cache_resource
def get_qa_chain():
    llm = get_llm()
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 1})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# Fix: Rename st_cache_resource to st.cache_resource
@st.cache_resource
def get_llm():
    model = CTransformersModel.from_pretrained(
        r"D:\Documents\MY PROJECTS\Personal Projects\MyAi.advisor\model\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama",
        max_new_tokens=200,
        temperature=0.3,
        context_length=2048
    )
    return CTransformersLLM(model=model)

# Initialize QA chain
try:
    qa_chain = get_qa_chain()
except Exception as e:
    st.error(f"Error initializing QA chain: {str(e)}")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from user
if prompt := st.chat_input("Ask about our AI solutions..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            result = qa_chain({"query": prompt})
            answer = result["result"]
            sources = result["source_documents"]
        except Exception as e:
            answer = "Sorry, I couldn't process your request."
            sources = []
            st.error(f"Error generating response: {str(e)}")

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("See source"):
                st.write(sources[0].page_content[:500] + "..." if len(sources[0].page_content) > 500 else sources[0].page_content)
    st.session_state.messages.append({"role": "assistant", "content": answer})
