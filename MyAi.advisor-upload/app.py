# app.py
import streamlit as st
from langchain_community.document_loaders import TextLoader
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

# Load documents
def load_docs():
    loader1 = TextLoader("data/sample1.txt", encoding="utf-8")
    loader2 = TextLoader("data/sample2.txt", encoding="utf-8")
    docs = loader1.load() + loader2.load()
    return docs

# Split text into chunks
@st.cache_resource
def get_vectorstore():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(texts, embedding=embeddings)

# Custom LLM wrapper for ctransformers
class CTransformersLLM(LLM):
    model: Any  # Use Any to avoid strict type checking

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

# Initialize QA chain
qa_chain = get_qa_chain()

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
        result = qa_chain({"query": prompt})
        answer = result["result"]
        sources = result["source_documents"]

    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("See source"):
            st.write(sources[0].page_content)
    st.session_state.messages.append({"role": "assistant", "content": answer})