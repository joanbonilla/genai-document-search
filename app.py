import os
import sys
import sys
import streamlit as st

from typing import List, Tuple

from langchain.llms import VertexAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../modules"))

from GoogleEmbeddings import GoogleEmbeddings

def generate_response(uploaded_file, query_text):

    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = GoogleEmbeddings()
        # Select LLM
        llm = VertexAI(
            model_name="text-bison@001",
            max_output_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            verbose=True
            )
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            verbose=True
            )

        return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 GenAI Document Search with ChromaDB and Langchain')
st.title('🦜🔗 GenAI Document Search with ChromaDB and Langchain')

# File upload
uploaded_file = st.file_uploader('Sube un documento', type='txt')
# Query text
query_text = st.text_input('Pregunta:', placeholder = 'Proporciona un resumen...', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(response)
