import os
import sys
import sys
import streamlit as st

from typing import List, Tuple

from langchain.llms import VertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../modules"))

from GoogleEmbeddings import GoogleEmbeddings

template = """SYSTEM: You are an intelligent assistant helping the users with their questions on research papers.

Question: {question}

Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
 - If the context is empty, just say "I do not know the answer to that."

=============
{context}
=============

Question: {question}
Helpful Answer:"""

def generate_response(uploaded_file, question):

    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        print(f"# of documents = {len(texts)}")
     
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
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 1,
                "search_distance": 0.7,
            },
        )
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=template,
                    input_variables=["context", "question"],
                    ),
                }
            )

        qa.combine_documents_chain.verbose = True
        qa.combine_documents_chain.llm_chain.verbose = True
        qa.combine_documents_chain.llm_chain.llm.verbose = True

        return qa.run(question)

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
