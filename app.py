import os
import sys
import tempfile
import streamlit as st

from typing import List, Tuple
from PyPDF2 import PdfReader

from langchain.document_loaders import PyPDFLoader
from langchain.llms import VertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../modules"))

from GoogleEmbeddings import GoogleEmbeddings

template = """SYSTEM: Eres un asistente inteligente que ayuda a los usuarios con sus preguntas sobre documentos.

Pregunta: {question}

Utiliza estrictamente SÓLO los siguientes fragmentos de contexto para responder la pregunta al final. Piensa paso a paso y luego responde.

No intentes inventar una respuesta:
  - Si la respuesta a la pregunta no se puede determinar únicamente a partir del contexto, diga "No puedo determinar la respuesta a eso".
  - Si el contexto está vacío, simplemente diga "No he encontrado información relativa a la pregunta".

=============
{context}
=============

Pregunta: {question}
Respuesta:"""

def generate_response(uploaded_files, question):

    if uploaded_files is not None:
        #documents = [uploaded_file.read().decode()]
        #pdf_reader = PyPDFLoader(uploaded_file)
        #text = ""
        #for page in pdf_reader.pages:
        #    text += page.extract_text()

        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())
      
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
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
                "search_distance": 0.5,
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
st.set_page_config(page_title='🦜🔗 GenAI Document Search with PaLM2, Langchain and ChromaDB')
st.title('🦜🔗 GenAI Document Search with PaLM2, Langchain and ChromaDB')

# File upload
uploaded_files = st.file_uploader('Sube un documento', type='pdf', accept_multiple_files=True)
# Query text
query_text = st.text_input('Pregunta:', placeholder = 'Proporciona un resumen...', disabled=not uploaded_files)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, query_text)
            result.append(response)

if len(result):
    st.info(response)
