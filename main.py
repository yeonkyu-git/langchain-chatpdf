__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st  
import tempfile
import os

## 제목
st.title("ChatPDF")
st.write("--")

## 파일업로드
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
st.write("--")

## 1. Upload & Load PDF
def pdf_to_document(uploaded_file):
  temp_dir = tempfile.TemporaryDirectory()
  temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
  with open(temp_filepath, "wb") as f:
      f.write(uploaded_file.getvalue())
  loader = PyPDFLoader(temp_filepath)
  pages = loader.load_and_split()
  return pages


if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file, type="pdf")

  ## 2. Split Text
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
  )
  texts = text_splitter.split_documents(pages)

  ## 3. Create Embeddings
  embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

  ## 4. Load to Vector Store
  db = Chroma.from_documents(texts, embeddings)

  ## 5. 질문 & 답변
  st.header("PDF에게 질문해보세요!!")
  question = st.text_input('질문을 입력하세요')

  if st.button('질문하기'):
    with st.spinner('Wait for it...'):
      llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
      retriever = db.as_retriever()
      system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
      )
      prompt = ChatPromptTemplate.from_messages(
        [
          ("system", system_prompt),
          ("human", "{input}"),
        ]
      )
      question_answer_chain = create_stuff_documents_chain(llm, prompt)
      chain = create_retrieval_chain(retriever, question_answer_chain)
      result = chain.invoke({"input": question})
      st.write(result["answer"])


