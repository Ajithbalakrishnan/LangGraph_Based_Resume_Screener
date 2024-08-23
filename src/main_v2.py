import os
import sys
from io import BytesIO

import streamlit as st
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import *

from streamlit_modal import Modal
import PyPDF2
from dotenv import load_dotenv, dotenv_values
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from langchain_chroma import Chroma
# embeddings = FakeEmbeddings(size=4096)

# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': False}
# hf = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

load_dotenv()

def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content="Welcome")]
    chroma_db.clear()  

def upload_resume():
    if st.session_state.resume_uploaded is not None:
        for file in st.session_state.resume_uploaded:
            bytes_data = file.read()
            resume_text = extract_text(bytes_data)
            
            summarize_resume_google(resume_text)
            st.write("Resume Name:", file.name)

def extract_text(file_data):
    try:
        pdf_stream = BytesIO(file_data)
        pdf_reader = PyPDF2.PdfFileReader(pdf_stream)
        text = ""
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extractText()
        
    except UnicodeDecodeError:
        st.error("Error: Could not decode PDF. Please try a different file.")
        return 
    except Exception as e:
        st.error("An error occurred while extracting text:", e)

        return ""
    else:
        return text

def summarize_resume_openai(resume_text):

    openai_api_key = st.session_state.get("api_key")
    
    client = OpenAI(api_key=os.getenv("OPEN_AI_SECRET_KEY"),)



   
    prompts = {
        "experience": f"Summarize the professional experience section of the following text:\n{resume_text}",
        "technical_skills": f"Summarize the technical skills section of the following text:\n{resume_text}",
        "education": f"Summarize the education background section of the following text:\n{resume_text}",
    }

    for aspect, prompt in prompts.items():
        # summary = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=100)
        summary = client.chat.completions.create(
                messages=[
                    {
                        "role": "You are a document summary creator focussed on Resumes",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
            )
        summary_text = summary.choices[0].text.strip()
        st.write(f"{aspect.title()} Summary:")
        st.write(summary_text)
    
        embedding = HuggingFaceEmbeddings().encode(summary_text)
        chroma_db.insert(f"resume_{aspect}", embedding)

def summarize_resume_google(resume_text):

    # openai_api_key = st.session_state.get("api_key")
    data=list()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
   
    prompts = {
        "experience": f"Summarize the professional experience section of the following text:\n{resume_text}",
        "technical_skills": f"Summarize the technical skills section of the following text:\n{resume_text}",
        "education": f"Summarize the education background section of the following text:\n{resume_text}",
    }

    for aspect, prompt in prompts.items():
        
        messages = [
                    (
                        "system",
                        "You are a professional document summary creator who mainly focussed on job resumes",
                    ),
                    ("human", prompt),
                ]
        summary = llm.invoke(messages)
        summary_text = summary.content
        st.write(f"{aspect.title()} Summary:")
        st.write(summary_text)
        data.append(summary_text)
        
    return data

def init_vector_store():
    vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )
    
def store_vector_data(data):
    
    
def upload_job_req():
    if st.session_state.job_req_uploaded != None:
        for file in st.session_state.job_req_uploaded:
            bytes_data = file.read()
            st.write("Job Requirement:", file.name)


user_query = st.chat_input("Type your message here...")
init_vector_store()
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"), model_kwargs={"device": "cpu"})

with st.sidebar:
    st.text_input("OpenAI's API Key", type="password", key="api_key")
    st.file_uploader("Upload resumes", type=["pdf"], key="resume_uploaded", on_change=upload_resume, accept_multiple_files=True)
    st.button("Clear", on_click=clear_message)
    st.file_uploader("Upload job requirement", type=["pdf"], key="job_req_uploaded", on_change=upload_job_req)


def main():
    st.title("Resume Screener")

if __name__ == "__main__":
    main()