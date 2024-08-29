import os
import sys
from io import BytesIO
import time

import random 
import PyPDF2
import streamlit as st
from openai import OpenAI
from uuid import uuid4
from dotenv import load_dotenv, dotenv_values

from streamlit_modal import Modal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import *
from utilities import save_pdfs, generate_fake_summary, check_chroma_db
from hf_spaces_inference import hf_spaces_infr
from groq_infrence import groq_infr

random.seed(10)
load_dotenv()
index_to_docstore_id = {}


def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content="Welcome")]
  
    # st.session_state.vector_store.clear()  

def delete_collection():
    try:
        your_collection_name = "your_collection_name"
        chroma_client = PersistentClient(path=".chroma")
        chroma_client.delete_collection(your_collection_name)
        print(f"Collection {collection_name} deleted successfully.")
    except Exception as e:
        raise Exception(f"Unable to delete collection: {e}")

def upload_resume():
    # modal_key = f"resume_error_{random.randint(1, 1000)}" 
    # modal = Modal(key=modal_key, title="File Error", max_width=500)
    if st.session_state.resume_uploaded is not None:
        summary_list = list()
        with st.toast('Processing the uploaded data. This may take a while...'):
            for file in st.session_state.resume_uploaded:
                try:
                    
                    bytes_data = file.read()
                    resume_text = extract_text(bytes_data)
                   
                    summary_data = generate_fake_summary(resume_text)
                    st.write(f"summary_data: {summary_data}")
                    print("summary_data : ",summary_data)
                   
                    document = Document(page_content= summary_data, metadata={"Resume_Name": str(file.name)}, id=random.random())
                    summary_list.append(document)
                    st.write("Resume Name:", file.name)
                    st.write(summary_data)
                    save_pdfs(data= bytes_data, file_name = str(file.name), st=st)
                    

# TODO WHile appending to chromadb, we have to remove the file from the list of documents
                
                except Exception as error:
                    # with modal.container():
                        st.write("The uploaded resume file returns the following error message. Please check your pdf file again.")
                        st.write(error)
                        print("The uploaded resume file returns the following error message ", error)
            try:
                store_vector_data(summary_list)
                
            except Exception as error:
                    # with modal.container():
                        st.write("The uploaded resume files couldn't save in DB. Please check the error message.")
                        st.write(error)
            else:
                st.write("Sucessfully stored the resume files to DB")
                # st.session_state.resume_uploaded=None
                

    
def extract_text(file_data):
    try:
        pdf_stream = BytesIO(file_data)
        pdf_reader = PyPDF2.PdfFileReader(pdf_stream)
        text = ""
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extract_text()
        return text
    except UnicodeDecodeError:
        st.write("Error: Could not decode PDF. Please try a different file.")
        return 
    except Exception as e:
        st.write (f"An error occurred while extracting text:{e}")

        return ""

def summarize_resume_openai(resume_text):

    openai_api_key = st.session_state.get("api_key")
    
    client = OpenAI(api_key=os.getenv("OPEN_AI_SECRET_KEY"),)



   
    prompts = {
        "experience": f"Summarize the professional experience section of the following text:\n{resume_text}",
        "technical_skills": f"Summarize the technical skills section of the following text:\n{resume_text}",
        "education": f"Summarize the education background section of the following text:\n{resume_text}",
    }
    data=list()
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
        data.append(summary_text)
        
    return ' '.join(data)

def summarize_resume_google(resume_text):

    # openai_api_key = st.session_state.get("api_key")
    data=list()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY2")
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

        data.append(summary_text)
        
    return ' '.join(data)

def summarize_resume_hf(resume_text):
    prompts = {
        "experience": f"Summarize the professional experience section of the following text:\n{resume_text}",
        "technical_skills": f"Summarize the technical skills section of the following text:\n{resume_text}",
        "education": f"Summarize the education background section of the following text:\n{resume_text}",
    }
    data = list()
    sys_instruction = "You are a professional document summary creator who mainly focussed on job resumes"
    for aspect, prompt in prompts.items():
        summary_text = hf_spaces_infr(sys_prompt=sys_instruction, message=prompt)
        data.append(summary_text)
    
    return ' '.join(data)

def summarize_resume_groq(resume_text):
    prompt = f"""
        Analize the following profile information of a candidate and generate a short summary on the each given aspects.
        
        Aspects:
            experience: Summarize the professional experience section of the following text,
            technical_skills: Summarize the technical skills section of the following text,
            education: Summarize the education background section of the following text,
        
        Profile Information: \n{resume_text}
    """
   
    sys_instruction = "You are a professional document summary creator who mainly focussed on job resumes"

    summary_text = groq_infr(sys_prompt=sys_instruction, message=prompt)
    
    return summary_text 
        
def init_vector_store():
    st.session_state.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=st.session_state.embedding_model,
            persist_directory="./chroma_langchain_db",
        )
    
def store_vector_data(documents):
    
    uuids = [str(uuid4()) for _ in range(len(documents))]

    st.session_state.vector_store.add_documents(documents=documents, ids=uuids)
    
def upload_job_req():
    # modal_key = f"job_req_error_{random.randint(1, 1000)}"
    # modal = Modal(key=modal_key, title="File Error", max_width=500)
    if st.session_state.job_req_uploaded is not None:
        try:  
            bytes_data = st.session_state.job_req_uploaded.read()
            job_req_text = extract_text(bytes_data)
            st.write("Job Requirement:", st.session_state.job_req_uploaded.name)
            
        except Exception as error:
            with st.toast('The uploaded job requirement file returns the following error message. Please check your pdf file again'):
                st.write(error)
                
        else:
            extract_K_chunks(job_req_text)
            

def extract_K_chunks(job_req_text):
    results = st.session_state.vector_store.similarity_search_by_vector(
    embedding= st.session_state.embedding_model.embed_query(job_req_text), k=10)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")
    
    
    
user_query = st.chat_input("Type your message here...")

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

if "rag_pipeline" not in st.session_state:
       
    if os.path.exists(FAISS_INDEXER_DIR):
        st.session_state.vectordb = FAISS.load_local(FAISS_INDEXER_DIR, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE,  allow_dangerous_deserialization=True)
    else:
        st.session_state.vectordb = FAISS(st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE)

init_vector_store()

with st.sidebar:
    st.text_input("OpenAI's API Key", type="password", key="api_key")
    st.file_uploader("Upload resumes", type=["pdf"], key="resume_uploaded", on_change=upload_resume, accept_multiple_files=True)
    st.button("Clear", on_click=clear_message)
    st.file_uploader("Upload job requirement", type=["pdf"], key="job_req_uploaded", on_change=upload_job_req)
    st.button("Profile Analysis", on_click=clear_message)

def main():
    st.title("Resume Screener")

if __name__ == "__main__":
    main()