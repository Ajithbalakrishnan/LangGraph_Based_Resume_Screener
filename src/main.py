import os
import sys
import streamlit as st

import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from read_pdf import extract_text

def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content="Welcome")]
  
def upload_resume():
    
    if st.session_state.resume_uploaded != None:
        for file in st.session_state.resume_uploaded:
            bytes_data = file.read()
            st.write("Resume Name:", file.name)

def upload_job_req():
    
    if st.session_state.job_req_uploaded != None:
        for file in st.session_state.job_req_uploaded:
            bytes_data = file.read()
            st.write("job_req_uploaded:", file.name)
            
user_query = st.chat_input("Type your message here...")
    
with st.sidebar:
  st.markdown(" Configuration settings")

#   st.text_input("OpenAI's API Key", type="password", key="api_key")
#   st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
#   st.text_input("GPT Model", "gpt-3.5-turbo", key="gpt_selection")
  st.file_uploader("Upload resumes", type=["pdf"], key="resume_uploaded", on_change=upload_resume,  accept_multiple_files=True)
  st.button("Clear", on_click=clear_message)
  
  st.file_uploader("Upload job requirement", type=["pdf"], key="job_req_uploaded", on_change=upload_job_req)

#   st.divider()
#   st.markdown("Resume Screener using LangGraph ")
#   st.divider()
#   st.markdown("AB")

def main():
    st.title("Resume Screener using LangGraph")



if __name__ == "__main__":
    main()
