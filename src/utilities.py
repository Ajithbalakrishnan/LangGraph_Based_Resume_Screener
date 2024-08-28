import os
import sys
from io import BytesIO
import PyPDF2
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

from config import *


def save_pdfs(data, file_name, st):

    if not os.path.exists(TEMP_DATA_DIR):
        os.makedirs(TEMP_DATA_DIR)
        
    pdf_stream = BytesIO(data)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    pdf_writer = PyPDF2.PdfWriter()

    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.getPage(page_num))


    pdf_filename = file_name.split('.')[0] + '_saved.pdf'
    save_path = os.path.join(TEMP_DATA_DIR, pdf_filename)
    with open(save_path, 'wb') as output_file:
        pdf_writer.write(output_file)

    st.success(f"PDF saved to: {save_path}")
    
def generate_fake_summary(resume_text):
    fake_summary = resume_text[:500] 
    return fake_summary

def check_chroma_db():
    # client = chromadb.Client(Settings(is_persistent=True,
    #                                     persist_directory= "./chroma_langchain_db",))
    client = chromadb.Client( settings=Settings(is_persistent=True,
                                        persist_directory= "./chroma_langchain_db",),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
        )
    coll = client.get_collection("example_collection")
    
    print(f"Collection Count: {coll.count()}")
    
if __name__ == "__main__":
    check_chroma_db()