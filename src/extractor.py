from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


# persist_directory = "chroma_db"

# vectordb = Chroma.from_documents(
#     documents=docs, embedding=embeddings, persist_directory=persist_directory
# )

# vectordb.persist()
