import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

FAISS_INDEXER_DIR = ""
TEMP_DATA_DIR = "./temp_data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_FEATURES = ["technical skills", "Education_background", "Relavent Experiance"]