from fastapi import APIRouter, Query
import google.generativei as genai 
import faiss
import os
import pickle
from config import settings


router = APIRouter()

# Configure GEMINI API 
genai.configure(api_key = settings.GEMINI_API_KEY)


# Load or Create FAISS Index
INDEX_PATH = settings.VECTOR_STORE_PATH
DIM = 768

if not os.path.exists(INDEX_PATH):
    os.makedirs(INDEX_PATH, exist_ok=True)
    index = faiss.IndexFlat2(DIM)
    pickle.dump(index, open(f"{INDEX_PATH}/fiass_index.pkl","wb"))

else:
    index = pickle.load(open(f"{INDEX_PATH}/faiss_index.pkl","rb"))