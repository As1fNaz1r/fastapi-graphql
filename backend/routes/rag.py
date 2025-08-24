from fastapi import APIRouter, Query, Body
from pydantic import BaseModel
import google.generativeai as genai 
import faiss
import os
import json
from typing import List, Dict, Any
import pickle
from config import settings


router = APIRouter()

#----------------------------------------------
# Paths
#----------------------------------------------

INDEX_DIR = settings.VECTOR_STORE_PATH
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
STORE_FILE = os.path.join(INDEX_DIR, "store.jsonl")


#--------------------------------------------------
# Gemini Configuration
#--------------------------------------------------

genai.configure(api_key=settings.GEMINI_API_KEY)
EMBED_MODEL = "text-embedding-004"
GEN_MODEL = "gemini-pro"


#-------------------------------------------------------
# Utilities
#------------------------------------------------------
def _ensure_dirs():
    os.makedirs(INDEX_DIR, exist_ok=True)

def _write_index(index: faiss.Index):
    faiss.write_index(index, INDEX_FILE)

def _read_index() -> faiss.Index:
    return faiss.read_index(INDEX_FILE)

def _store_append()



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

@router.get("/ask")
def ask_agent(q: str = Query(..., description='Your question to the agent')):
    """ 
    1. Search FAISS (stub for now)
    2. Call gemini for response
    """
    context = "This is a placeholder context from FAISS"
    prompt = f"Answer the question based on the following context:\n{context}\n\n Question:{q}"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return {"question":q,"answer":response.text}
