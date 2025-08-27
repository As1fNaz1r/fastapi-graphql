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

def _store_append(records: List[Dict[str, Any]]):
    with open(STORE_FILE, "a", encoding="utf-8") as f:
        for f in records:
            f.write(jjson.dumps(r, ensure_ascii=False)+"\n")



def _store_load_all() -> List[Dict[str, Any]]:
    if not os.path.exists(STORE_FILE):
        return []
    with open(STORE_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _embed_text(text: str) -> np.ndarray:
    """Return a (1, D) numpy array embedding for the text."""
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    vec = np.array(resp["embedding"], dtype="float32")
    return vec.reshape(1, -1)

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Simple char-based chunking (good enough to start)."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def _load_or_create_index(first_vector: np.ndarray | None = None) -> faiss.Index:
    """Load FAISS index if available; otherwise create using provided vector dimension."""
    _ensure_dirs()
    if os.path.exists(INDEX_FILE):
        return _read_index()
    if first_vector is None:
        # Fresh start but no vector yet: make a tiny placeholder; we'll rebuild when we get first vector.
        # However, FAISS requires dimension at construction; so if not known, we delay creation.
        raise RuntimeError("Index does not exist yet. Provide a first_vector to create it.")
    dim = first_vector.shape[1]
    index = faiss.IndexFlatL2(dim)
    _write_index(index)
    return index



#----------------------------------------------------------
# Schemas
#----------------------------------------------------------

class IngestRequest(BaseModel):
    



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
