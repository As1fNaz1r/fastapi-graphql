from fastapi import FastAPI
from routes import rag
from config import settings

app = FastAPI(
    title ="LLM",
    version = "1.0.0",
    description = "A minimal app for LLM API"
)

#include routes from rag

app.include_routes()