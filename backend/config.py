from pydantic_settings import BaseSettings
class settings(BaseSettings):
    ENV: str = "dev"
    GEMINI_API_KEY: str
    VECTOR_STORE_PATH = "data/vector_store"

    class Config:
        env_file =".env"


settings = Settings()
