"""MIA Backend - Configuration"""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    APP_NAME: str = "MIA"
    DEBUG: bool = Field(default=True, alias="MIA_DEBUG")
    HOST: str = Field(default="0.0.0.0", alias="MIA_HOST")
    PORT: int = Field(default=8000, alias="MIA_PORT")
    WORKERS: int = 4
    MODEL_PATH: str = Field(default="weights/best_mia.pt", alias="MIA_MODEL_PATH")
    CONFIDENCE_THRESHOLD: float = 0.5
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./data/mia_violations.db", alias="MIA_DATABASE_URL")
    UPLOAD_DIR: str = "data/uploads"
    OUTPUT_DIR: str = "data/outputs"
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    LOG_DIR: str = "logs/api"
    CORS_ORIGINS: List[str] = ["*"]
    WS_MAX_CONNECTIONS: int = 10
    
    class Config:
        env_prefix = "MIA_"
        env_file = ".env"
        extra = "ignore"
    
    def get_upload_path(self, filename: str) -> Path:
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        return Path(self.UPLOAD_DIR) / filename
    
    def get_output_path(self, filename: str) -> Path:
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        return Path(self.OUTPUT_DIR) / filename
    
    def is_allowed_file(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.ALLOWED_EXTENSIONS

settings = Settings()
