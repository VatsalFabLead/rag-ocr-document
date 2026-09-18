import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

# Project root directory: custom_ocr/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Custom OCR & Document Intelligence Pipeline"
    API_V1_STR: str = "/api"
    DEBUG: bool = True

    # Directories
    UPLOAD_DIR: Path = PROJECT_ROOT / "uploads"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DETECTION_MODEL_DIR: Path = MODELS_DIR / "detection_model"
    RECOGNITION_MODEL_DIR: Path = MODELS_DIR / "recognition_model"

    # Validation Limits
    MAX_FILE_SIZE_MB: int = 25
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024
    MAX_PDF_PAGES: int = 50
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"
    ]
    ALLOWED_MIME_TYPES: List[str] = [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
        "image/webp"
    ]

    # OCR Settings
    DEFAULT_OCR_ENGINE: str = "auto"  # options: auto, custom_cv, tesseract, easyocr
    OCR_DPI: int = 300
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40

    # Preprocessing toggles
    ENABLE_RESIZE: bool = True
    ENABLE_GRAYSCALE: bool = True
    ENABLE_DENOISE: bool = True
    ENABLE_CONTRAST_ENHANCEMENT: bool = True
    ENABLE_DESKEW: bool = True
    ENABLE_THRESHOLDING: bool = True
    SAVE_PREPROCESSING_SNAPSHOTS: bool = True

    # LLM & Custom Model Settings (100% Permanently Free & Local)
    CUSTOM_MODEL_NAME: str = "custom-rag-doc-v1"
    CUSTOM_EMBEDDING_NAME: str = "custom-doc-embed-v1"
    DEFAULT_MASTER_KEY: str = os.getenv("CUSTOM_MODEL_API_KEY", "sk-custom-ocr-master-v1.1")
    API_KEYS_FILE: Path = OUTPUT_DIR / "api_keys.json"
    RAG_DATA_DIR: Path = OUTPUT_DIR / "rag_store"

    # Optional legacy third-party variables (deprecated in favor of custom model)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_PROVIDER: str = "custom_local"  # options: custom_local, rule_based

    model_config = {
        "env_file": ".env",
        "extra": "allow"
    }

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.DETECTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.RECOGNITION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)
