import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.routes import upload, ocr, documents, rag, openai_compat, keys

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="End-to-End Custom OCR & Permanently Free Local Document Intelligence / RAG Pipeline with Image Preprocessing, Text Detection & Recognition, Table Extraction, Grounded Document Q&A, and Custom Model API Key Auth.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend and external project integrations (Flutter, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under /api
app.include_router(upload.router, prefix=settings.API_V1_STR)
app.include_router(ocr.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)
app.include_router(rag.router, prefix=settings.API_V1_STR)
app.include_router(keys.router, prefix=settings.API_V1_STR)

# Register OpenAI Compatible endpoints at root (/v1) and under /api/v1
app.include_router(openai_compat.router)
app.include_router(openai_compat.router, prefix=settings.API_V1_STR)

# Mount Output directory for image & preview serving
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")

# Static directory path
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Mount Static directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Custom OCR API is running. Navigate to /docs for Swagger UI or open frontend."}

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "upload_dir": str(settings.UPLOAD_DIR),
        "output_dir": str(settings.OUTPUT_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
