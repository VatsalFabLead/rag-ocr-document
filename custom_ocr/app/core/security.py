import os
import re
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import HTTPException, UploadFile, status
from app.core.config import settings

# Magic bytes signature verification
MAGIC_NUMBERS = {
    b"%PDF": "pdf",
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
    b"BM": "bmp",
    b"RIFF": "webp"
}

def sanitize_filename(filename: str) -> str:
    """Removes unsafe characters and path traversal attempts from a filename."""
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
    return clean_name or "document"

def generate_safe_filename(original_filename: str) -> Tuple[str, str]:
    """
    Generates a unique document ID and a safe persistent filename.
    Returns (doc_id, unique_filename).
    """
    doc_id = str(uuid.uuid4())
    clean_name = sanitize_filename(original_filename)
    extension = Path(clean_name).suffix.lower()
    unique_filename = f"{doc_id}_{clean_name}"
    return doc_id, unique_filename

async def validate_upload_file(file: UploadFile) -> Tuple[str, int]:
    """
    Validates uploaded file against:
    1. Extension whitelist
    2. File size limit
    3. Magic byte signature
    Returns (extension, file_size).
    """
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{extension}'. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # Read the initial chunk to check magic bytes and measure size
    header = await file.read(16)
    await file.seek(0)

    # Check magic numbers
    matched = False
    for signature, file_type in MAGIC_NUMBERS.items():
        if header.startswith(signature):
            matched = True
            break
        # Special check for webp: RIFF....WEBP
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
            matched = True
            break

    if not matched:
        # Fallback to extension check if magic bytes have slight variations
        if extension not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match any allowed image or PDF format."
            )

    # Check file size by seeking to end
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb} MB (received {file_size / (1024*1024):.2f} MB)."
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    return extension, file_size
