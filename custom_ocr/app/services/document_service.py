import json
from pathlib import Path
from typing import Dict, Optional, List
from fastapi import UploadFile
import cv2
from PIL import Image

from app.core.config import settings
from app.core.security import generate_safe_filename, validate_upload_file
from app.models.document import DocumentMetadata, DocumentPage, DocumentStatus, DocumentType
from app.utils.pdf_utils import convert_pdf_to_images, get_pdf_page_count
from app.utils.image_utils import load_image

class DocumentService:
    """
    Step 1 — Upload & Document Processor:
    - Ingests file & validates
    - Saves temporarily
    - Converts PDF pages into images or normalizes image files
    - Manages document lifecycle and persistence
    """

    def __init__(self):
        # In-memory document registry with file backup
        self._registry: Dict[str, DocumentMetadata] = {}

    async def process_uploaded_file(self, file: UploadFile) -> DocumentMetadata:
        """Validates and processes uploaded file into document pages."""
        extension, file_size = await validate_upload_file(file)
        doc_id, unique_filename = generate_safe_filename(file.filename or "document")

        # Save original file to uploads/
        file_path = settings.UPLOAD_DIR / unique_filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        doc_output_dir = settings.OUTPUT_DIR / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        pages: List[DocumentPage] = []

        if extension == ".pdf":
            # PDF Processing: extract page count and render each page to image
            raw_pages = convert_pdf_to_images(
                pdf_path=str(file_path),
                output_dir=str(doc_output_dir),
                dpi=settings.OCR_DPI
            )
            for page_num, img_path, w, h in raw_pages:
                pages.append(DocumentPage(
                    page_number=page_num,
                    image_path=img_path,
                    width=w,
                    height=h,
                    dpi=settings.OCR_DPI
                ))
            page_count = len(pages)
            mime_type = "application/pdf"
        else:
            # Single image processing
            dest_img_path = doc_output_dir / f"page_1{extension}"
            with open(dest_img_path, "wb") as f:
                f.write(content)

            # Get image dimensions
            img = load_image(dest_img_path)
            h, w = img.shape[:2]

            pages.append(DocumentPage(
                page_number=1,
                image_path=str(dest_img_path),
                width=w,
                height=h,
                dpi=settings.OCR_DPI
            ))
            page_count = 1
            mime_type = file.content_type or f"image/{extension.lstrip('.')}"

        metadata = DocumentMetadata(
            doc_id=doc_id,
            original_filename=file.filename or "uploaded_document",
            saved_filename=unique_filename,
            file_path=str(file_path),
            file_size_bytes=file_size,
            mime_type=mime_type,
            page_count=page_count,
            status=DocumentStatus.PENDING,
            pages=pages
        )

        self._registry[doc_id] = metadata
        self._save_metadata_to_disk(metadata)
        return metadata

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        if doc_id in self._registry:
            return self._registry[doc_id]

        # Check disk if server restarted
        meta_file = settings.OUTPUT_DIR / doc_id / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = DocumentMetadata.model_validate(data)
                self._registry[doc_id] = meta
                return meta
        return None

    def update_document_status(
        self, doc_id: str, status: DocumentStatus, error_message: Optional[str] = None
    ) -> Optional[DocumentMetadata]:
        doc = self.get_document(doc_id)
        if doc:
            doc.status = status
            doc.error_message = error_message
            self._save_metadata_to_disk(doc)
        return doc

    def _save_metadata_to_disk(self, metadata: DocumentMetadata):
        doc_dir = settings.OUTPUT_DIR / metadata.doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        meta_file = doc_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

document_service = DocumentService()
