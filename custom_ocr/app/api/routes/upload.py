from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.api.schemas.upload_schema import UploadResponse, PageInfo
from app.services.document_service import document_service

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("", response_model=UploadResponse, summary="Upload Image / PDF / Scan")
async def upload_document(file: UploadFile = File(...)):
    """
    Step 1 — Upload & Validation
    - Validates file type, size, and magic bytes
    - Converts PDFs into high-resolution page images
    - Prepares document for OCR pipeline
    """
    try:
        doc_metadata = await document_service.process_uploaded_file(file)

        page_infos = [
            PageInfo(
                page_number=p.page_number,
                image_url=f"/outputs/{doc_metadata.doc_id}/page_{p.page_number}.png"
                if doc_metadata.mime_type == "application/pdf"
                else f"/outputs/{doc_metadata.doc_id}/page_1{file.filename[file.filename.rfind('.'):] if '.' in file.filename else '.png'}",
                width=p.width,
                height=p.height
            )
            for p in doc_metadata.pages
        ]

        return UploadResponse(
            doc_id=doc_metadata.doc_id,
            filename=doc_metadata.original_filename,
            file_size_bytes=doc_metadata.file_size_bytes,
            mime_type=doc_metadata.mime_type,
            page_count=doc_metadata.page_count,
            pages=page_infos,
            uploaded_at=doc_metadata.created_at,
            message="Document uploaded, validated, and converted into pages successfully."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing upload: {str(e)}"
        )
