import os
import base64
import tempfile
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.auth import verify_custom_api_key, optional_custom_api_key
from app.services.api_key_service import APIKeyRecord
from app.api.schemas.ocr_schema import (
    OCRProcessRequest, OCRProcessResponse, PreprocessingResponse, PreprocessingStepImage
)
from app.services.ocr_service import ocr_service
from app.services.document_service import document_service
from app.services.preprocessing_service import PreprocessingService
from app.ocr.recognizer import CustomRecognizer
from app.utils.image_utils import load_image

router = APIRouter(prefix="/ocr", tags=["Custom OCR Model Engine"])

class OCRBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image data string")
    filename: Optional[str] = Field("uploaded_image.png", description="Optional filename")
    engine: str = Field("auto", description="OCR engine: auto, rapidocr, custom_model, tesseract, builtin_cv")
    extract_fields: bool = Field(True, description="Extract structured document entities")
    extract_tables: bool = Field(True, description="Extract table rows and columns")
    document_type_hint: Optional[str] = Field(None, description="Hint: invoice, receipt, id_card, generic")

def _format_prediction_output(doc_metadata: Any, response: OCRProcessResponse) -> Dict[str, Any]:
    """Formats a clean, comprehensive response dictionary for external project consumption."""
    ocr_result = response.ocr_result
    extraction = response.extraction_result

    # Format fields dictionary
    fields_dict = {}
    if extraction and extraction.fields:
        for k, v in extraction.fields.items():
            fields_dict[k] = {
                "field_name": getattr(v, "field_name", k),
                "value": getattr(v, "value", str(v)),
                "confidence": getattr(v, "confidence", 1.0),
                "page_number": getattr(v, "page_number", 1)
            }

    # Format key-values
    kv_list = []
    if extraction and extraction.key_values:
        for kv in extraction.key_values:
            kv_list.append({
                "key": getattr(kv, "key", ""),
                "value": getattr(kv, "value", ""),
                "confidence": getattr(kv, "confidence", 1.0)
            })

    # Format tables
    tables_list = []
    for p in ocr_result.pages:
        for t in p.tables:
            table_rows = []
            for row in t.rows:
                table_rows.append([getattr(c, "text", str(c)) for c in row.cells])
            tables_list.append({
                "page_number": getattr(t, "page_number", p.page_number),
                "rows": table_rows
            })

    # Format pages with line bounding boxes
    pages_list = []
    for p in ocr_result.pages:
        lines_data = []
        for l in p.lines:
            bbox_dict = l.bbox.model_dump() if hasattr(l.bbox, "model_dump") else l.bbox
            lines_data.append({
                "text": l.text,
                "confidence": l.confidence,
                "bbox": bbox_dict
            })
        pages_list.append({
            "page_number": p.page_number,
            "full_text": p.full_text,
            "line_count": len(p.lines),
            "lines": lines_data
        })

    full_raw_text = "\n".join(p.full_text for p in ocr_result.pages)

    return {
        "status": "success",
        "model_name": "custom-ocr-v1",
        "doc_id": doc_metadata.doc_id,
        "filename": doc_metadata.original_filename,
        "page_count": doc_metadata.page_count,
        "raw_text": full_raw_text,
        "overall_confidence": round(ocr_result.overall_confidence, 3),
        "total_words": ocr_result.total_words,
        "processing_time_total_ms": response.processing_time_total_ms,
        "fields": fields_dict,
        "key_values": kv_list,
        "tables": tables_list,
        "pages": pages_list
    }

@router.post("/predict", summary="Direct Custom OCR Model Prediction (File Upload)")
async def predict_custom_ocr(
    file: UploadFile = File(..., description="Image (.png, .jpg, .jpeg, .tiff, .webp) or PDF document"),
    engine: str = Form("auto", description="OCR engine: auto, rapidocr, custom_model, tesseract, builtin_cv"),
    extract_fields: bool = Form(True, description="Extract structured document entities"),
    extract_tables: bool = Form(True, description="Extract table rows and columns"),
    document_type_hint: Optional[str] = Form(None, description="Hint: invoice, receipt, id_card, generic"),
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    Direct 1-Call Custom OCR Model Inference:
    - Accepts file upload (Image or PDF)
    - Authenticated with your Custom Model API Key
    - Runs complete end-to-end OCR pipeline (Preprocessing, Text Detection, Recognition, Tables, Fields)
    - Returns structured JSON for immediate consumption in Flutter, Python, or Web projects.
    """
    try:
        # Step 1: Upload & convert document
        doc_metadata = await document_service.process_uploaded_file(file)

        # Step 2: Run Custom OCR Pipeline
        process_req = OCRProcessRequest(
            doc_id=doc_metadata.doc_id,
            engine=engine,
            extract_fields=extract_fields,
            extract_tables=extract_tables,
            document_type_hint=document_type_hint
        )
        response = ocr_service.process_document(process_req)

        return _format_prediction_output(doc_metadata, response)

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom OCR model failure: {str(e)}"
        )

@router.post("/predict-base64", summary="Direct Custom OCR Model Prediction (Base64 JSON)")
async def predict_custom_ocr_base64(
    request: OCRBase64Request,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    Direct 1-Call Custom OCR Model Inference from Base64 Image:
    - Useful for mobile cameras, Flutter, web canvas, and microservices
    - Authenticated with your Custom Model API Key
    - Returns structured JSON with lines, bounding boxes, fields, and tables.
    """
    try:
        raw_base64 = request.image_base64
        # Strip data URL prefix if present
        if "," in raw_base64:
            raw_base64 = raw_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(raw_base64)
        filename = request.filename or "image.png"

        # Create mock UploadFile
        mock_file = UploadFile(
            filename=filename,
            file=io.BytesIO(image_bytes)
        )

        doc_metadata = await document_service.process_uploaded_file(mock_file)

        process_req = OCRProcessRequest(
            doc_id=doc_metadata.doc_id,
            engine=request.engine,
            extract_fields=request.extract_fields,
            extract_tables=request.extract_tables,
            document_type_hint=request.document_type_hint
        )
        response = ocr_service.process_document(process_req)

        return _format_prediction_output(doc_metadata, response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Base64 Custom OCR model failure: {str(e)}"
        )

@router.post("/process", response_model=OCRProcessResponse, summary="Execute OCR & Field Extraction on Stored Document")
async def run_ocr_pipeline(
    request: OCRProcessRequest,
    key_record: Optional[APIKeyRecord] = Depends(optional_custom_api_key)
):
    """
    Executes the OCR pipeline on a previously uploaded doc_id.
    Accepts Custom Model API key or executes for internal dashboard.
    """
    try:
        response = ocr_service.process_document(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failure: {str(e)}"
        )

@router.post("/preprocess-only", response_model=PreprocessingResponse, summary="Run and preview Preprocessing steps")
async def run_preprocessing_only(request: OCRProcessRequest):
    """
    Runs only the 7-stage image preprocessing pipeline and returns intermediate images.
    """
    doc = document_service.get_document(request.doc_id)
    if not doc or not doc.pages:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document or pages not found.")

    first_page = doc.pages[0]
    img = load_image(first_page.image_path)
    doc_out = settings.OUTPUT_DIR / request.doc_id

    preprocessor = PreprocessingService()
    _, skew_angle, steps = preprocessor.preprocess(
        image=img,
        output_dir=doc_out,
        page_num=1,
        options=request.preprocessing
    )

    return PreprocessingResponse(
        doc_id=request.doc_id,
        page_number=1,
        skew_angle=skew_angle,
        steps=steps
    )

@router.get("/model-info", summary="Get Custom OCR Model Architecture Info")
async def get_model_info():
    """
    Returns custom OCR model specifications, supported engines, and permanently free offline status.
    """
    rec = CustomRecognizer()
    return {
        "model_name": "custom-ocr-v1",
        "active_engine": rec.active_engine,
        "cost": "100% Permanently Free",
        "hosting": "Self-Hosted & Offline Capable (Zero Third-Party Cloud APIs)",
        "capabilities": [
            "End-to-End Deep Learning Text Detection & Recognition",
            "7-Stage CLAHE, Deskew & Adaptive Preprocessing",
            "Table Structure Detection & Cell Parsing",
            "Spatial Key-Value Entity Extraction",
            "Direct 1-Step REST API (/api/ocr/predict)"
        ]
    }
