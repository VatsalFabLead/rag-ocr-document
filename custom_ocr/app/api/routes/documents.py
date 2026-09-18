import csv
import io
import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, Response

from app.core.config import settings
from app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.get("/{doc_id}", summary="Get Document Details & Results")
async def get_document_details(doc_id: str):
    """Retrieves document processing status and results."""
    doc = document_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    doc_dir = settings.OUTPUT_DIR / doc_id

    ocr_result = None
    ocr_file = doc_dir / "ocr_result.json"
    if ocr_file.exists():
        with open(ocr_file, "r", encoding="utf-8") as f:
            ocr_result = json.load(f)

    structured_output = None
    struct_file = doc_dir / "structured_output.json"
    if struct_file.exists():
        with open(struct_file, "r", encoding="utf-8") as f:
            structured_output = json.load(f)

    return {
        "metadata": doc,
        "ocr_result": ocr_result,
        "structured_output": structured_output
    }

@router.get("/{doc_id}/export/{export_format}", summary="Export Extracted Data")
async def export_document_data(doc_id: str, export_format: str):
    """
    Exports structured data in 'json' or 'csv' format.
    """
    doc_dir = settings.OUTPUT_DIR / doc_id
    struct_file = doc_dir / "structured_output.json"

    if not struct_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Structured extraction output not found for this document."
        )

    with open(struct_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    format_lower = export_format.lower()

    if format_lower == "json":
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": f"attachment; filename=extracted_{doc_id}.json"}
        )
    elif format_lower == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Category", "Field / Key", "Value", "Confidence", "Extraction Method"])

        # Write Extracted Fields
        fields = data.get("fields", {})
        for field_name, f_obj in fields.items():
            writer.writerow([
                "Primary Field",
                f_obj.get("field_name", field_name),
                f_obj.get("value", ""),
                f_obj.get("confidence", 0.0),
                f_obj.get("extraction_method", "")
            ])

        # Write Key-Value Pairs
        key_vals = data.get("key_values", [])
        for kv in key_vals:
            writer.writerow([
                "Key-Value Pair",
                kv.get("key", ""),
                kv.get("value", ""),
                kv.get("confidence", 0.0),
                "spatial_kv"
            ])

        csv_content = output.getvalue()
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=extracted_{doc_id}.csv"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{export_format}'. Supported formats: json, csv"
        )
