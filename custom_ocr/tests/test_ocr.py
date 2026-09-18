import io
import pytest
import numpy as np
import cv2
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

def create_sample_png_bytes() -> bytes:
    img = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(img, "INVOICE # 98765", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "TOTAL: $ 199.99", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    success, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_upload_and_pipeline_flow():
    png_bytes = create_sample_png_bytes()
    files = {"file": ("test_invoice.png", io.BytesIO(png_bytes), "image/png")}

    # 1. Step 1: Upload
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert "doc_id" in upload_data
    doc_id = upload_data["doc_id"]
    assert upload_data["page_count"] == 1

    # 2. Step 2: Preprocess Only preview
    prep_res = client.post("/api/ocr/preprocess-only", json={"doc_id": doc_id})
    assert prep_res.status_code == 200
    prep_data = prep_res.json()
    assert len(prep_data["steps"]) >= 7

    # 3. Step 3-5: Run Full OCR and Extraction Pipeline
    ocr_res = client.post("/api/ocr/process", json={
        "doc_id": doc_id,
        "engine": "custom_cv",
        "extract_fields": True,
        "extract_tables": True
    })
    assert ocr_res.status_code == 200
    ocr_data = ocr_res.json()
    assert ocr_data["status"] == "completed"
    assert "ocr_result" in ocr_data
    assert "extraction_result" in ocr_data
    assert ocr_data["extraction_result"] is not None
    assert len(ocr_data["extraction_result"]["fields"]) > 0

    # 4. Step 6: Export JSON and CSV
    export_json = client.get(f"/api/documents/{doc_id}/export/json")
    assert export_json.status_code == 200

    export_csv = client.get(f"/api/documents/{doc_id}/export/csv")
    assert export_csv.status_code == 200
    assert "Category,Field / Key,Value" in export_csv.text
