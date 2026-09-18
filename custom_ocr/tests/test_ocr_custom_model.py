import cv2
import numpy as np
import base64
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services.api_key_service import api_key_service

client = TestClient(app)

def _create_test_image_bytes():
    """Generates a synthetic invoice image for OCR testing."""
    img = np.ones((400, 800, 3), dtype=np.uint8) * 255
    # Draw dark text
    cv2.putText(img, "GLOBAL TECH SERVICES TAX INVOICE", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Invoice Number: INV-2026-7788", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img, "Date: 15-Oct-2026", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img, "Total Amount Due: $3,250.00 USD", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(img, "Thank you for your business", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
    
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()

def test_ocr_model_info():
    """Verifies that the model info endpoint reports custom OCR specs and free status."""
    resp = client.get("/api/ocr/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "custom-ocr-v1"
    assert "Permanently Free" in data["cost"]
    assert "capabilities" in data

def test_ocr_predict_auth_protection():
    """Verifies that the Custom OCR Model endpoint enforces API key authentication."""
    img_bytes = _create_test_image_bytes()

    # 1. No key provided
    resp_no_key = client.post(
        "/api/ocr/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert resp_no_key.status_code == 401

    # 2. Fake / invalid key
    resp_bad_key = client.post(
        "/api/ocr/predict",
        headers={"Authorization": "Bearer sk-invalid-key-xyz"},
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert resp_bad_key.status_code == 401

def test_ocr_direct_predict_success():
    """Verifies 1-step direct OCR prediction using custom API key."""
    img_bytes = _create_test_image_bytes()
    master_key = api_key_service.get_primary_key()

    resp = client.post(
        "/api/ocr/predict",
        headers={"Authorization": f"Bearer {master_key}"},
        files={"file": ("sample_invoice.png", img_bytes, "image/png")},
        data={"extract_fields": "true", "extract_tables": "true"}
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "success"
    assert data["model_name"] == "custom-ocr-v1"
    assert data["page_count"] == 1
    assert len(data["pages"]) > 0
    assert "raw_text" in data
    assert "fields" in data
    assert len(data["raw_text"]) > 0
    assert "GLOBAL" in data["raw_text"]

def test_ocr_predict_base64():
    """Verifies direct OCR prediction with Base64 JSON payload."""
    img_bytes = _create_test_image_bytes()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    master_key = api_key_service.get_primary_key()

    resp = client.post(
        "/api/ocr/predict-base64",
        headers={"Authorization": f"Bearer {master_key}"},
        json={
            "image_base64": b64_str,
            "filename": "invoice_cam.png",
            "extract_fields": True
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["model_name"] == "custom-ocr-v1"
    assert data["filename"] == "invoice_cam.png"
