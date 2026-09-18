"""
100% Permanently Free Custom OCR Model Client for Python.
Zero Third-Party APIs | Local Inference | Structured Text & Entity Extraction
"""

import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import httpx

class CustomOCRClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "sk-custom-ocr-master-v1.1",
        timeout: float = 60.0
    ):
        """
        Initializes the client with your Custom OCR Model server URL and API key.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}"
        }

    def process_file(
        self,
        file_path: str,
        engine: str = "auto",
        extract_fields: bool = True,
        extract_tables: bool = True,
        doc_type_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs Custom OCR Model inference on a local image or PDF file.
        Returns full recognized text, line bounding boxes, extracted fields, and tables.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        url = f"{self.base_url}/api/ocr/predict"
        data = {
            "engine": engine,
            "extract_fields": str(extract_fields).lower(),
            "extract_tables": str(extract_tables).lower(),
        }
        if doc_type_hint:
            data["document_type_hint"] = doc_type_hint

        with open(path, "rb") as f:
            files = {"file": (path.name, f)}
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=self.headers, data=data, files=files)
                resp.raise_for_status()
                return resp.json()

    def process_base64(
        self,
        base64_image: str,
        filename: str = "image.png",
        engine: str = "auto",
        extract_fields: bool = True,
        extract_tables: bool = True,
        doc_type_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs Custom OCR Model inference on a base64 encoded image string.
        """
        url = f"{self.base_url}/api/ocr/predict-base64"
        payload = {
            "image_base64": base64_image,
            "filename": filename,
            "engine": engine,
            "extract_fields": extract_fields,
            "extract_tables": extract_tables,
            "document_type_hint": doc_type_hint
        }
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Returns Custom OCR Model specifications and engine information."""
        url = f"{self.base_url}/api/ocr/model-info"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

if __name__ == "__main__":
    client = CustomOCRClient()
    print("Fetching Custom OCR Model info...")
    try:
        info = client.get_model_info()
        print("Model Info:", info)
    except Exception as e:
        print(f"Could not connect to server: {e}")
