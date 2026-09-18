"""
100% Permanently Free Custom Model RAG Client for Python.
Zero Third-Party APIs | Grounded Citations | Self-Hosted Model Key Auth
"""

import json
from typing import Dict, List, Any, Optional
import httpx

class CustomRAGClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "sk-custom-ocr-master-v1.1",
        timeout: float = 30.0
    ):
        """
        Initializes client with your local custom model endpoint and API key.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def ingest_text(
        self,
        doc_id: str,
        text: str,
        title: Optional[str] = None,
        page_number: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingests raw text or OCR content into the custom vector store."""
        url = f"{self.base_url}/api/rag/ingest"
        payload = {
            "doc_id": doc_id,
            "text": text,
            "title": title or doc_id,
            "page_number": page_number,
            "metadata": metadata or {}
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def query(
        self,
        doc_id: str,
        question: str,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Queries the document with grounded citations and confidence scoring."""
        url = f"{self.base_url}/api/rag/query"
        payload = {
            "doc_id": doc_id,
            "query": question,
            "top_k": top_k
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def chat(
        self,
        doc_id: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Multi-turn conversational dialogue with document grounding."""
        url = f"{self.base_url}/api/rag/chat"
        payload = {
            "doc_id": doc_id,
            "messages": messages
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def create_api_key(self, name: str = "New Project Key") -> Dict[str, Any]:
        """Generates a new custom model API key."""
        url = f"{self.base_url}/api/keys"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json={"name": name})
            resp.raise_for_status()
            return resp.json()

    def list_keys(self) -> Dict[str, Any]:
        """Lists all custom model API keys."""
        url = f"{self.base_url}/api/keys"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()

    def get_openai_client(self):
        """
        Convenience helper to return a standard OpenAI client configured
        to communicate with your local custom model.
        """
        try:
            from openai import OpenAI
            return OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError(
                "The 'openai' package is not installed. Install it with 'pip install openai' "
                "or use standard CustomRAGClient methods."
            )

if __name__ == "__main__":
    # Self-test when executed
    client = CustomRAGClient()
    print("Testing Custom Model Connection...")
    try:
        keys = client.list_keys()
        print(f"Connected! Available keys: {len(keys.get('keys', []))}")
    except Exception as e:
        print(f"Note: Server not running or unreachable at http://127.0.0.1:8000 ({e})")
