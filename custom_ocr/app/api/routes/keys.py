from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.api_key_service import api_key_service

router = APIRouter(prefix="/keys", tags=["Custom Model API Key Manager"])

class CreateKeyRequest(BaseModel):
    name: str = Field("New Project Client", description="Project or client name, e.g., 'Flutter Mobile App', 'Web Dashboard'")

@router.get("", summary="List All Custom Model API Keys")
async def list_keys():
    """Lists all active and revoked API keys with usage statistics."""
    return {
        "status": "success",
        "keys": api_key_service.list_keys()
    }

@router.post("", summary="Generate New Custom Model API Key")
async def create_key(request: CreateKeyRequest):
    """
    Generates a new custom model API key that can be passed to any project or application.
    Zero cost, permanently free, self-hosted.
    """
    record = api_key_service.create_key(name=request.name)
    return {
        "status": "success",
        "message": "Custom model API key generated successfully. Save this key in your project configuration.",
        "key": record.to_public_dict()
    }

@router.delete("/{key_id}", summary="Revoke an API Key")
async def revoke_key(key_id: str):
    """Deactivates an API key so it can no longer be used."""
    success = api_key_service.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID '{key_id}' not found."
        )
    return {
        "status": "success",
        "message": f"API key '{key_id}' revoked."
    }

@router.get("/active-primary", summary="Get Active Primary Key for Instant Setup")
async def get_active_primary():
    """Returns the primary active key string for quick copy-pasting."""
    primary_key = api_key_service.get_primary_key()
    return {
        "status": "success",
        "api_key": primary_key,
        "sample_curl": f'curl -X POST "http://127.0.0.1:8000/api/rag/query" -H "Authorization: Bearer {primary_key}" -H "Content-Type: application/json" -d \'{{"doc_id": "doc_sample", "query": "What is the total amount?"}}\''
    }
