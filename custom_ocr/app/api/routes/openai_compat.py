import time
import uuid
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import verify_custom_api_key
from app.services.api_key_service import APIKeyRecord
from app.services.custom_embedding_model import custom_embedding_model
from app.services.custom_rag_service import custom_rag_service

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible Endpoints"])

class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="custom-rag-doc-v1")
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    doc_id: Optional[str] = Field(None, description="Optional doc_id to ground in specific document")

class EmbeddingRequest(BaseModel):
    model: str = Field(default="custom-doc-embed-v1")
    input: Union[str, List[str]]

@router.get("/models", summary="List Custom Models (OpenAI Compatible)")
async def list_models(
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """Returns available local models in standard OpenAI format."""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "custom-rag-doc-v1",
                "object": "model",
                "created": now,
                "owned_by": "custom-local-engine",
                "permission": [],
                "root": "custom-rag-doc-v1",
                "parent": None
            },
            {
                "id": "custom-doc-embed-v1",
                "object": "model",
                "created": now,
                "owned_by": "custom-local-engine",
                "permission": [],
                "root": "custom-doc-embed-v1",
                "parent": None
            }
        ]
    }

@router.post("/chat/completions", summary="Create Chat Completion (OpenAI Compatible)")
async def create_chat_completion(
    request: ChatCompletionRequest,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    OpenAI-compatible Chat Completion endpoint.
    Allows standard OpenAI SDKs (Python, Node.js, Flutter dart_openai, LangChain)
    to query your 100% free custom model using your custom API key.
    """
    user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    if not user_message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No user message provided.")

    # Check if doc_id is provided in request or can be inferred
    target_doc_id = request.doc_id
    if not target_doc_id:
        # Check system message for doc_id tag, e.g. "doc_id: xyz"
        for m in request.messages:
            if m.role == "system" and "doc_id:" in m.content.lower():
                parts = m.content.split("doc_id:")
                if len(parts) > 1:
                    target_doc_id = parts[1].strip().split()[0]
                    break

    # If document specified, answer with grounded RAG
    if target_doc_id:
        rag_res = custom_rag_service.query(target_doc_id, user_message, top_k=3)
        answer_text = rag_res.answer
    else:
        # Fallback to general conversational or search latest doc
        docs = custom_rag_service.list_documents()
        if docs:
            latest_doc = docs[-1]["doc_id"]
            rag_res = custom_rag_service.query(latest_doc, user_message, top_k=3)
            answer_text = rag_res.answer
        else:
            answer_text = f"Custom Model response to: '{user_message}'. No documents are currently loaded. Ingest a document or run OCR to start querying document contents."

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    completion_tokens = len(answer_text.split())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

@router.post("/embeddings", summary="Create Vector Embeddings (OpenAI Compatible)")
async def create_embeddings(
    request: EmbeddingRequest,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    OpenAI-compatible Vector Embedding endpoint.
    Computes 384-dimensional dense vectors using local custom embedding model.
    """
    inputs = [request.input] if isinstance(request.input, str) else request.input
    data = []
    total_tokens = 0

    for idx, text in enumerate(inputs):
        vec = custom_embedding_model.embed_text(text)
        total_tokens += len(text.split())
        data.append({
            "object": "embedding",
            "index": idx,
            "embedding": vec
        })

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }
