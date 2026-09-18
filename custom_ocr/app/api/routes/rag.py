from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import verify_custom_api_key
from app.services.api_key_service import APIKeyRecord
from app.services.custom_rag_service import (
    custom_rag_service, RAGQueryResponse, RAGChatResponse, RAGChatMessage
)

router = APIRouter(prefix="/rag", tags=["Custom Model RAG Engine"])

class IngestTextRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document ID")
    text: str = Field(..., description="Raw text content to index")
    title: Optional[str] = Field(None, description="Human-readable document title")
    page_number: int = Field(1, description="Page number for citations")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks_created: int
    message: str

class QueryRequest(BaseModel):
    doc_id: str = Field(..., description="ID of previously indexed document")
    query: str = Field(..., description="Question or prompt to answer")
    top_k: int = Field(4, ge=1, le=10, description="Number of source passages to retrieve")

class ChatRequest(BaseModel):
    doc_id: str = Field(..., description="ID of previously indexed document")
    messages: List[RAGChatMessage] = Field(..., description="Conversational message history")

@router.post("/ingest", response_model=IngestResponse, summary="Ingest Document Text into Local RAG Vector Store")
async def ingest_document(
    request: IngestTextRequest,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    Indexes raw document text or extracted OCR text into the 100% free local vector store.
    Secured with your Custom Model API Key.
    """
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty."
        )

    count = custom_rag_service.ingest_text(
        doc_id=request.doc_id,
        text=request.text,
        title=request.title,
        page_number=request.page_number,
        metadata=request.metadata
    )

    return IngestResponse(
        status="success",
        doc_id=request.doc_id,
        chunks_created=count,
        message=f"Successfully indexed {count} semantic chunks into local RAG store."
    )

@router.post("/query", response_model=RAGQueryResponse, summary="Query Document with Grounded Citations")
async def query_document(
    request: QueryRequest,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    Answers questions grounded in the indexed document with page citations and confidence scores.
    Secured with your Custom Model API Key.
    """
    response = custom_rag_service.query(
        doc_id=request.doc_id,
        query=request.query,
        top_k=request.top_k
    )
    return response

@router.post("/chat", response_model=RAGChatResponse, summary="Multi-turn Conversational Chat with Document")
async def chat_document(
    request: ChatRequest,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """
    Maintains conversational multi-turn dialogue grounded in the specified document context.
    Secured with your Custom Model API Key.
    """
    response = custom_rag_service.chat(
        doc_id=request.doc_id,
        messages=request.messages
    )
    return response

@router.get("/documents", summary="List Indexed RAG Documents")
async def list_documents(
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """Lists all documents currently indexed in the local RAG engine."""
    return {
        "status": "success",
        "total_documents": len(custom_rag_service.list_documents()),
        "documents": custom_rag_service.list_documents()
    }

@router.delete("/documents/{doc_id}", summary="Delete Document from RAG Vector Store")
async def delete_document(
    doc_id: str,
    key_record: APIKeyRecord = Depends(verify_custom_api_key)
):
    """Deletes document chunks and vector index from local storage."""
    deleted = custom_rag_service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in RAG store."
        )
    return {"status": "success", "message": f"Document '{doc_id}' deleted."}

@router.get("/health", summary="Check Custom Model Engine Health")
async def rag_health():
    """Returns local custom model engine health and status."""
    return {
        "status": "online",
        "model_name": "custom-rag-doc-v1",
        "embedding_model": "custom-doc-embed-v1",
        "cost": "100% Permanently Free",
        "hosting": "Local & Offline Capable (Zero Third-Party APIs)",
        "indexed_documents": len(custom_rag_service.list_documents())
    }
