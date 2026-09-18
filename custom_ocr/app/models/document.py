from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    ID_CARD = "id_card"
    FORM = "form"
    GENERIC = "generic"

class DocumentPage(BaseModel):
    page_number: int
    image_path: str
    width: int
    height: int
    dpi: int = 300
    preprocessed_images: Dict[str, str] = Field(default_factory=dict)

class DocumentMetadata(BaseModel):
    doc_id: str
    original_filename: str
    saved_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    page_count: int = 1
    document_type: DocumentType = DocumentType.GENERIC
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    pages: List[DocumentPage] = Field(default_factory=list)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
