from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class PageInfo(BaseModel):
    page_number: int
    image_url: str
    width: int
    height: int

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    file_size_bytes: int
    mime_type: str
    page_count: int
    pages: List[PageInfo]
    uploaded_at: datetime
    message: str = "File uploaded and validated successfully."
