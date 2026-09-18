from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    # Normalized coordinates (0.0 to 1.0) for responsive UI rendering
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    w_norm: Optional[float] = None
    h_norm: Optional[float] = None

    @classmethod
    def from_coords(cls, x: int, y: int, width: int, height: int, img_w: int = 1, img_h: int = 1):
        x_norm = round(x / max(img_w, 1), 4)
        y_norm = round(y / max(img_h, 1), 4)
        w_norm = round(width / max(img_w, 1), 4)
        h_norm = round(height / max(img_h, 1), 4)
        return cls(x=x, y=y, width=width, height=height, x_norm=x_norm, y_norm=y_norm, w_norm=w_norm, h_norm=h_norm)

class WordBox(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBox

class LineBox(BaseModel):
    line_number: int
    text: str
    confidence: float
    bbox: BoundingBox
    words: List[WordBox] = Field(default_factory=list)

class TableCell(BaseModel):
    row_idx: int
    col_idx: int
    text: str
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None

class TableResult(BaseModel):
    table_id: str
    bbox: Optional[BoundingBox] = None
    num_rows: int
    num_cols: int
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)

class PageOCRResult(BaseModel):
    page_number: int
    full_text: str
    lines: List[LineBox] = Field(default_factory=list)
    words: List[WordBox] = Field(default_factory=list)
    tables: List[TableResult] = Field(default_factory=list)
    average_confidence: float = 0.0
    skew_angle: float = 0.0
    image_width: int = 0
    image_height: int = 0
    processing_time_ms: float = 0.0

class DocumentOCRResult(BaseModel):
    doc_id: str
    total_pages: int
    pages: List[PageOCRResult] = Field(default_factory=list)
    full_text: str = ""
    total_words: int = 0
    overall_confidence: float = 0.0
    engine_used: str = "custom_ocr"
