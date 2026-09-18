from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.models.ocr_result import DocumentOCRResult
from app.models.extracted_field import DocumentExtractionResult

class PreprocessingOptions(BaseModel):
    enable_resize: bool = True
    enable_grayscale: bool = True
    enable_denoise: bool = True
    enable_contrast_enhancement: bool = True
    enable_deskew: bool = True
    enable_thresholding: bool = True
    threshold_method: str = "adaptive"  # adaptive or otsu
    target_dpi: int = 300

class OCRProcessRequest(BaseModel):
    doc_id: str
    engine: str = "auto"  # auto, custom_cv, tesseract, easyocr
    preprocessing: Optional[PreprocessingOptions] = Field(default_factory=PreprocessingOptions)
    extract_fields: bool = True
    extract_tables: bool = True
    document_type_hint: Optional[str] = None  # invoice, receipt, generic

class PreprocessingStepImage(BaseModel):
    step_key: str
    title: str
    description: str
    image_url: str

class PreprocessingResponse(BaseModel):
    doc_id: str
    page_number: int
    skew_angle: float
    steps: List[PreprocessingStepImage]

class OCRProcessResponse(BaseModel):
    doc_id: str
    status: str = "completed"
    ocr_result: DocumentOCRResult
    extraction_result: Optional[DocumentExtractionResult] = None
    preprocessing_steps: Optional[List[PreprocessingStepImage]] = None
    processing_time_total_ms: float = 0.0
