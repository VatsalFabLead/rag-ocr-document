from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.models.ocr_result import BoundingBox, TableResult

class FieldType(str, Enum):
    INVOICE_NUMBER = "invoice_number"
    INVOICE_DATE = "invoice_date"
    DUE_DATE = "due_date"
    TOTAL_AMOUNT = "total_amount"
    SUBTOTAL = "subtotal"
    TAX_AMOUNT = "tax_amount"
    CURRENCY = "currency"
    VENDOR_NAME = "vendor_name"
    CUSTOMER_NAME = "customer_name"
    EMAIL = "email"
    PHONE = "phone"
    ID_NUMBER = "id_number"
    BALANCE_AMOUNT = "balance_amount"
    BANK_NAME = "bank_name"
    ACCOUNT_NUMBER = "account_number"
    IFSC_CODE = "ifsc_code"
    FORM_FIELD = "form_field"
    GENERIC_KEY_VALUE = "generic_key_value"
    CUSTOM = "custom"

class ExtractedField(BaseModel):
    field_name: str
    value: Any
    raw_text: str
    field_type: FieldType
    confidence: float
    page_number: int = 1
    bbox: Optional[BoundingBox] = None
    extraction_method: str = "rule_based"  # rule_based, spatial_kv, nlp, llm

class KeyValuePair(BaseModel):
    key: str
    value: str
    confidence: float = 0.85
    page_number: int = 1
    key_bbox: Optional[BoundingBox] = None
    value_bbox: Optional[BoundingBox] = None

class DocumentExtractionResult(BaseModel):
    doc_id: str
    document_type: str = "generic"
    fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    key_values: List[KeyValuePair] = Field(default_factory=list)
    tables: List[TableResult] = Field(default_factory=list)
    summary: Optional[str] = None
    raw_text: str = ""
    overall_confidence: float = 0.0
