import pytest
import numpy as np
import cv2
from app.services.extraction_service import ExtractionService
from app.services.table_service import TableService
from app.models.ocr_result import PageOCRResult, LineBox, WordBox, BoundingBox

def test_extraction_rule_based_fields():
    extractor = ExtractionService()

    # Construct mock PageOCRResult with known invoice tokens
    words = [
        WordBox(text="TAX", confidence=0.98, bbox=BoundingBox.from_coords(50, 50, 60, 20)),
        WordBox(text="INVOICE", confidence=0.98, bbox=BoundingBox.from_coords(120, 50, 90, 20)),
        WordBox(text="Invoice", confidence=0.95, bbox=BoundingBox.from_coords(50, 100, 70, 20)),
        WordBox(text="Number:", confidence=0.95, bbox=BoundingBox.from_coords(125, 100, 80, 20)),
        WordBox(text="INV-9942", confidence=0.96, bbox=BoundingBox.from_coords(210, 100, 90, 20)),
        WordBox(text="Invoice", confidence=0.95, bbox=BoundingBox.from_coords(50, 140, 70, 20)),
        WordBox(text="Date:", confidence=0.95, bbox=BoundingBox.from_coords(125, 140, 60, 20)),
        WordBox(text="2026-09-10", confidence=0.95, bbox=BoundingBox.from_coords(190, 140, 110, 20)),
        WordBox(text="Total:", confidence=0.97, bbox=BoundingBox.from_coords(50, 200, 50, 20)),
        WordBox(text="$", confidence=0.99, bbox=BoundingBox.from_coords(110, 200, 15, 20)),
        WordBox(text="1,250.00", confidence=0.95, bbox=BoundingBox.from_coords(130, 200, 80, 20)),
        WordBox(text="billing@enterprise.io", confidence=0.99, bbox=BoundingBox.from_coords(50, 260, 200, 20)),
        WordBox(text="+1-800-555-0199", confidence=0.95, bbox=BoundingBox.from_coords(50, 300, 150, 20)),
    ]

    lines = [
        LineBox(line_number=1, text="TAX INVOICE", confidence=0.98, bbox=BoundingBox.from_coords(50, 50, 160, 20), words=words[0:2]),
        LineBox(line_number=2, text="Invoice Number: INV-9942", confidence=0.95, bbox=BoundingBox.from_coords(50, 100, 250, 20), words=words[2:5]),
        LineBox(line_number=3, text="Invoice Date: 2026-09-10", confidence=0.95, bbox=BoundingBox.from_coords(50, 140, 250, 20), words=words[5:8]),
        LineBox(line_number=4, text="Total: $ 1,250.00", confidence=0.96, bbox=BoundingBox.from_coords(50, 200, 160, 20), words=words[8:11]),
        LineBox(line_number=5, text="Email: billing@enterprise.io", confidence=0.99, bbox=BoundingBox.from_coords(50, 260, 250, 20), words=[words[11]]),
        LineBox(line_number=6, text="Phone: +1-800-555-0199", confidence=0.95, bbox=BoundingBox.from_coords(50, 300, 200, 20), words=[words[12]]),
    ]

    page = PageOCRResult(
        page_number=1,
        full_text="\n".join([l.text for l in lines]),
        lines=lines,
        words=words,
        tables=[],
        average_confidence=0.96
    )

    result = extractor.extract(doc_id="test_doc", pages=[page])

    assert result.document_type == "invoice"
    assert "invoice_number" in result.fields
    assert result.fields["invoice_number"].value == "INV-9942"
    assert "invoice_date" in result.fields
    assert result.fields["invoice_date"].value == "2026-09-10"
    assert "total_amount" in result.fields
    assert result.fields["total_amount"].value == 1250.00
    assert "email" in result.fields
    assert result.fields["email"].value == "billing@enterprise.io"
    assert "phone" in result.fields
    assert "+1-800-555-0199" in result.fields["phone"].value
    assert result.summary is not None
    assert "INV-9942" in result.summary

def test_table_service_detection():
    service = TableService()
    # Create an image containing an explicit grid table
    img = np.ones((500, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 100), (550, 350), (0, 0, 0), 2)
    # Horizontal lines
    cv2.line(img, (50, 180), (550, 180), (0, 0, 0), 2)
    cv2.line(img, (50, 260), (550, 260), (0, 0, 0), 2)
    # Vertical lines
    cv2.line(img, (200, 100), (200, 350), (0, 0, 0), 2)
    cv2.line(img, (400, 100), (400, 350), (0, 0, 0), 2)

    words = [
        WordBox(text="Item", confidence=0.9, bbox=BoundingBox.from_coords(70, 130, 40, 20)),
        WordBox(text="Qty", confidence=0.9, bbox=BoundingBox.from_coords(230, 130, 30, 20)),
        WordBox(text="Price", confidence=0.9, bbox=BoundingBox.from_coords(430, 130, 40, 20)),
    ]

    tables = service.detect_tables(img, words)
    assert len(tables) >= 1
    assert tables[0].num_rows >= 2
    assert tables[0].num_cols >= 2
