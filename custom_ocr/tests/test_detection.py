import pytest
import numpy as np
import cv2
from pathlib import Path

from app.ocr.detector import CustomDetector
from app.services.preprocessing_service import PreprocessingService
from app.api.schemas.ocr_schema import PreprocessingOptions

def create_synthetic_text_image() -> np.ndarray:
    """Creates a clean synthetic white image with black text blocks for testing."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Draw sample text lines
    cv2.putText(img, "INVOICE NUMBER: INV-10023", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "DATE: 2026-09-10", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "TOTAL DUE: $ 450.00", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img

def test_preprocessing_pipeline(tmp_path: Path):
    service = PreprocessingService()
    img = create_synthetic_text_image()
    opts = PreprocessingOptions(enable_deskew=True, enable_contrast_enhancement=True)

    clean_img, skew_angle, steps = service.preprocess(
        image=img,
        output_dir=tmp_path,
        page_num=1,
        options=opts
    )

    assert clean_img is not None
    assert clean_img.shape[0] > 0 and clean_img.shape[1] > 0
    assert isinstance(skew_angle, float)
    assert len(steps) >= 7  # All 7 stages documented

    # Verify intermediate image files were created
    for step in steps:
        filename = step.image_url.split('/')[-1]
        assert (tmp_path / filename).exists()

def test_custom_text_detector():
    detector = CustomDetector(min_area=30)
    img = create_synthetic_text_image()

    boxes = detector.detect(img)
    assert len(boxes) >= 3  # Detected the text blocks

    # Verify coordinates are valid and positive
    for b in boxes:
        assert b.x >= 0 and b.y >= 0
        assert b.width > 0 and b.height > 0
        assert 0.0 <= b.x_norm <= 1.0
        assert 0.0 <= b.y_norm <= 1.0

    # Verify reading order: First box should be above the last box
    assert boxes[0].y <= boxes[-1].y
