import time
import json
from pathlib import Path
from typing import Optional, List, Tuple
import cv2

from app.core.config import settings
from app.models.document import DocumentStatus
from app.models.ocr_result import PageOCRResult, DocumentOCRResult
from app.models.extracted_field import DocumentExtractionResult
from app.api.schemas.ocr_schema import (
    OCRProcessRequest, OCRProcessResponse, PreprocessingStepImage
)
from app.services.document_service import document_service
from app.services.preprocessing_service import PreprocessingService
from app.services.table_service import TableService
from app.services.extraction_service import ExtractionService
from app.ocr.detector import CustomDetector
from app.ocr.recognizer import CustomRecognizer
from app.ocr.tokenizer import DocumentTokenizer
from app.ocr.postprocessor import OCRPostProcessor
from app.utils.image_utils import load_image, save_image, draw_bounding_boxes

class OCRService:
    """
    Master OCR Orchestration Service:
    Coordinates Preprocessing -> Detection -> Recognition -> Tokenization
    -> Table Extraction -> Field Extraction -> Output Export.
    """

    def __init__(self):
        self.preprocessor = PreprocessingService()
        self.detector = CustomDetector()
        self.tokenizer = DocumentTokenizer()
        self.postprocessor = OCRPostProcessor()
        self.table_service = TableService()
        self.extraction_service = ExtractionService()

    def process_document(self, request: OCRProcessRequest) -> OCRProcessResponse:
        start_time = time.time()
        doc_id = request.doc_id

        doc = document_service.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document with ID '{doc_id}' not found.")

        document_service.update_document_status(doc_id, DocumentStatus.PROCESSING)

        doc_output_dir = settings.OUTPUT_DIR / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        recognizer = CustomRecognizer(engine_mode=request.engine)
        page_results: List[PageOCRResult] = []
        all_preprocessing_steps: List[PreprocessingStepImage] = []

        for page in doc.pages:
            page_start = time.time()
            img_bgr = load_image(page.image_path)
            orig_h, orig_w = img_bgr.shape[:2]

            # 1. Step 2: Image Preprocessing Pipeline
            clean_img, skew_angle, steps_meta = self.preprocessor.preprocess(
                image=img_bgr,
                output_dir=doc_output_dir,
                page_num=page.page_number,
                options=request.preprocessing
            )
            all_preprocessing_steps.extend(steps_meta)

            # 2. Step 3: Text Detection
            detected_boxes = self.detector.detect(clean_img)

            # 3. Step 3: Text Recognition
            recognized_words = recognizer.recognize_full_image(img_bgr, detected_boxes)

            # 4. Tokenization (grouping into lines)
            clean_h, clean_w = clean_img.shape[:2]
            line_boxes = self.tokenizer.group_into_lines(recognized_words, img_w=clean_w, img_h=clean_h)

            # 5. Step 4: Post Processing
            page_time_ms = (time.time() - page_start) * 1000
            page_ocr = self.postprocessor.process_page(
                page_number=page.page_number,
                lines=line_boxes,
                words=recognized_words,
                img_w=clean_w,
                img_h=clean_h,
                skew_angle=skew_angle,
                processing_time_ms=page_time_ms
            )

            # 6. Table Extraction
            if request.extract_tables:
                tables = self.table_service.detect_tables(clean_img, recognized_words)
                page_ocr.tables = tables

            page_results.append(page_ocr)

            # Draw visual debug bounding boxes on copy of page image
            annotated_boxes = [
                {
                    'x': w.bbox.x,
                    'y': w.bbox.y,
                    'width': w.bbox.width,
                    'height': w.bbox.height,
                    'text': w.text,
                    'confidence': w.confidence
                }
                for w in recognized_words
            ]
            annotated_img = draw_bounding_boxes(img_bgr, annotated_boxes)
            annotated_path = doc_output_dir / f"page_{page.page_number}_annotated.png"
            save_image(annotated_img, annotated_path)

        # 7. Aggregate full Document OCR Result
        doc_ocr_result = self.postprocessor.aggregate_document(
            doc_id=doc_id,
            pages=page_results,
            engine_used=recognizer.active_engine
        )

        # 8. Step 5: Document Analysis & Automatic Field Extraction
        extraction_result = None
        if request.extract_fields:
            extraction_result = self.extraction_service.extract(
                doc_id=doc_id,
                pages=page_results,
                doc_type_hint=request.document_type_hint
            )

        total_time_ms = (time.time() - start_time) * 1000

        # Save structured outputs to disk
        self._save_results(doc_output_dir, doc_ocr_result, extraction_result)

        # 9. Auto-index into Local Custom RAG Vector Store (100% Free & Local)
        try:
            from app.services.custom_rag_service import custom_rag_service
            custom_rag_service.ingest_ocr_result(
                doc_id=doc_id,
                ocr_result=doc_ocr_result,
                extraction_result=extraction_result,
                title=doc.original_filename if doc else doc_id
            )
        except Exception as e:
            print(f"[OCRService] Warning: Failed to auto-index into RAG store: {e}")

        document_service.update_document_status(doc_id, DocumentStatus.COMPLETED)

        return OCRProcessResponse(
            doc_id=doc_id,
            status="completed",
            ocr_result=doc_ocr_result,
            extraction_result=extraction_result,
            preprocessing_steps=all_preprocessing_steps,
            processing_time_total_ms=round(total_time_ms, 2)
        )

    def _save_results(
        self,
        output_dir: Path,
        ocr_result: DocumentOCRResult,
        extraction_result: Optional[DocumentExtractionResult]
    ):
        with open(output_dir / "ocr_result.json", "w", encoding="utf-8") as f:
            f.write(ocr_result.model_dump_json(indent=2))

        if extraction_result:
            with open(output_dir / "structured_output.json", "w", encoding="utf-8") as f:
                f.write(extraction_result.model_dump_json(indent=2))

ocr_service = OCRService()
