from typing import List
from app.models.ocr_result import LineBox, WordBox, PageOCRResult, DocumentOCRResult
from app.utils.text_utils import clean_ocr_text, correct_numeric_ocr_errors

class OCRPostProcessor:
    """
    Step 4 — OCR Post Processing
    - Cleans raw text
    - Corrects common OCR formatting and character misinterpretations
    - Normalizes coordinates
    - Computes aggregate confidence scores
    """

    def clean_word(self, word: WordBox) -> WordBox:
        raw_text = word.text.strip()
        # If the word resembles a number/amount, correct letter confusion
        if any(c.isdigit() for c in raw_text):
            cleaned_text = correct_numeric_ocr_errors(raw_text)
        else:
            cleaned_text = raw_text

        cleaned_text = clean_ocr_text(cleaned_text)
        return WordBox(
            text=cleaned_text,
            confidence=word.confidence,
            bbox=word.bbox
        )

    def process_page(
        self,
        page_number: int,
        lines: List[LineBox],
        words: List[WordBox],
        img_w: int,
        img_h: int,
        skew_angle: float = 0.0,
        processing_time_ms: float = 0.0
    ) -> PageOCRResult:
        """Post-processes recognized words and lines for a single document page."""
        cleaned_words: List[WordBox] = []
        for w in words:
            cleaned = self.clean_word(w)
            if cleaned.text:
                cleaned_words.append(cleaned)

        cleaned_lines: List[LineBox] = []
        for line in lines:
            line_words = [self.clean_word(w) for w in line.words if w.text]
            line_text = clean_ocr_text(line.text)
            if line_text:
                cleaned_lines.append(LineBox(
                    line_number=line.line_number,
                    text=line_text,
                    confidence=line.confidence,
                    bbox=line.bbox,
                    words=line_words
                ))

        # Build full text representation separated by newlines
        full_text = "\n".join([line.text for line in cleaned_lines])

        # Compute page average confidence
        if cleaned_words:
            avg_conf = sum(w.confidence for w in cleaned_words) / len(cleaned_words)
        else:
            avg_conf = 0.0

        return PageOCRResult(
            page_number=page_number,
            full_text=full_text,
            lines=cleaned_lines,
            words=cleaned_words,
            tables=[],  # Populated later by table extraction service
            average_confidence=round(avg_conf, 3),
            skew_angle=round(skew_angle, 2),
            image_width=img_w,
            image_height=img_h,
            processing_time_ms=round(processing_time_ms, 2)
        )

    def aggregate_document(self, doc_id: str, pages: List[PageOCRResult], engine_used: str = "custom_ocr") -> DocumentOCRResult:
        """Aggregates multiple PageOCRResults into a single DocumentOCRResult."""
        full_text = "\n\n--- PAGE BREAK ---\n\n".join([p.full_text for p in pages if p.full_text])
        total_words = sum(len(p.words) for p in pages)

        if pages:
            overall_conf = sum(p.average_confidence for p in pages) / len(pages)
        else:
            overall_conf = 0.0

        return DocumentOCRResult(
            doc_id=doc_id,
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            total_words=total_words,
            overall_confidence=round(overall_conf, 3),
            engine_used=engine_used
        )
