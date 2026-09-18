# Custom OCR & Intelligent Document Processing Pipeline

An enterprise-grade, end-to-end Computer Vision & OCR architecture designed for automated document ingestion, image preprocessing, custom text detection, text recognition, OCR post-processing, and structured field extraction (Invoices, Receipts, Forms, and Tabular structures).

---

## 🏛️ System Architecture

```
                    ┌─────────────────────┐
                    │   User Uploads      │
                    │ Image / PDF / Scan  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   File Validation   │
                    │ type / size / pages │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Document Processor  │
                    │ PDF → Images        │
                    │ Image → Preprocess  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Custom OCR Engine │
                    │                     │
                    │ Text Detection      │
                    │        ↓            │
                    │ Text Recognition    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ OCR Post Processing │
                    │                     │
                    │ Clean text          │
                    │ Correct formatting  │
                    │ Coordinates         │
                    │ Confidence scores   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Document Analysis   │
                    │                     │
                    │ Identify fields     │
                    │ Tables              │
                    │ Key-value pairs     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Structured Output   │
                    │                     │
                    │ JSON / DB / API     │
                    └─────────────────────┘
```

---

## 📁 Repository Structure

```
custom_ocr/
├── app/
│   ├── main.py                     # FastAPI application & Web UI mount
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── upload.py           # Upload & validation endpoints
│   │   │   ├── ocr.py              # OCR execution & preprocessing preview
│   │   │   └── documents.py        # Document details & CSV/JSON export
│   │   │
│   │   └── schemas/
│   │       ├── upload_schema.py    # Request/Response schemas for uploads
│   │       └── ocr_schema.py       # OCR options & result models
│   │
│   ├── core/
│   │   ├── config.py               # Pydantic environment configuration
│   │   └── security.py             # Magic byte signature & file safety
│   │
│   ├── services/
│   │   ├── document_service.py     # PDF-to-image converter & lifecycle
│   │   ├── preprocessing_service.py# 7-stage computer vision pipeline
│   │   ├── ocr_service.py          # Master OCR pipeline orchestrator
│   │   ├── extraction_service.py   # Regex & spatial key-value extraction
│   │   └── table_service.py        # CV morphological table & cell extractor
│   │
│   ├── ocr/
│   │   ├── detector.py             # Gradient, Sobel & contour text detector
│   │   ├── recognizer.py           # Pluggable deep learning / CV recognizer
│   │   ├── tokenizer.py            # Reading order & line assembler
│   │   └── postprocessor.py        # Text cleaner, numeric corrector & scoring
│   │
│   ├── models/
│   │   ├── document.py             # Document & Page metadata models
│   │   ├── ocr_result.py           # BoundingBox, WordBox, LineBox, TableCell
│   │   └── extracted_field.py      # ExtractedField & KeyValuePair models
│   │
│   ├── utils/
│   │   ├── image_utils.py          # Safe OpenCV/PIL encoding and drawing
│   │   ├── pdf_utils.py            # High-DPI PDF page rendering
│   │   └── text_utils.py           # Regex patterns & date/currency sanitizers
│   │
│   └── static/                     # Modern interactive frontend
│       ├── index.html              # Dashboard with canvas overlay & inspector
│       ├── css/style.css           # Glassmorphism dark mode styling
│       └── js/app.js               # Reactive client application logic
│
├── models/
│   ├── detection_model/            # Placeholder for custom ONNX / PyTorch weights
│   └── recognition_model/          # Placeholder for custom CRNN / TrOCR weights
│
├── tests/
│   ├── test_ocr.py                 # End-to-end API pipeline integration tests
│   ├── test_detection.py           # Computer vision & preprocessing unit tests
│   └── test_extraction.py          # Field, table & regex extraction unit tests
│
├── uploads/                        # Temporary uploaded file storage
├── outputs/                        # Preprocessing snapshots & structured JSON
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation & guide
```

---

## 🚀 Pipeline Workflow (Steps 1 – 6)

### Step 1 — Ingestion & Document Processor
- Ingests scanned documents, images (PNG, JPG, TIFF, WEBP), and multi-page PDFs.
- Validates file format using **magic byte signatures** and checks maximum size thresholds.
- Converts multi-page PDFs into high-resolution 300 DPI bitmaps via `pypdfium2`.

### Step 2 — 7-Stage Image Preprocessing
```
Original Image → Resize → Grayscale → Noise Removal → Contrast Enhancement (CLAHE) → Deskew → Thresholding → Clean Image
```
1. **Resize & DPI Normalization**: Rescales document while strictly preserving aspect ratio.
2. **Grayscale**: Collapses color channels to isolate luminance.
3. **Noise Removal**: Bilateral filtering eliminates scanner grain while keeping character edges crisp.
4. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) equalizes uneven lighting and shadows.
5. **Deskewing**: Hough lines & minimum bounding area detect document rotation angle and rotate the image back to 0.0°.
6. **Adaptive Thresholding**: Binarizes foreground text from background texture.
7. **Clean Image**: Morphological opening strips isolated noise artifacts.

### Step 3 — Custom OCR Engine
- **Text Detection (`app/ocr/detector.py`)**: Computes directional Sobel gradients, horizontal morphological closures, and contour analysis to locate text blocks and word clusters.
- **Text Recognition (`app/ocr/recognizer.py`)**: Modular recognizer interface with automatic fallback:
  - Custom ONNX/PyTorch models in `models/recognition_model/`
  - Tesseract OCR (if installed in system PATH)
  - EasyOCR deep learning framework
  - Fast native Computer Vision feature analyzer.

### Step 4 — OCR Post-Processing
- Assembles words into lines adhering to natural top-to-bottom, left-to-right reading order.
- Fixes character misrecognitions in numerical sequences (e.g., `O` $\rightarrow$ `0`, `l` $\rightarrow$ `1` in currencies and dates).
- Calculates normalized coordinate bounding boxes `[0.0 - 1.0]` for responsive UI rendering.
- Computes character, word, and page-level confidence metrics.

### Step 5 — Document Analysis & Automatic Field Extraction
- **Rule-based & Regex Extraction**: Parses Invoice Numbers, Dates, Due Dates, Total Amounts, Subtotals, Tax/VAT, Emails, and Phone Numbers.
- **Spatial Key-Value Pairing**: Uses geometric spatial proximity to pair key tokens (e.g. `Bill To:`, `Total:`) with nearest candidate values to the right or underneath.
- **Table Extraction Service**: Uses horizontal and vertical line structuring kernels to reconstruct grid intersections, rows, and tabular cells.
- **Pluggable LLM/NLP Structuring**: Schema-ready JSON for seamless OpenAI, Gemini, or Ollama prompt formatting.

### Step 6 — Complete Architecture, API & Interactive Frontend
- **FastAPI REST API**: Comprehensive endpoints with automatic Swagger documentation at `/docs`.
- **Interactive Web Interface**:
  - Live drag-and-drop document upload.
  - Interactive canvas bounding box overlay with hover tooltips (text, confidence %, coordinates).
  - Preprocessing stage visualizer tab slider.
  - Extracted fields cards with confidence badges.
  - Interactive table grid inspector.
  - Structured JSON / CSV export buttons.

---

## ⚡ Quickstart Guide

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the Application
Run the FastAPI development server:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
- **Interactive Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run Automated Tests
Execute the pytest suite:
```bash
pytest tests/ -v
```

---

## 📡 API Reference

### 1. Upload Document
`POST /api/upload`
- **Request**: Multipart form-data with `file`
- **Response**: Document ID, page count, and page preview URLs.

### 2. Run OCR & Extraction Pipeline
`POST /api/ocr/process`
- **Request Body**:
```json
{
  "doc_id": "YOUR_DOCUMENT_UUID",
  "engine": "auto",
  "preprocessing": {
    "enable_deskew": true,
    "enable_contrast_enhancement": true,
    "enable_denoise": true,
    "enable_thresholding": true
  },
  "extract_fields": true,
  "extract_tables": true
}
```
- **Response Body**: Full `DocumentOCRResult`, `DocumentExtractionResult`, and processing metrics.

### 3. Export Data
- `GET /api/documents/{doc_id}/export/json` — Downloads structured JSON output.
- `GET /api/documents/{doc_id}/export/csv` — Downloads tabular CSV summary.
