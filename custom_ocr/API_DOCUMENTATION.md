# 📑 VisionOCR & Document Intelligence API Documentation

Production REST API Reference for Custom OCR, 7-Stage Computer Vision Preprocessing, Structured Key-Value Extraction, Table Reconstruction, and Grounded Document Q&A / RAG.

---

## 🌐 Server Base URLs

| Environment | Base URL | Status |
| :--- | :--- | :--- |
| **Production (Render Cloud)** | `https://rag-ocr-document.onrender.com` | **Live** |
| **Local Development** | `http://127.0.0.1:8001` | **Local** |
| **Interactive Swagger UI** | `https://rag-ocr-document.onrender.com/docs` | Interactive |
| **ReDoc Documentation** | `https://rag-ocr-document.onrender.com/redoc` | Interactive |

---

## 🔐 Authentication

All OCR processing, RAG, and key management endpoints are protected via Bearer Token authorization.

### Default Master Key
```text
sk-custom-ocr-master-v1.1
```

### Authorization Header
Include this header in every authenticated request:
```http
Authorization: Bearer sk-custom-ocr-master-v1.1
```
*(Alternatively: `X-API-Key: sk-custom-ocr-master-v1.1`)*

---

## 📋 Endpoints Overview

| Category | Method | Endpoint | Description |
| :--- | :---: | :--- | :--- |
| **Health** | `GET` | `/health` | Check service health and directory status |
| **Upload** | `POST` | `/api/upload` | Upload and validate PDF or image file |
| **OCR Pipeline** | `POST` | `/api/ocr/process` | Run OCR, field extraction, & table detection |
| **Preprocessing** | `GET` | `/api/ocr/preprocess-preview/{doc_id}` | Retrieve 7-stage visual preview snapshots |
| **Documents** | `GET` | `/api/documents/{doc_id}` | Get document metadata and processing status |
| **Export** | `GET` | `/api/documents/{doc_id}/export/json` | Download structured JSON results |
| **Export** | `GET` | `/api/documents/{doc_id}/export/csv` | Download tabular CSV summary |
| **RAG Ingest** | `POST` | `/api/rag/ingest` | Index document text into local vector store |
| **RAG Query** | `POST` | `/api/rag/query` | Grounded question answering with citations |
| **RAG Chat** | `POST` | `/api/rag/chat` | Conversational document Q&A history |
| **OpenAI Compat** | `POST` | `/v1/chat/completions` | Standard OpenAI-compatible chat API |
| **API Keys** | `POST` | `/api/keys` | Generate a new custom API key |
| **API Keys** | `GET` | `/api/keys` | List all active API keys |

---

## 1. System Health Check

Verify that the API engine and file storage paths are operational.

- **Method**: `GET`
- **URL**: `/health`
- **Auth**: None

#### Example Response (`200 OK`)
```json
{
  "status": "healthy",
  "service": "Custom OCR & Document Intelligence Pipeline",
  "version": "1.0.0",
  "upload_dir": "/app/uploads",
  "output_dir": "/app/outputs"
}
```

---

## 2. Upload Document

Upload an image or multi-page PDF. The server validates magic byte headers, enforces file limits, and converts PDF pages to high-resolution 300 DPI bitmaps.

- **Method**: `POST`
- **URL**: `/api/upload`
- **Content-Type**: `multipart/form-data`
- **Auth**: None
- **Limits**:
  - Allowed Formats: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`
  - Max File Size: `25 MB`
  - Max PDF Pages: `50 pages`

#### Form Parameters
| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `file` | Binary File | **Yes** | PDF or image file to upload |

#### Example Response (`200 OK`)
```json
{
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "filename": "tax_invoice_2026.pdf",
  "file_size_bytes": 142850,
  "mime_type": "application/pdf",
  "page_count": 1,
  "pages": [
    {
      "page_number": 1,
      "image_url": "/outputs/9b12a84e-289c-4f71-a477-d6e24177d64a/page_1.png",
      "width": 2479,
      "height": 3508
    }
  ],
  "uploaded_at": "2026-09-18T13:00:00.000000",
  "message": "File uploaded and validated successfully."
}
```

---

## 3. Run OCR & Extraction Pipeline

Executes the end-to-end intelligence pipeline:
1. **7-Stage CV Preprocessing**: Resize, Grayscale, Bilateral Denoise, CLAHE Contrast, Deskew, Adaptive Thresholding, and Morphological Opening.
2. **Custom OCR**: Directional gradient text detection + CRNN/Tesseract recognition.
3. **Post-Processing**: Reading order assembly, coordinate normalization `[0.0 - 1.0]`, and numeric format correction.
4. **Analysis**: Regex & spatial key-value extraction + morphological table extraction.

- **Method**: `POST`
- **URL**: `/api/ocr/process`
- **Content-Type**: `application/json`
- **Auth**: `Authorization: Bearer <API_KEY>`

#### Request Body
```json
{
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "engine": "auto",
  "preprocessing": {
    "enable_resize": true,
    "enable_grayscale": true,
    "enable_denoise": true,
    "enable_contrast_enhancement": true,
    "enable_deskew": true,
    "enable_thresholding": true,
    "threshold_method": "adaptive",
    "target_dpi": 300
  },
  "extract_fields": true,
  "extract_tables": true,
  "document_type_hint": "invoice"
}
```

#### Request Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| `doc_id` | String | **Required** | The ID returned from `/api/upload` |
| `engine` | String | `"auto"` | `"auto"`, `"custom_cv"`, `"tesseract"`, or `"easyocr"` |
| `extract_fields` | Boolean | `true` | Extract invoice/form key-value pairs |
| `extract_tables` | Boolean | `true` | Detect tabular borders and cell data |
| `document_type_hint` | String | `null` | Optional hint: `"invoice"`, `"receipt"`, or `"generic"` |

#### Example Response (`200 OK`)
```json
{
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "status": "completed",
  "processing_time_total_ms": 328.4,
  "ocr_result": {
    "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
    "total_pages": 1,
    "full_text": "TAX INVOICE\nInvoice Number: INV-2026-9042\nDate: 2026-09-18\nDue Date: 2026-10-18\nTotal: $3,250.00",
    "overall_confidence": 0.952,
    "pages": [
      {
        "page_number": 1,
        "width": 2479,
        "height": 3508,
        "confidence": 0.952,
        "lines": [
          {
            "text": "Invoice Number: INV-2026-9042",
            "confidence": 0.97,
            "bbox": { "x_min": 0.12, "y_min": 0.22, "x_max": 0.48, "y_max": 0.25 }
          }
        ]
      }
    ]
  },
  "extraction_result": {
    "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
    "document_type": "invoice",
    "overall_confidence": 0.96,
    "fields": {
      "invoice_number": {
        "field_name": "invoice_number",
        "value": "INV-2026-9042",
        "raw_text": "INV-2026-9042",
        "field_type": "invoice_number",
        "confidence": 0.98,
        "page_number": 1
      },
      "invoice_date": {
        "field_name": "invoice_date",
        "value": "2026-09-18",
        "raw_text": "2026-09-18",
        "field_type": "invoice_date",
        "confidence": 0.96,
        "page_number": 1
      },
      "total_amount": {
        "field_name": "total_amount",
        "value": 3250.00,
        "raw_text": "$3,250.00",
        "field_type": "total_amount",
        "confidence": 0.97,
        "page_number": 1
      }
    },
    "tables": [
      {
        "page_number": 1,
        "rows": 3,
        "cols": 4,
        "headers": ["Description", "Quantity", "Rate", "Amount"],
        "data": [
          ["Enterprise OCR License", "1", "$2,500.00", "$2,500.00"],
          ["Cloud Deployment Setup", "1", "$750.00", "$750.00"]
        ]
      }
    ]
  }
}
```

---

## 4. Document Q&A / RAG Engine

Ask intelligent questions about indexed documents with verified text grounding and exact passage citations.

### 4A. Ingest Document Text (`POST /api/rag/ingest`)
Index extracted OCR text or raw document content into the vector store.

- **URL**: `/api/rag/ingest`
- **Auth**: `Authorization: Bearer <API_KEY>`

```json
{
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "title": "Tax Invoice 2026",
  "text": "Invoice Number: INV-2026-9042. Total Amount: $3,250.00. Payment due within 30 days via wire transfer to FabLead Account #88392.",
  "page_number": 1
}
```

#### Response (`200 OK`)
```json
{
  "status": "success",
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "chunks_created": 1,
  "message": "Indexed 1 chunk(s) into local vector store."
}
```

---

### 4B. Query Document (`POST /api/rag/query`)
Submit natural language questions against the document.

- **URL**: `/api/rag/query`
- **Auth**: `Authorization: Bearer <API_KEY>`

```json
{
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "query": "What is the payment terms and bank account?",
  "top_k": 3
}
```

#### Response (`200 OK`)
```json
{
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "query": "What is the payment terms and bank account?",
  "answer": "Payment is due within 30 days via wire transfer to FabLead Account #88392.",
  "confidence": 0.95,
  "citations": [
    {
      "chunk_id": "chunk_0",
      "text": "Payment due within 30 days via wire transfer to FabLead Account #88392.",
      "page_number": 1,
      "similarity_score": 0.91
    }
  ]
}
```

---

## 5. OpenAI-Compatible Chat Completions

Seamlessly connect existing OpenAI SDKs (Python `openai`, LangChain, LlamaIndex) directly to this pipeline.

- **Method**: `POST`
- **URL**: `/v1/chat/completions` (or `/api/v1/chat/completions`)
- **Headers**:
  ```http
  Content-Type: application/json
  Authorization: Bearer sk-custom-ocr-master-v1.1
  ```

#### Request Body
```json
{
  "model": "custom-rag-doc-v1",
  "doc_id": "9b12a84e-289c-4f71-a477-d6e24177d64a",
  "messages": [
    { "role": "system", "content": "You are a professional document analysis assistant." },
    { "role": "user", "content": "What is the invoice number and grand total?" }
  ],
  "temperature": 0.2
}
```

#### Response Body
```json
{
  "id": "chatcmpl-9b12a84e",
  "object": "chat.completion",
  "created": 1726660800,
  "model": "custom-rag-doc-v1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The invoice number is INV-2026-9042 and the grand total is $3,250.00."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 22,
    "total_tokens": 150
  }
}
```

---

## 6. Export Extracted Data

Download processed document data in standardized formats.

* **Export JSON**: `GET /api/documents/{doc_id}/export/json`
  * Returns complete structured JSON payload.
* **Export CSV**: `GET /api/documents/{doc_id}/export/csv`
  * Downloads comma-separated tabular format suitable for Excel or accounting ingestion.

---

## 7. API Key Management

Generate and manage scoped API keys for different client applications.

### Generate Key (`POST /api/keys`)
- **URL**: `/api/keys`
- **Body**: `{ "name": "iOS Mobile Application" }`
- **Response**:
```json
{
  "key_id": "key_e41b2c",
  "name": "iOS Mobile Application",
  "api_key": "sk-custom-doc-e41b2cf80918...",
  "created_at": "2026-09-18T13:00:00.000000",
  "is_active": true
}
```

### List Keys (`GET /api/keys`)
- **URL**: `/api/keys`
- **Response**: Returns array of all active registered keys and metadata.

---

## 8. Multi-Language Integration Code Snippets

### cURL
```bash
# Step 1: Upload
UPLOAD_RES=$(curl -s -X POST "https://rag-ocr-document.onrender.com/api/upload" \
  -F "file=@invoice.pdf")
DOC_ID=$(echo $UPLOAD_RES | jq -r '.doc_id')

# Step 2: Process OCR
curl -s -X POST "https://rag-ocr-document.onrender.com/api/ocr/process" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-custom-ocr-master-v1.1" \
  -d '{
    "doc_id": "'"$DOC_ID"'",
    "extract_fields": true,
    "extract_tables": true
  }' | jq .
```

### JavaScript / TypeScript / React / Node.js
```javascript
const API_URL = "https://rag-ocr-document.onrender.com";
const API_KEY = "sk-custom-ocr-master-v1.1";

async function processInvoice(file) {
  // 1. Upload File
  const formData = new FormData();
  formData.append("file", file);

  const uploadRes = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });
  const { doc_id } = await uploadRes.json();

  // 2. Run OCR & Extraction
  const ocrRes = await fetch(`${API_URL}/api/ocr/process`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${API_KEY}`
    },
    body: JSON.stringify({
      doc_id: doc_id,
      extract_fields: true,
      extract_tables: true
    })
  });

  return await ocrRes.json();
}
```

### Flutter / Dart
```dart
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

class VisionOcrApi {
  static const String baseUrl = 'https://rag-ocr-document.onrender.com';
  static const String apiKey = 'sk-custom-ocr-master-v1.1';

  static Future<Map<String, dynamic>> extractDocument(File file) async {
    // 1. Upload
    var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/upload'));
    request.files.add(await http.MultipartFile.fromPath('file', file.path));
    var streamedResponse = await request.send();
    var uploadBody = await streamedResponse.stream.bytesToString();
    var docId = jsonDecode(uploadBody)['doc_id'];

    // 2. Process
    var response = await http.post(
      Uri.parse('$baseUrl/api/ocr/process'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $apiKey',
      },
      body: jsonEncode({
        'doc_id': docId,
        'extract_fields': true,
        'extract_tables': true,
      }),
    );

    return jsonDecode(response.body);
  }
}
```

### Python
```python
import requests

BASE_URL = "https://rag-ocr-document.onrender.com"
API_KEY = "sk-custom-ocr-master-v1.1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def extract_invoice(file_path: str):
    # 1. Upload
    with open(file_path, "rb") as f:
        upload_res = requests.post(f"{BASE_URL}/api/upload", files={"file": f}).json()

    doc_id = upload_res["doc_id"]

    # 2. Extract
    ocr_res = requests.post(
        f"{BASE_URL}/api/ocr/process",
        headers=HEADERS,
        json={"doc_id": doc_id, "extract_fields": True, "extract_tables": True}
    ).json()

    return ocr_res["extraction_result"]["fields"]
```

### PHP / Laravel
```php
namespace App\Services;

use Illuminate\Support\Facades\Http;

class OcrService
{
    protected string $baseUrl = 'https://rag-ocr-document.onrender.com';
    protected string $apiKey = 'sk-custom-ocr-master-v1.1';

    public function processDocument(string $filePath): array
    {
        // 1. Upload Document
        $uploadResponse = Http::attach(
            'file', file_get_contents($filePath), basename($filePath)
        )->post("{$this->baseUrl}/api/upload");

        $docId = $uploadResponse->json('doc_id');

        // 2. Run OCR Pipeline
        $ocrResponse = Http::withToken($this->apiKey)
            ->post("{$this->baseUrl}/api/ocr/process", [
                'doc_id' => $docId,
                'extract_fields' => true,
                'extract_tables' => true,
            ]);

        return $ocrResponse->json();
    }
}
```

---

## 9. Error Codes Reference

| HTTP Status | Error Code | Description |
| :---: | :--- | :--- |
| `400` | `BAD_REQUEST` | Unsupported file type, corrupt file, or empty document text |
| `401` | `UNAUTHORIZED` | Missing, invalid, or revoked API Bearer token |
| `404` | `NOT_FOUND` | `doc_id` does not exist or has expired |
| `413` | `PAYLOAD_TOO_LARGE` | File exceeds maximum `25 MB` limit |
| `422` | `UNPROCESSABLE_ENTITY` | Validation error in JSON schema |
| `500` | `INTERNAL_SERVER_ERROR` | Server engine error during OCR/CV computation |
