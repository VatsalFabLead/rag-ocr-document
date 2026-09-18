# ⚡ VisionOCR — Unified API & Key Manager Documentation

> **Fast Integration Guide for Developers**: Access the entire end-to-end OCR model, text recognition, entity extraction, bounding boxes, and table reconstruction using **only your API Key and a single endpoint**. No multi-step uploads or complex pipelines required.

---

## 🌐 Production Server URLs

| Environment | Base URL |
| :--- | :--- |
| **Live Production API** | `https://rag-ocr-document.onrender.com` |
| **Local Development API** | `http://127.0.0.1:8001` |
| **Interactive Swagger UI** | `https://rag-ocr-document.onrender.com/docs` |
| **Web Dashboard** | `https://rag-ocr-document.onrender.com/` |

---

## 🔑 Part 1: API Key Manager

All requests require an API key passed in the `Authorization` header.

### 1. Instant Active Master Key
Use this key immediately for any project without setup:
```text
sk-custom-ocr-master-v1.1
```

### 2. Generate a New API Key (Per Project or Client)
Create dedicated API keys for different apps (e.g. Flutter Mobile, Web Client, Laravel Backend):

- **Method**: `POST`
- **URL**: `https://rag-ocr-document.onrender.com/api/keys`
- **Header**: `Content-Type: application/json`

#### Request Body:
```json
{
  "name": "Flutter Mobile App"
}
```

#### Response Body (`200 OK`):
```json
{
  "status": "success",
  "message": "Custom model API key generated successfully. Save this key in your project configuration.",
  "key": {
    "key_id": "key_8f1b2c",
    "name": "Flutter Mobile App",
    "api_key": "sk-custom-doc-8f1b2c45e90a...",
    "created_at": "2026-09-18T13:00:00.000000",
    "is_active": true
  }
}
```

### 3. List All Keys & Usage
- **Method**: `GET`
- **URL**: `https://rag-ocr-document.onrender.com/api/keys`

### 4. Revoke a Key
- **Method**: `DELETE`
- **URL**: `https://rag-ocr-document.onrender.com/api/keys/{key_id}`

---

## 🚀 Part 2: 1-Call Full OCR Model Access

Instead of implementing separate upload, conversion, preprocessing, and extraction APIs, call **one single endpoint**. It automatically performs:
1. File validation & PDF-to-high-res image rasterization
2. 7-stage CV preprocessing (deskew, denoise, CLAHE contrast, adaptive thresholding)
3. Deep-learning text detection & bounding box coordinates
4. CRNN neural character recognition
5. Tabular grid detection & cell reconstruction
6. Key-value & entity parsing (Invoice No, Dates, Totals, Taxes, Currency)

---

### Endpoint A: Direct File Upload (Recommended)

- **Method**: `POST`
- **URL**: `https://rag-ocr-document.onrender.com/api/ocr/predict`
- **Header**:
  ```http
  Authorization: Bearer sk-custom-ocr-master-v1.1
  Content-Type: multipart/form-data
  ```

#### Form-Data Parameters:
| Field | Type | Required | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `file` | Binary | **Yes** | — | Image (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.webp`) or PDF file (up to 25 MB) |
| `extract_fields` | Boolean | No | `true` | Automatically extract fields (Invoice #, Date, Total, etc.) |
| `extract_tables` | Boolean | No | `true` | Automatically detect and reconstruct tables |
| `engine` | String | No | `auto` | OCR Engine: `auto`, `custom_model`, `tesseract` |

#### Response Body (`200 OK`):
```json
{
  "status": "success",
  "model_name": "custom-ocr-v1",
  "doc_id": "4a18f85a-0d86-4e56-993f-c6de0bf2cb14",
  "filename": "invoice_sample.pdf",
  "page_count": 1,
  "overall_confidence": 0.942,
  "total_words": 142,
  "processing_time_total_ms": 412.5,
  "raw_text": "INVOICE\nInvoice Number: INV-2026-091\nDate: 2026-09-15\nTotal: $1,450.00\nTax: $150.00",
  "fields": {
    "invoice_number": {
      "field_name": "invoice_number",
      "value": "INV-2026-091",
      "confidence": 0.98,
      "page_number": 1
    },
    "invoice_date": {
      "field_name": "invoice_date",
      "value": "2026-09-15",
      "confidence": 0.96,
      "page_number": 1
    },
    "total_amount": {
      "field_name": "total_amount",
      "value": 1450.00,
      "confidence": 0.97,
      "page_number": 1
    },
    "tax_amount": {
      "field_name": "tax_amount",
      "value": 150.00,
      "confidence": 0.95,
      "page_number": 1
    }
  },
  "tables": [
    {
      "page_number": 1,
      "rows": [
        ["Item Description", "Qty", "Rate", "Amount"],
        ["Cloud Server Setup", "1", "$850.00", "$850.00"],
        ["SSL & Domain Config", "1", "$600.00", "$600.00"]
      ]
    }
  ],
  "pages": [
    {
      "page_number": 1,
      "full_text": "INVOICE\nInvoice Number: INV-2026-091...",
      "line_count": 18,
      "lines": [
        {
          "text": "Invoice Number: INV-2026-091",
          "confidence": 0.96,
          "bbox": {
            "x": 45,
            "y": 120,
            "width": 280,
            "height": 24,
            "x_norm": 0.045,
            "y_norm": 0.12,
            "w_norm": 0.28,
            "h_norm": 0.024
          }
        }
      ]
    }
  ]
}
```

---

### Endpoint B: Base64 JSON (For Mobile Cameras & Web)

If your app already has an image in memory or Base64 format (e.g. Flutter camera or web `<canvas>`):

- **Method**: `POST`
- **URL**: `https://rag-ocr-document.onrender.com/api/ocr/predict-base64`
- **Header**:
  ```http
  Authorization: Bearer sk-custom-ocr-master-v1.1
  Content-Type: application/json
  ```

#### Request Body:
```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "filename": "camera_capture.png",
  "extract_fields": true,
  "extract_tables": true
}
```

#### Response Body:
*Same comprehensive structured JSON output as Endpoint A.*

---

## 💻 Quick Code Snippets (1 Call = Complete Result)

### 1. JavaScript / React / Next.js
```javascript
async function runOCR(file, apiKey = "sk-custom-ocr-master-v1.1") {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("https://rag-ocr-document.onrender.com/api/ocr/predict", {
    method: "POST",
    headers: { "Authorization": `Bearer ${apiKey}` },
    body: formData
  });

  const data = await response.json();
  console.log("Full Text:", data.raw_text);
  console.log("Fields:", data.fields);
  console.log("Tables:", data.tables);
  return data;
}
```

### 2. Flutter / Dart
```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> runOCR(File file, {String apiKey = 'sk-custom-ocr-master-v1.1'}) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('https://rag-ocr-document.onrender.com/api/ocr/predict'),
  );
  request.headers['Authorization'] = 'Bearer $apiKey';
  request.files.add(await http.MultipartFile.fromPath('file', file.path));

  var streamedResponse = await request.send();
  var responseString = await streamedResponse.stream.bytesToString();
  return jsonDecode(responseString);
}
```

### 3. PHP / Laravel
```php
use Illuminate\Support\Facades\Http;

function runOCR($filePath, $apiKey = 'sk-custom-ocr-master-v1.1') {
    $response = Http::withToken($apiKey)
        ->attach('file', file_get_contents($filePath), basename($filePath))
        ->post('https://rag-ocr-document.onrender.com/api/ocr/predict');

    return $response->json();
}
```

### 4. Python
```python
import requests

def run_ocr(file_path: str, api_key: str = "sk-custom-ocr-master-v1.1"):
    with open(file_path, "rb") as f:
        res = requests.post(
            "https://rag-ocr-document.onrender.com/api/ocr/predict",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": f}
        )
    return res.json()
```

### 5. cURL
```bash
curl -X POST "https://rag-ocr-document.onrender.com/api/ocr/predict" \
     -H "Authorization: Bearer sk-custom-ocr-master-v1.1" \
     -F "file=@/path/to/invoice.pdf"
```
