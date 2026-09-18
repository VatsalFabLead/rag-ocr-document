# 100% Permanently Free Custom OCR Model & API Key Guide

A complete guide to integrating and deploying your **own Custom OCR Model** across any project (Flutter, Python, React/Web, REST APIs, or microservices) with **zero third-party API dependencies** and **your own custom model API key**.

---

## 🌟 Why This Custom OCR Model is 100% Free Forever

| Feature | Third-Party OCR Cloud APIs (Google Vision / AWS Textract) | **Your Custom OCR Model (`custom-ocr-v1`)** |
|---|---|---|
| **Cost** | $1.50 - $5.00 per 1,000 pages, credit card required | **$0.00 Forever (100% Permanently Free)** |
| **API Keys** | Third-party cloud vendor console | **Your own custom API keys (`sk-rag-live-...`)** |
| **Privacy** | Sensitive documents sent to cloud servers | **100% Private, strictly processed on your machine** |
| **Offline Capable** | No (requires internet connection) | **Yes (100% local CPU/GPU inference)** |
| **Rate Limits** | Strict tier limits & billing surprises | **Unlimited local requests** |
| **Pipeline** | Generic text only | **Text + Bounding Boxes + Tables + Structured Fields** |

---

## 🔑 Your Custom OCR Model API Key System

Your server verifies its own cryptographic API keys with no external accounts needed.

### 1. Default Master Key
Out of the box, the server comes configured with an active master key:
```text
sk-custom-ocr-master-v1.1
```

### 2. Generate Project-Specific Keys
You can generate unique keys for each project (e.g. Flutter mobile app, web dashboard, Python worker):

#### Via Web Dashboard:
1. Open the UI at `http://127.0.0.1:8000/`.
2. Click **"Custom Model Keys"** in the top navigation bar.
3. Enter your project name (e.g. `Flutter Mobile OCR`) and click **"Generate Key"**.
4. Copy the newly generated key (`sk-rag-live-xxxxxxxx`).

#### Via REST API (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/api/keys" \
     -H "Content-Type: application/json" \
     -d '{"name": "Flutter Mobile App"}'
```

---

## ⚙️ How to Pass Your Custom API Key

Authenticate any Custom OCR Model request using any of the following 3 ways:

### Method 1: Standard HTTP Bearer Header (Recommended)
```http
Authorization: Bearer sk-custom-ocr-master-v1.1
```

### Method 2: Custom Header
```http
X-Custom-API-Key: sk-custom-ocr-master-v1.1
```

### Method 3: Query Parameter
```http
POST http://127.0.0.1:8000/api/ocr/predict?api_key=sk-custom-ocr-master-v1.1
```

---

## 📱 Implementation in Flutter (Mobile App)

Drop `custom_ocr/sdk/flutter/custom_ocr_client.dart` into your Flutter app (e.g. `lib/core/services/custom_ocr_client.dart`).

```dart
import 'dart:io';
import 'package:your_app/core/services/custom_ocr_client.dart';

// 1. Initialize client with your server URL and Custom OCR Key
final ocrClient = CustomOCRClient(
  baseUrl: 'http://10.0.2.2:8000', // Use 10.0.2.2 for Android Emulator or your server LAN IP
  apiKey: 'sk-custom-ocr-master-v1.1',
);

// 2. Process an image or PDF file in 1 single line of code
void runCustomOCR(File documentFile) async {
  try {
    final response = await ocrClient.processFile(
      documentFile,
      extractFields: true,
      extractTables: true,
    );

    print('=== Recognized Full Text ===');
    print(response.rawText);

    print('\n=== Core Document Entities ===');
    response.fields.forEach((key, field) {
      print('${field.fieldName}: ${field.value} (${(field.confidence * 100).toStringAsFixed(1)}% conf)');
    });

    print('\n=== Extracted Tables ===');
    for (var table in response.tables) {
      print('Table on Page ${table.pageNumber}:');
      for (var row in table.rows) {
        print('  ${row.join(' | ')}');
      }
    }

    print('\n=== Line Coordinates (Bounding Boxes) ===');
    for (var page in response.pages) {
      for (var line in page.lines) {
        print('[${line.bbox.x}, ${line.bbox.y}, ${line.bbox.width}, ${line.bbox.height}] -> ${line.text}');
      }
    }
  } catch (e) {
    print('OCR Model Error: $e');
  }
}
```

---

## 🐍 Implementation in Python (Any Backend or Script)

Import `custom_ocr/sdk/python/custom_ocr_client.py`:

```python
from custom_ocr_client import CustomOCRClient

# Initialize client
client = CustomOCRClient(
    base_url="http://127.0.0.1:8000",
    api_key="sk-custom-ocr-master-v1.1"
)

# 1. Process any local image or PDF in 1 call
result = client.process_file(
    file_path="invoice.png",
    extract_fields=True,
    extract_tables=True
)

print("Recognized Text:")
print(result["raw_text"])

print("\nExtracted Entities:")
for field_key, field_data in result["fields"].items():
    print(f"- {field_data['field_name']}: {field_data['value']}")

print("\nDetected Tables:")
for table in result["tables"]:
    for row in table["rows"]:
        print(" | ".join(row))
```

---

## 🌐 REST API Endpoints Overview

| Method | Endpoint | Description | Payload |
|---|---|---|---|
| `POST` | `/api/ocr/predict` | **Direct 1-step OCR inference on uploaded image/PDF** | `multipart/form-data` with `file` |
| `POST` | `/api/ocr/predict-base64` | **Direct 1-step OCR inference from Base64 string** | `application/json` with `image_base64` |
| `GET` | `/api/ocr/model-info` | Get custom OCR model architecture & offline status | None |
| `GET` | `/api/keys` | List custom model API keys | None |
| `POST` | `/api/keys` | Generate a new custom model API key | `{"name": "Project Name"}` |
| `DELETE`| `/api/keys/{key_id}` | Revoke an API key | None |

### Direct cURL Request Examples:

#### 1. Uploading an Image File:
```bash
curl -X POST "http://127.0.0.1:8000/api/ocr/predict" \
     -H "Authorization: Bearer sk-custom-ocr-master-v1.1" \
     -F "file=@sample_invoice.png" \
     -F "extract_fields=true" \
     -F "extract_tables=true"
```

#### 2. Sending Base64 Image:
```bash
curl -X POST "http://127.0.0.1:8000/api/ocr/predict-base64" \
     -H "Authorization: Bearer sk-custom-ocr-master-v1.1" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
       "filename": "scan.png",
       "extract_fields": true
     }'
```

---

## 🧠 OCR Pipeline Stages Executed Locally

When you invoke the Custom OCR Model, it automatically executes:
1. **Adaptive Image Preprocessing**: Resize, Grayscale, CLAHE Contrast Enhancement, Skew Detection & Deskewing, Noise Filtering.
2. **Text Detection**: Identifies word/line bounding polygons across the document.
3. **Text Recognition**: Deep learning recognition model (RapidOCR ONNX PP-OCRv4 / Custom CRNN) decoding characters with confidence scores.
4. **Table Structure Extraction**: Detects grid lines, headers, data cells, and rows.
5. **Intelligent Field Extraction**: Extracts structured invoice numbers, vendor names, dates, amounts, taxes, and spatial key-value pairs.
