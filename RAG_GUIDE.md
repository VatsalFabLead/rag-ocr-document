# 100% Permanently Free Custom Model RAG & API Key Guide

A complete guide to integrating and deploying your **own Custom Document Intelligence & RAG Model** across any project (Flutter, Python, React/Web, REST APIs, LangChain, or OpenAI SDKs) with **zero third-party API dependencies** and **your own custom model API key**.

---

## 🌟 Why This Model is 100% Permanently Free

| Dimension | Standard Third-Party APIs (OpenAI / Gemini / Groq) | **Your Custom Model (`custom-rag-doc-v1`)** |
|---|---|---|
| **Cost** | Paid or restrictive expiring tiers | **$0.00 Forever (100% Free)** |
| **API Keys** | Third-party accounts, credit cards, quotas | **Your own custom API keys (`sk-rag-live-...`)** |
| **Rate Limits** | 15 req/min, token throttles, downtime | **Unlimited local processing on your CPU/GPU** |
| **Data Privacy** | Documents sent to external cloud servers | **100% Private, strictly processed on your machine** |
| **Internet Requirement**| Constant high-speed cloud connection | **100% Offline & Local Capable** |

---

## 🔑 Your Custom Model API Key System

You do not need to register on any external websites. Your server manages and verifies its own cryptographic API keys.

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
3. Enter your project name (e.g. `Flutter Mobile App`) and click **"Generate Key"**.
4. Copy the newly generated key (`sk-rag-live-xxxxxxxx`).

#### Via REST API (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/api/keys" \
     -H "Content-Type: application/json" \
     -d '{"name": "Flutter Mobile App"}'
```

---

## ⚙️ How to Pass Your Custom API Key

Authenticate any API call in any project using any of the following 3 ways:

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
GET http://127.0.0.1:8000/api/rag/documents?api_key=sk-custom-ocr-master-v1.1
```

---

## 📱 Flutter Implementation (Mobile App)

Drop `custom_ocr/sdk/flutter/rag_client.dart` into your Flutter app (e.g. `lib/core/services/rag_client.dart`).

```dart
import 'package:your_app/core/services/rag_client.dart';

// 1. Initialize client with your server URL and Custom Model Key
final ragClient = CustomRAGClient(
  baseUrl: 'http://10.0.2.2:8000', // Use 10.0.2.2 for Android Emulator or your server LAN IP
  apiKey: 'sk-custom-ocr-master-v1.1',
);

// 2. Query a document with grounded page citations
void queryInvoice() async {
  try {
    final response = await ragClient.query(
      docId: 'invoice_2026_01',
      question: 'What is the total payable amount and the due date?',
      topK: 4,
    );

    print('AI Answer: ${response.answer}');
    print('Confidence: ${(response.confidenceScore * 100).toStringAsFixed(1)}%');

    for (var citation in response.citations) {
      print('📄 Page ${citation.pageNumber} (${(citation.score * 100).toStringAsFixed(1)}% match):');
      print('   ${citation.text}');
    }
  } catch (e) {
    print('Error querying document: $e');
  }
}

// 3. Conversational multi-turn chat with document context
void chatWithDocument() async {
  final chatResponse = await ragClient.chat(
    docId: 'invoice_2026_01',
    messages: [
      RAGChatMessage(role: 'user', content: 'What items are listed in the invoice?'),
    ],
  );

  print('Chat Response: ${chatResponse.message.content}');
}
```

---

## 🐍 Python Implementation (Any Backend or Script)

Import `custom_ocr/sdk/python/rag_client.py`:

```python
from rag_client import CustomRAGClient

# Initialize client
client = CustomRAGClient(
    base_url="http://127.0.0.1:8000",
    api_key="sk-custom-ocr-master-v1.1"
)

# 1. Ingest raw text or OCR output
client.ingest_text(
    doc_id="contract_001",
    title="Cloud Architecture Agreement",
    text="Apex Systems agrees to deliver Cloud Architecture services for a total contract price of $85,000 USD due on December 15, 2026. Payment terms are net 30 days."
)

# 2. Query document with citations
res = client.query(
    doc_id="contract_001",
    question="What is the total contract price and payment terms?"
)

print("AI Answer:", res["answer"])
print("Confidence:", res["confidence_score"])
for citation in res["citations"]:
    print(f"- Page {citation['page_number']}: {citation['text']}")
```

---

## 🤖 Direct Integration with Standard OpenAI SDKs

Because the engine exposes standard OpenAI endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`), you can plug this custom model directly into any library that supports OpenAI (Python `openai`, LangChain, LlamaIndex, Flutter `dart_openai`) with **zero changes to your code structure**:

```python
from openai import OpenAI

# Simply point base_url to your local custom model server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-custom-ocr-master-v1.1"
)

# Chat completions powered by your custom model
response = client.chat.completions.create(
    model="custom-rag-doc-v1",
    messages=[
        {"role": "user", "content": "What is the total amount on the invoice?"}
    ],
    extra_body={"doc_id": "invoice_2026_01"}
)

print(response.choices[0].message.content)
```

---

## 🌐 REST API Endpoints Overview

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/rag/ingest` | Ingest raw text or OCR output into local semantic index | Custom Key |
| `POST` | `/api/rag/query` | Grounded question answering with page citations | Custom Key |
| `POST` | `/api/rag/chat` | Multi-turn conversational chat with document | Custom Key |
| `GET` | `/api/rag/documents` | List all indexed documents & chunk counts | Custom Key |
| `DELETE`| `/api/rag/documents/{id}`| Remove document from vector store | Custom Key |
| `GET` | `/api/rag/health` | Model health status (100% Free & Local) | Public |
| `POST` | `/v1/chat/completions`| OpenAI-compatible chat completions | Custom Key |
| `POST` | `/v1/embeddings` | OpenAI-compatible 384-d vector embeddings | Custom Key |
| `GET` | `/v1/models` | List local custom models | Custom Key |
| `GET` | `/api/keys` | List active custom model API keys | Public/Admin |
| `POST` | `/api/keys` | Generate a new custom model API key | Public/Admin |
| `DELETE`| `/api/keys/{id}` | Revoke an API key | Public/Admin |

---

## 🔄 Automatic OCR to RAG Ingestion Pipeline

When you process an image or PDF via the OCR pipeline (`POST /api/ocr/process`), the system **automatically indexes**:
1. All recognized text lines and bounding box coordinates.
2. High-priority structured fields (invoice number, date, vendor, total amounts, taxes).
3. Detected table rows and line items.

You can query the document immediately after OCR using its `doc_id` with zero extra steps!
