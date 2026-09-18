// ==============================================================================
// 100% Permanently Free Custom Model RAG Client for Flutter & Dart
// Zero Third-Party API Keys | Unlimited Local Inference | Grounded Citations
// ==============================================================================

import 'dart:convert';
import 'package:http/http.dart' as http;

/// Citation returned by the custom model pointing directly to document pages.
class RAGCitation {
  final int pageNumber;
  final String text;
  final double score;
  final Map<String, dynamic>? bbox;

  RAGCitation({
    required this.pageNumber,
    required this.text,
    required this.score,
    this.bbox,
  });

  factory RAGCitation.fromJson(Map<String, dynamic> json) {
    return RAGCitation(
      pageNumber: json['page_number'] as int? ?? 1,
      text: json['text'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      bbox: json['bbox'] as Map<String, dynamic>?,
    );
  }
}

/// Response returned from asking questions to a document.
class RAGQueryResponse {
  final String answer;
  final String docId;
  final double confidenceScore;
  final List<RAGCitation> citations;
  final int retrievalCount;
  final String modelName;
  final double processingTimeMs;

  RAGQueryResponse({
    required this.answer,
    required this.docId,
    required this.confidenceScore,
    required this.citations,
    required this.retrievalCount,
    required this.modelName,
    required this.processingTimeMs,
  });

  factory RAGQueryResponse.fromJson(Map<String, dynamic> json) {
    var rawCitations = json['citations'] as List<dynamic>? ?? [];
    return RAGQueryResponse(
      answer: json['answer'] as String? ?? '',
      docId: json['doc_id'] as String? ?? '',
      confidenceScore: (json['confidence_score'] as num?)?.toDouble() ?? 0.0,
      citations: rawCitations.map((e) => RAGCitation.fromJson(e as Map<String, dynamic>)).toList(),
      retrievalCount: json['retrieval_count'] as int? ?? 0,
      modelName: json['model_name'] as String? ?? 'custom-rag-doc-v1',
      processingTimeMs: (json['processing_time_ms'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

/// Chat message model for conversational multi-turn document dialogue.
class RAGChatMessage {
  final String role; // 'user' or 'assistant'
  final String content;

  RAGChatMessage({required this.role, required this.content});

  Map<String, dynamic> toJson() => {'role': role, 'content': content};

  factory RAGChatMessage.fromJson(Map<String, dynamic> json) => RAGChatMessage(
        role: json['role'] as String? ?? 'assistant',
        content: json['content'] as String? ?? '',
      );
}

/// Multi-turn chat response.
class RAGChatResponse {
  final RAGChatMessage message;
  final String docId;
  final List<RAGCitation> citations;
  final double confidenceScore;

  RAGChatResponse({
    required this.message,
    required this.docId,
    required this.citations,
    required this.confidenceScore,
  });

  factory RAGChatResponse.fromJson(Map<String, dynamic> json) {
    var rawCitations = json['citations'] as List<dynamic>? ?? [];
    return RAGChatResponse(
      message: RAGChatMessage.fromJson(json['message'] as Map<String, dynamic>),
      docId: json['doc_id'] as String? ?? '',
      citations: rawCitations.map((e) => RAGCitation.fromJson(e as Map<String, dynamic>)).toList(),
      confidenceScore: (json['confidence_score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

/// Primary Custom Model Client for Flutter Apps.
class CustomRAGClient {
  final String baseUrl;
  final String apiKey;
  final http.Client _client;

  /// [baseUrl] e.g. "http://10.0.2.2:8000" (Android Emulator) or "http://192.168.1.X:8000" (Physical Device)
  /// [apiKey] Your custom model API key (e.g. "sk-custom-rag-live-...")
  CustomRAGClient({
    required this.baseUrl,
    required this.apiKey,
    http.Client? client,
  })  : _client = client ?? http.Client();

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $apiKey',
      };

  /// Query an indexed document and receive a grounded answer with page citations.
  Future<RAGQueryResponse> query({
    required String docId,
    required String question,
    int topK = 4,
  }) async {
    final uri = Uri.parse('$baseUrl/api/rag/query');
    final res = await _client.post(
      uri,
      headers: _headers,
      body: jsonEncode({
        'doc_id': docId,
        'query': question,
        'top_k': topK,
      }),
    );

    if (res.statusCode == 200) {
      return RAGQueryResponse.fromJson(jsonDecode(res.body));
    } else {
      throw Exception('RAG Query Failed [${res.statusCode}]: ${res.body}');
    }
  }

  /// Have a multi-turn conversation with a document.
  Future<RAGChatResponse> chat({
    required String docId,
    required List<RAGChatMessage> messages,
  }) async {
    final uri = Uri.parse('$baseUrl/api/rag/chat');
    final res = await _client.post(
      uri,
      headers: _headers,
      body: jsonEncode({
        'doc_id': docId,
        'messages': messages.map((m) => m.toJson()).toList(),
      }),
    );

    if (res.statusCode == 200) {
      return RAGChatResponse.fromJson(jsonDecode(res.body));
    } else {
      throw Exception('RAG Chat Failed [${res.statusCode}]: ${res.body}');
    }
  }

  /// Ingest raw text or OCR output into the local vector store.
  Future<bool> ingestText({
    required String docId,
    required String text,
    String? title,
    int pageNumber = 1,
  }) async {
    final uri = Uri.parse('$baseUrl/api/rag/ingest');
    final res = await _client.post(
      uri,
      headers: _headers,
      body: jsonEncode({
        'doc_id': docId,
        'text': text,
        'title': title,
        'page_number': pageNumber,
      }),
    );

    return res.statusCode == 200;
  }

  /// Check health and engine status.
  Future<Map<String, dynamic>> checkHealth() async {
    final uri = Uri.parse('$baseUrl/api/rag/health');
    final res = await _client.get(uri);
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  void dispose() {
    _client.close();
  }
}
