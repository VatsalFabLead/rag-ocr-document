// ==============================================================================
// 100% Permanently Free Custom OCR Model Client for Flutter & Dart
// Zero Third-Party API Keys | Unlimited Local Inference | Structured Extraction
// ==============================================================================

import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

/// Bounding box coordinates for lines and words.
class OCRBoundingBox {
  final int x;
  final int y;
  final int width;
  final int height;

  OCRBoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory OCRBoundingBox.fromJson(Map<String, dynamic> json) {
    return OCRBoundingBox(
      x: json['x'] as int? ?? 0,
      y: json['y'] as int? ?? 0,
      width: json['width'] as int? ?? 0,
      height: json['height'] as int? ?? 0,
    );
  }
}

/// A line of recognized text with confidence and coordinates.
class OCRLine {
  final String text;
  final double confidence;
  final OCRBoundingBox bbox;

  OCRLine({
    required this.text,
    required this.confidence,
    required this.bbox,
  });

  factory OCRLine.fromJson(Map<String, dynamic> json) {
    return OCRLine(
      text: json['text'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      bbox: OCRBoundingBox.fromJson(json['bbox'] as Map<String, dynamic>? ?? {}),
    );
  }
}

/// A page recognized by the Custom OCR Model.
class OCRPage {
  final int pageNumber;
  final String fullText;
  final int lineCount;
  final List<OCRLine> lines;

  OCRPage({
    required this.pageNumber,
    required this.fullText,
    required this.lineCount,
    required this.lines,
  });

  factory OCRPage.fromJson(Map<String, dynamic> json) {
    var rawLines = json['lines'] as List<dynamic>? ?? [];
    return OCRPage(
      pageNumber: json['page_number'] as int? ?? 1,
      fullText: json['full_text'] as String? ?? '',
      lineCount: json['line_count'] as int? ?? 0,
      lines: rawLines.map((e) => OCRLine.fromJson(e as Map<String, dynamic>)).toList(),
    );
  }
}

/// A structured field extracted by the model (e.g. Total, Invoice #, Date, Vendor).
class OCRField {
  final String fieldName;
  final String value;
  final double confidence;
  final int pageNumber;

  OCRField({
    required this.fieldName,
    required this.value,
    required this.confidence,
    required this.pageNumber,
  });

  factory OCRField.fromJson(Map<String, dynamic> json) {
    return OCRField(
      fieldName: json['field_name'] as String? ?? '',
      value: json['value'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      pageNumber: json['page_number'] as int? ?? 1,
    );
  }
}

/// A spatial key-value pair detected on the document.
class OCRKeyValue {
  final String key;
  final String value;
  final double confidence;

  OCRKeyValue({required this.key, required this.value, required this.confidence});

  factory OCRKeyValue.fromJson(Map<String, dynamic> json) {
    return OCRKeyValue(
      key: json['key'] as String? ?? '',
      value: json['value'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

/// A detected table structure.
class OCRTable {
  final int pageNumber;
  final List<List<String>> rows;

  OCRTable({required this.pageNumber, required this.rows});

  factory OCRTable.fromJson(Map<String, dynamic> json) {
    var rawRows = json['rows'] as List<dynamic>? ?? [];
    List<List<String>> parsedRows = [];
    for (var r in rawRows) {
      if (r is List) {
        parsedRows.add(r.map((c) => c.toString()).toList());
      }
    }
    return OCRTable(
      pageNumber: json['page_number'] as int? ?? 1,
      rows: parsedRows,
    );
  }
}

/// Master output returned by the Custom OCR Model.
class CustomOCRResponse {
  final String status;
  final String modelName;
  final String docId;
  final String filename;
  final int pageCount;
  final String rawText;
  final double overallConfidence;
  final int totalWords;
  final double processingTimeTotalMs;
  final Map<String, OCRField> fields;
  final List<OCRKeyValue> keyValues;
  final List<OCRTable> tables;
  final List<OCRPage> pages;

  CustomOCRResponse({
    required this.status,
    required this.modelName,
    required this.docId,
    required this.filename,
    required this.pageCount,
    required this.rawText,
    required this.overallConfidence,
    required this.totalWords,
    required this.processingTimeTotalMs,
    required this.fields,
    required this.keyValues,
    required this.tables,
    required this.pages,
  });

  factory CustomOCRResponse.fromJson(Map<String, dynamic> json) {
    var rawFields = json['fields'] as Map<String, dynamic>? ?? {};
    Map<String, OCRField> parsedFields = {};
    rawFields.forEach((k, v) {
      if (v is Map<String, dynamic>) {
        parsedFields[k] = OCRField.fromJson(v);
      }
    });

    var rawKvs = json['key_values'] as List<dynamic>? ?? [];
    var rawTables = json['tables'] as List<dynamic>? ?? [];
    var rawPages = json['pages'] as List<dynamic>? ?? [];

    return CustomOCRResponse(
      status: json['status'] as String? ?? 'success',
      modelName: json['model_name'] as String? ?? 'custom-ocr-v1',
      docId: json['doc_id'] as String? ?? '',
      filename: json['filename'] as String? ?? '',
      pageCount: json['page_count'] as int? ?? 1,
      rawText: json['raw_text'] as String? ?? '',
      overallConfidence: (json['overall_confidence'] as num?)?.toDouble() ?? 0.0,
      totalWords: json['total_words'] as int? ?? 0,
      processingTimeTotalMs: (json['processing_time_total_ms'] as num?)?.toDouble() ?? 0.0,
      fields: parsedFields,
      keyValues: rawKvs.map((e) => OCRKeyValue.fromJson(e as Map<String, dynamic>)).toList(),
      tables: rawTables.map((e) => OCRTable.fromJson(e as Map<String, dynamic>)).toList(),
      pages: rawPages.map((e) => OCRPage.fromJson(e as Map<String, dynamic>)).toList(),
    );
  }
}

/// Primary Custom OCR Model Client for Flutter Apps.
class CustomOCRClient {
  final String baseUrl;
  final String apiKey;
  final http.Client _client;

  /// [baseUrl] e.g. "http://10.0.2.2:8000" (Android Emulator) or "http://192.168.1.X:8000"
  /// [apiKey] Your custom model API key (e.g. "sk-custom-ocr-master-v1.1")
  CustomOCRClient({
    required this.baseUrl,
    required this.apiKey,
    http.Client? client,
  })  : _client = client ?? http.Client();

  /// Process an Image or PDF file with 1 line of code.
  Future<CustomOCRResponse> processFile(
    File file, {
    String engine = 'auto',
    bool extractFields = true,
    bool extractTables = true,
    String? docTypeHint,
  }) async {
    final uri = Uri.parse('$baseUrl/api/ocr/predict');
    final request = http.MultipartRequest('POST', uri);

    request.headers['Authorization'] = 'Bearer $apiKey';
    request.fields['engine'] = engine;
    request.fields['extract_fields'] = extractFields.toString();
    request.fields['extract_tables'] = extractTables.toString();
    if (docTypeHint != null) {
      request.fields['document_type_hint'] = docTypeHint;
    }

    request.files.add(await http.MultipartFile.fromPath('file', file.path));

    final streamedResponse = await _client.send(request);
    final responseBody = await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode == 200) {
      return CustomOCRResponse.fromJson(jsonDecode(responseBody));
    } else {
      throw Exception('Custom OCR Model Failed [${streamedResponse.statusCode}]: $responseBody');
    }
  }

  /// Process raw image bytes (e.g. from camera capture or picker).
  Future<CustomOCRResponse> processBytes(
    List<int> bytes, {
    String filename = 'capture.png',
    String engine = 'auto',
    bool extractFields = true,
    bool extractTables = true,
    String? docTypeHint,
  }) async {
    final uri = Uri.parse('$baseUrl/api/ocr/predict');
    final request = http.MultipartRequest('POST', uri);

    request.headers['Authorization'] = 'Bearer $apiKey';
    request.fields['engine'] = engine;
    request.fields['extract_fields'] = extractFields.toString();
    request.fields['extract_tables'] = extractTables.toString();
    if (docTypeHint != null) {
      request.fields['document_type_hint'] = docTypeHint;
    }

    request.files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));

    final streamedResponse = await _client.send(request);
    final responseBody = await streamedResponse.stream.bytesToString();

    if (streamedResponse.statusCode == 200) {
      return CustomOCRResponse.fromJson(jsonDecode(responseBody));
    } else {
      throw Exception('Custom OCR Model Failed [${streamedResponse.statusCode}]: $responseBody');
    }
  }

  /// Process a Base64-encoded image string.
  Future<CustomOCRResponse> processBase64(
    String base64Image, {
    String filename = 'image.png',
    String engine = 'auto',
    bool extractFields = true,
    bool extractTables = true,
    String? docTypeHint,
  }) async {
    final uri = Uri.parse('$baseUrl/api/ocr/predict-base64');
    final res = await _client.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $apiKey',
      },
      body: jsonEncode({
        'image_base64': base64Image,
        'filename': filename,
        'engine': engine,
        'extract_fields': extractFields,
        'extract_tables': extractTables,
        'document_type_hint': docTypeHint,
      }),
    );

    if (res.statusCode == 200) {
      return CustomOCRResponse.fromJson(jsonDecode(res.body));
    } else {
      throw Exception('Custom OCR Model Failed [${res.statusCode}]: ${res.body}');
    }
  }

  /// Check custom OCR model status.
  Future<Map<String, dynamic>> getModelInfo() async {
    final uri = Uri.parse('$baseUrl/api/ocr/model-info');
    final res = await _client.get(uri);
    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  void dispose() {
    _client.close();
  }
}
