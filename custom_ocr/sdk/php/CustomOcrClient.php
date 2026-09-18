<?php

namespace App\Services\Ocr;

use Exception;

/**
 * Custom OCR & Document Intelligence Pipeline PHP Client
 * 
 * Works with any PHP application (Laravel, Symfony, WordPress, or Vanilla PHP).
 * Communicates with the FastAPI local or remote server over standard HTTP.
 */
class CustomOcrClient
{
    private string $baseUrl;
    private string $apiKey;
    private int $timeout;

    /**
     * @param string $baseUrl e.g. "http://127.0.0.1:8000" or "https://api.yourdomain.com"
     * @param string $apiKey  Your Custom Model API Key (e.g. "sk-custom-ocr-master-v1.1")
     * @param int    $timeout Request timeout in seconds (default: 60)
     */
    public function __construct(string $baseUrl = 'http://127.0.0.1:8000', string $apiKey = 'sk-custom-ocr-master-v1.1', int $timeout = 60)
    {
        $this->baseUrl = rtrim($baseUrl, '/');
        $this->apiKey = $apiKey;
        $this->timeout = $timeout;
    }

    /**
     * Check if the FastAPI server is online and healthy.
     * 
     * @return array
     * @throws Exception
     */
    public function healthCheck(): array
    {
        $url = "{$this->baseUrl}/health";
        return $this->sendRequest('GET', $url);
    }

    /**
     * Direct 1-Call OCR prediction from a local file path (PDF, PNG, JPG, etc.).
     * 
     * @param string      $filePath          Absolute or relative path to document/image
     * @param string      $engine            "auto", "custom_model", "tesseract", "builtin_cv"
     * @param bool        $extractFields     Whether to extract invoice/receipt/ID fields
     * @param bool        $extractTables     Whether to extract structured tables
     * @param string|null $documentTypeHint  "invoice", "receipt", "id_card", "generic"
     * @return array
     * @throws Exception
     */
    public function predictFile(
        string $filePath,
        string $engine = 'auto',
        bool $extractFields = true,
        bool $extractTables = true,
        ?string $documentTypeHint = null
    ): array {
        if (!file_exists($filePath)) {
            throw new Exception("File not found at path: {$filePath}");
        }

        $url = "{$this->baseUrl}/api/ocr/predict";

        $mimeType = mime_content_type($filePath) ?: 'application/octet-stream';
        $fileName = basename($filePath);

        $postFields = [
            'file' => new \CURLFile($filePath, $mimeType, $fileName),
            'engine' => $engine,
            'extract_fields' => $extractFields ? 'true' : 'false',
            'extract_tables' => $extractTables ? 'true' : 'false',
        ];

        if ($documentTypeHint !== null) {
            $postFields['document_type_hint'] = $documentTypeHint;
        }

        return $this->sendMultipartRequest($url, $postFields);
    }

    /**
     * Direct 1-Call OCR prediction from a Base64-encoded image string.
     * 
     * @param string      $base64Image       Base64 image string (with or without data URL prefix)
     * @param string      $filename          Filename reference (e.g. "scan.png")
     * @param string      $engine            "auto", "custom_model", "tesseract", "builtin_cv"
     * @param bool        $extractFields     Whether to extract invoice/receipt/ID fields
     * @param bool        $extractTables     Whether to extract structured tables
     * @param string|null $documentTypeHint  "invoice", "receipt", "id_card", "generic"
     * @return array
     * @throws Exception
     */
    public function predictBase64(
        string $base64Image,
        string $filename = 'image.png',
        string $engine = 'auto',
        bool $extractFields = true,
        bool $extractTables = true,
        ?string $documentTypeHint = null
    ): array {
        $url = "{$this->baseUrl}/api/ocr/predict-base64";

        $payload = [
            'image_base64' => $base64Image,
            'filename' => $filename,
            'engine' => $engine,
            'extract_fields' => $extractFields,
            'extract_tables' => $extractTables,
            'document_type_hint' => $documentTypeHint,
        ];

        return $this->sendRequest('POST', $url, $payload);
    }

    /**
     * Ask a question about an indexed document via the RAG pipeline.
     * 
     * @param string $docId    Document UUID from a previous OCR run
     * @param string $question Question to answer from the document content
     * @param int    $topK     Number of context chunks to retrieve
     * @return array
     * @throws Exception
     */
    public function askRagQuestion(string $docId, string $question, int $topK = 5): array
    {
        $url = "{$this->baseUrl}/api/rag/ask";

        $payload = [
            'doc_id' => $docId,
            'question' => $question,
            'top_k' => $topK,
        ];

        return $this->sendRequest('POST', $url, $payload);
    }

    /**
     * Send standard JSON HTTP request using cURL.
     */
    private function sendRequest(string $method, string $url, ?array $body = null): array
    {
        $ch = curl_init();

        $headers = [
            'Authorization: Bearer ' . $this->apiKey,
            'Accept: application/json',
        ];

        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, $this->timeout);

        if ($method === 'POST') {
            curl_setopt($ch, CURLOPT_POST, true);
            if ($body !== null) {
                $jsonBody = json_encode($body);
                curl_setopt($ch, CURLOPT_POSTFIELDS, $jsonBody);
                $headers[] = 'Content-Type: application/json';
            }
        }

        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);

        if ($response === false) {
            throw new Exception("FastAPI Connection Error: {$error}");
        }

        $decoded = json_decode($response, true);

        if ($httpCode >= 400) {
            $msg = $decoded['detail'] ?? "HTTP Error {$httpCode}: {$response}";
            throw new Exception("FastAPI Request Failed [{$httpCode}]: {$msg}");
        }

        return $decoded ?? [];
    }

    /**
     * Send multipart/form-data request using cURL for file uploads.
     */
    private function sendMultipartRequest(string $url, array $postFields): array
    {
        $ch = curl_init();

        $headers = [
            'Authorization: Bearer ' . $this->apiKey,
            'Accept: application/json',
        ];

        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $postFields);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, $this->timeout);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);

        if ($response === false) {
            throw new Exception("FastAPI File Upload Error: {$error}");
        }

        $decoded = json_decode($response, true);

        if ($httpCode >= 400) {
            $msg = $decoded['detail'] ?? "HTTP Error {$httpCode}: {$response}";
            throw new Exception("FastAPI Upload Failed [{$httpCode}]: {$msg}");
        }

        return $decoded ?? [];
    }
}
