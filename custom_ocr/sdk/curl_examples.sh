#!/bin/bash
# ==============================================================================
# 100% Permanently Free Custom Model API Key & RAG cURL Examples
# ==============================================================================

SERVER_URL="http://127.0.0.1:8000"
API_KEY="sk-custom-ocr-master-v1.1"

echo "=== 1. Check Custom Model Health ==="
curl -s -X GET "${SERVER_URL}/api/rag/health" | jq .

echo -e "\n=== 2. List or Generate New Custom Model Key ==="
curl -s -X POST "${SERVER_URL}/api/keys" \
     -H "Content-Type: application/json" \
     -d '{"name": "Flutter Mobile App"}' | jq .

echo -e "\n=== 3. Ingest Sample Document Text ==="
curl -s -X POST "${SERVER_URL}/api/rag/ingest" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${API_KEY}" \
     -d '{
       "doc_id": "contract_101",
       "title": "Master Services Agreement",
       "text": "Apex Systems agrees to deliver Cloud Architecture services for a total contract price of $85,000 USD due on December 15, 2026. Payment terms are net 30 days."
     }' | jq .

echo -e "\n=== 4. Grounded Document Query ==="
curl -s -X POST "${SERVER_URL}/api/rag/query" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${API_KEY}" \
     -d '{
       "doc_id": "contract_101",
       "query": "What is the total contract price and when is it due?",
       "top_k": 3
     }' | jq .

echo -e "\n=== 5. OpenAI-Compatible Chat Completion ==="
curl -s -X POST "${SERVER_URL}/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${API_KEY}" \
     -d '{
       "model": "custom-rag-doc-v1",
       "doc_id": "contract_101",
       "messages": [
         {"role": "user", "content": "What are the payment terms?"}
       ]
     }' | jq .
