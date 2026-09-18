import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services.api_key_service import api_key_service
from app.services.custom_embedding_model import custom_embedding_model
from app.services.custom_rag_service import custom_rag_service

client = TestClient(app)

def test_custom_embedding_model():
    """Verifies that the local embedding model generates normalized 384-d vectors."""
    text1 = "Invoice total is $4,500 due on December 15, 2026."
    text2 = "Total payment amount of $4,500 is due on Dec 15, 2026."
    text3 = "The quick brown fox jumps over the lazy dog."

    vec1 = custom_embedding_model.embed_text(text1)
    vec2 = custom_embedding_model.embed_text(text2)
    vec3 = custom_embedding_model.embed_text(text3)

    assert len(vec1) == 384
    assert len(vec2) == 384
    assert len(vec3) == 384

    # Semantically related texts should have higher cosine similarity than unrelated texts
    sim_related = custom_embedding_model.compute_similarity(vec1, vec2)
    sim_unrelated = custom_embedding_model.compute_similarity(vec1, vec3)

    assert sim_related > sim_unrelated
    assert -1.0 <= sim_related <= 1.0

def test_api_key_lifecycle():
    """Verifies custom API key creation, validation, and rejection."""
    # 1. Create a key
    new_key_record = api_key_service.create_key(name="Pytest Suite Key")
    assert new_key_record.api_key.startswith("sk-rag-live-")
    assert new_key_record.is_active is True

    # 2. Validate key
    validated = api_key_service.validate_key(new_key_record.api_key)
    assert validated is not None
    assert validated.request_count >= 1

    # 3. Inactive or invalid key rejection
    assert api_key_service.validate_key("sk-fake-invalid-key-999") is None

    # 4. Revoke key
    revoked = api_key_service.revoke_key(new_key_record.key_id)
    assert revoked is True
    assert api_key_service.validate_key(new_key_record.api_key) is None

def test_rag_ingestion_and_grounded_query():
    """Verifies that the custom RAG engine can ingest text and answer questions accurately."""
    doc_id = "test_doc_invoice_99"
    sample_text = (
        "Global Logistics Inc. Tax Invoice #INV-2026-9912\n"
        "Date of Issue: October 24, 2026. Due Date: November 24, 2026.\n"
        "Freight Transportation Services: $12,450.00\n"
        "State GST Tax (18%): $2,241.00\n"
        "Total Payable Amount: $14,691.00 USD."
    )

    chunks_count = custom_rag_service.ingest_text(
        doc_id=doc_id,
        text=sample_text,
        title="Global Logistics Invoice"
    )
    assert chunks_count > 0

    # Query 1: Total amount
    res = custom_rag_service.query(doc_id=doc_id, query="What is the total payable amount?")
    assert res.confidence_score > 0.5
    assert len(res.citations) > 0
    assert "14,691" in res.answer or "Total" in res.answer

    # Query 2: Invoice number
    res_inv = custom_rag_service.query(doc_id=doc_id, query="What is the invoice number?")
    assert "INV-2026-9912" in res_inv.answer

    # Cleanup
    custom_rag_service.delete_document(doc_id)

def test_api_auth_protection():
    """Verifies that RAG endpoints strictly reject unauthorized requests and accept valid keys."""
    # 1. Reject without key
    resp_no_key = client.post("/api/rag/query", json={
        "doc_id": "non_existent",
        "query": "Hello"
    })
    assert resp_no_key.status_code == 401

    # 2. Reject with fake key
    resp_fake_key = client.post(
        "/api/rag/query",
        headers={"Authorization": "Bearer sk-invalid-key-xyz"},
        json={"doc_id": "non_existent", "query": "Hello"}
    )
    assert resp_fake_key.status_code == 401

    # 3. Accept with valid master key
    master_key = api_key_service.get_primary_key()
    resp_valid = client.get(
        "/api/rag/documents",
        headers={"Authorization": f"Bearer {master_key}"}
    )
    assert resp_valid.status_code == 200

def test_openai_compatible_endpoints():
    """Verifies standard OpenAI compatible endpoints (/v1/chat/completions, /v1/embeddings, /v1/models)."""
    master_key = api_key_service.get_primary_key()
    headers = {"Authorization": f"Bearer {master_key}"}

    # 1. Models endpoint
    models_resp = client.get("/v1/models", headers=headers)
    assert models_resp.status_code == 200
    models_data = models_resp.json()
    model_ids = [m["id"] for m in models_data.get("data", [])]
    assert "custom-rag-doc-v1" in model_ids

    # 2. Embeddings endpoint
    emb_resp = client.post(
        "/v1/embeddings",
        headers=headers,
        json={"model": "custom-doc-embed-v1", "input": "Hello world document"}
    )
    assert emb_resp.status_code == 200
    emb_data = emb_resp.json()
    assert len(emb_data["data"][0]["embedding"]) == 384

    # 3. Chat Completions endpoint
    chat_resp = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "custom-rag-doc-v1",
            "messages": [{"role": "user", "content": "What is custom model?"}]
        }
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "choices" in chat_data
    assert len(chat_data["choices"]) > 0
    assert chat_data["choices"][0]["message"]["content"]
