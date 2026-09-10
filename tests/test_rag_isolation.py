from pathlib import Path
from types import SimpleNamespace
from threading import Lock

import faiss
import numpy as np
import pytest

from app.core.config import settings
from app.services.rag import AIProviderError, RAGService


class FakeSearchIndex:
    ntotal = 3

    def search(self, query_embedding, result_count):
        assert result_count == self.ntotal
        return [[0.2, 0.3, 0.4]], [[0, 1, 2]]


class FakeEmbedClient:
    def embed(self, **kwargs):
        return SimpleNamespace(embeddings=[[0.1] * 1024])


def test_retrieval_filters_contexts_by_authenticated_user():
    service = RAGService.__new__(RAGService)
    service.index = FakeSearchIndex()
    service.documents = ["other user context", "owned context", "owned context 2"]
    service.metadatas = [
        {"user_email": "other@example.com"},
        {"user_email": "owner@example.com"},
        {"user_email": "owner@example.com"},
    ]
    service.cohere_client = FakeEmbedClient()
    service._index_lock = Lock()

    contexts, metadatas = service._retrieve_contexts(
        "question",
        "owner@example.com",
        top_k=2,
    )

    assert contexts == ["owned context", "owned context 2"]
    assert all(item["user_email"] == "owner@example.com" for item in metadatas)


def test_index_metadata_is_saved_as_json(tmp_path):
    service = RAGService.__new__(RAGService)
    service.index_path = str(Path(tmp_path) / "index.bin")
    service.meta_path = str(Path(tmp_path) / "metadata.json")
    service.index = faiss.IndexFlatL2(2)
    service.index.add(np.array([[1.0, 0.0]], dtype="float32"))
    service.documents = ["owned context"]
    service.metadatas = [{"user_email": "owner@example.com"}]

    service._save_index()

    assert Path(service.meta_path).read_text(encoding="utf-8").startswith("{")
    restored = RAGService.__new__(RAGService)
    restored.index_path = service.index_path
    restored.meta_path = service.meta_path
    restored.documents = []
    restored.metadatas = []
    restored._load_index()

    assert restored.documents == service.documents
    assert restored.metadatas == service.metadatas


def test_retrieval_rejects_chunks_beyond_distance_threshold():
    service = RAGService.__new__(RAGService)
    service.index = SimpleNamespace(
        ntotal=3,
        search=lambda query_embedding, result_count: (
            [[0.2, 2.0, 0.7]],
            [[0, 1, 2]],
        ),
    )
    service.documents = ["first", "irrelevant", "third"]
    service.metadatas = [
        {"user_email": "owner@example.com"},
        {"user_email": "owner@example.com"},
        {"user_email": "owner@example.com"},
    ]
    service.cohere_client = FakeEmbedClient()
    service._index_lock = Lock()

    contexts, _ = service._retrieve_contexts(
        "question",
        "owner@example.com",
        top_k=3,
    )

    assert contexts == ["first", "third"]


def test_query_returns_safe_no_context_answer_without_calling_groq():
    service = RAGService.__new__(RAGService)
    service._retrieve_contexts = lambda question, user_email, top_k: ([], [])

    result = service.query("question", user_email="owner@example.com")

    assert result == {
        "answer": "I could not find relevant information in your uploaded documents.",
        "sources": [],
    }


def test_answer_validation_rejects_empty_provider_output():
    with pytest.raises(AIProviderError):
        RAGService._extract_answer(SimpleNamespace(choices=[]))


def test_provider_call_retries_transient_failures(monkeypatch):
    service = RAGService.__new__(RAGService)
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("temporary timeout")
        return "ok"

    monkeypatch.setattr(settings, "AI_RETRY_BACKOFF_SECONDS", 0)
    assert service._call_provider(operation, "test operation") == "ok"
    assert attempts["count"] == 3
