import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.models.conversation import ConversationRequest
from app.models.document import QueryRequest
from app.models.user import UserCreate


def test_query_request_rejects_empty_questions_and_invalid_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(question="")

    with pytest.raises(ValidationError):
        QueryRequest(question="valid question", top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(question="valid question", top_k=21)


def test_conversation_request_accepts_valid_bounds():
    request = ConversationRequest(
        question="What does this document explain?",
        conversation_id="conversation-1",
        top_k=5,
    )

    assert request.question == "What does this document explain?"
    assert request.conversation_id == "conversation-1"
    assert request.top_k == 5


def test_user_creation_rejects_short_passwords():
    with pytest.raises(ValidationError):
        UserCreate(email="owner@example.com", password="short")


def test_settings_rejects_placeholder_secret_key():
    with pytest.raises(ValidationError):
        Settings(
            GROQ_API_KEY="test-groq-key",
            COHERE_API_KEY="test-cohere-key",
            SECRET_KEY="replace_with_a_long_random_secret",
            _env_file=None,
        )