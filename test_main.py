"""
Tests for the Design Assistant Bot.

Run with: pytest test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient

from main import (
    app,
    conversations,
    detect_intent,
    generate_reply,
    add_to_conversation,
    get_conversation_context,
)


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_conversations():
    """Clear conversation memory before each test."""
    conversations.clear()
    yield
    conversations.clear()


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self) -> None:
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRootEndpoint:
    """Tests for the / endpoint."""

    def test_root_returns_api_info(self) -> None:
        """Root endpoint should return API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Design Assistant Bot"
        assert data["version"] == "2.0.0"
        assert data["endpoint"] == "POST /webhook"


class TestWebhookEndpoint:
    """Tests for the /webhook endpoint."""

    def test_calendar_start_month(self) -> None:
        """Should detect CALENDAR_START_MONTH intent."""
        response = client.post(
            "/webhook",
            json={
                "conversation_id": "test-123",
                "message": "Can I change my calendar start to March?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "CALENDAR_START_MONTH"
        assert data["confidence"] >= 0.7
        assert data["success"] is True

    def test_change_book_type_or_size(self) -> None:
        """Should detect CHANGE_BOOK_TYPE_OR_SIZE intent."""
        response = client.post(
            "/webhook",
            json={
                "conversation_id": "test-456",
                "message": "Can I change from 8x11 to 11x14?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "CHANGE_BOOK_TYPE_OR_SIZE"
        assert data["confidence"] >= 0.7

    def test_order_preview(self) -> None:
        """Should detect ORDER_PREVIEW intent."""
        response = client.post(
            "/webhook",
            json={
                "conversation_id": "test-789",
                "message": "How will my order look before I buy?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "ORDER_PREVIEW"
        assert data["confidence"] >= 0.7

    def test_upload_chronological(self) -> None:
        """Should detect UPLOAD_CHRONOLOGICAL intent."""
        response = client.post(
            "/webhook",
            json={
                "conversation_id": "test-abc",
                "message": "Can I upload my photos in date order?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "UPLOAD_CHRONOLOGICAL"
        assert data["confidence"] >= 0.7

    def test_upload_error_general(self) -> None:
        """Should detect UPLOAD_ERROR_GENERAL intent."""
        response = client.post(
            "/webhook",
            json={
                "conversation_id": "test-def",
                "message": "My upload is stuck at 0% in Chrome",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "UPLOAD_ERROR_GENERAL"
        assert data["confidence"] >= 0.7

    def test_unknown_intent(self) -> None:
        """Should return UNKNOWN for unrecognized questions."""
        response = client.post(
            "/webhook",
            json={
                "conversation_id": "test-unknown",
                "message": "Blah something random",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "UNKNOWN"
        assert data["confidence"] == 0.0

    def test_conversation_stored(self) -> None:
        """Should store conversation in memory."""
        conversation_id = "test-memory"
        
        # Send first message
        client.post(
            "/webhook",
            json={
                "conversation_id": conversation_id,
                "message": "Hello there",
            },
        )
        
        # Check conversation was stored
        response = client.get(f"/conversations/{conversation_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 2  # user + assistant
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello there"
        assert data["messages"][1]["role"] == "assistant"


class TestConversationMemory:
    """Tests for conversation memory functions."""

    def test_add_to_conversation(self) -> None:
        """Should add messages to conversation."""
        add_to_conversation("conv-1", "user", "Hello")
        add_to_conversation("conv-1", "assistant", "Hi there!")
        
        history = get_conversation_context("conv-1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_conversation_limit(self) -> None:
        """Should keep only last 5 messages."""
        for i in range(10):
            add_to_conversation("conv-2", "user", f"Message {i}")
        
        history = get_conversation_context("conv-2")
        assert len(history) == 5
        assert history[0]["content"] == "Message 5"
        assert history[4]["content"] == "Message 9"

    def test_separate_conversations(self) -> None:
        """Different conversation IDs should have separate histories."""
        add_to_conversation("conv-a", "user", "Hello A")
        add_to_conversation("conv-b", "user", "Hello B")
        
        history_a = get_conversation_context("conv-a")
        history_b = get_conversation_context("conv-b")
        
        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["content"] == "Hello A"
        assert history_b[0]["content"] == "Hello B"


class TestIntentDetection:
    """Unit tests for intent detection functions."""

    def test_high_confidence_multiple_keywords(self) -> None:
        """Should return 0.9 confidence when multiple keywords match."""
        reply, intent, confidence = generate_reply(
            "I want to change from 8x11 to 11x14 size",
            "test-conv"
        )
        assert intent == "CHANGE_BOOK_TYPE_OR_SIZE"
        assert confidence == 0.9

    def test_single_keyword_confidence(self) -> None:
        """Should return 0.7 confidence when exactly one keyword matches."""
        reply, intent, confidence = generate_reply("chronological please", "test-conv")
        assert intent == "UPLOAD_CHRONOLOGICAL"
        assert confidence == 0.7

    def test_case_insensitive_matching(self) -> None:
        """Should match keywords regardless of case."""
        reply, intent, confidence = generate_reply(
            "CALENDAR START month please",
            "test-conv"
        )
        assert intent == "CALENDAR_START_MONTH"

    def test_unknown_returns_fallback(self) -> None:
        """Should return UNKNOWN reply for unmatched messages."""
        reply, intent, confidence = generate_reply("gibberish xyz", "test-conv")
        assert intent == "UNKNOWN"
        assert confidence == 0.0
        assert "not sure" in reply.lower()
