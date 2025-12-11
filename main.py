"""
Design Assistant Bot - FastAPI Backend

A webhook-based service for Printerpix's design editor pages.
Receives messages from Freshchat, detects intent, and replies via Freshchat API.
"""

import logging
import os
from collections import defaultdict
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from intents import INTENTS, UNKNOWN_REPLY, IntentDefinition


# --- Load Environment Variables ---
load_dotenv()


# --- Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Configuration ---

FRESHCHAT_API_URL = os.getenv("FRESHCHAT_API_URL", "")
FRESHCHAT_API_KEY = os.getenv("FRESHCHAT_API_KEY", "")
MAX_CONTEXT_MESSAGES = 5

# Log config status on startup
if FRESHCHAT_API_URL and FRESHCHAT_API_KEY:
    logger.info(f"Freshchat API configured: {FRESHCHAT_API_URL}")
else:
    logger.warning("Freshchat API not configured - replies will not be sent")


# --- Conversation Memory ---

# In-memory store: {conversation_id: [list of messages]}
# Each message is {"role": "user"|"assistant", "content": "..."}
conversations: dict[str, list[dict[str, str]]] = defaultdict(list)


def add_to_conversation(conversation_id: str, role: str, content: str) -> None:
    """Add a message to conversation history, keeping only last N messages."""
    conversations[conversation_id].append({"role": role, "content": content})
    # Keep only the last MAX_CONTEXT_MESSAGES
    if len(conversations[conversation_id]) > MAX_CONTEXT_MESSAGES:
        conversations[conversation_id] = conversations[conversation_id][-MAX_CONTEXT_MESSAGES:]


def get_conversation_context(conversation_id: str) -> list[dict[str, str]]:
    """Get conversation history for a given conversation."""
    return conversations[conversation_id].copy()


# --- Pydantic Models ---


class WebhookRequest(BaseModel):
    """Request schema for the Freshchat webhook."""

    conversation_id: str = Field(..., description="Unique conversation identifier")
    message: str = Field(..., description="User's message")
    # Optional fields that Freshchat might send
    user_id: str | None = Field(default=None, description="User identifier")
    timestamp: str | None = Field(default=None, description="Message timestamp")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class WebhookResponse(BaseModel):
    """Response schema for the webhook endpoint."""

    success: bool = Field(..., description="Whether the message was processed successfully")
    intent: str = Field(..., description="Detected intent name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reply_sent: bool = Field(..., description="Whether reply was sent to Freshchat")


# --- Intent Detection Logic ---


def detect_intent(message: str) -> tuple[IntentDefinition | None, int]:
    """
    Detect the best matching intent for a given message.

    Args:
        message: The user's message (will be normalized to lowercase)

    Returns:
        Tuple of (matched IntentDefinition or None, number of keyword hits)
    """
    normalized = message.lower()

    best_intent: IntentDefinition | None = None
    best_hits = 0

    for intent in INTENTS:
        hits = sum(1 for keyword in intent.keywords if keyword in normalized)
        if hits > best_hits:
            best_hits = hits
            best_intent = intent

    return best_intent, best_hits


def calculate_confidence(hits: int) -> float:
    """
    Calculate confidence score based on number of keyword hits.

    Args:
        hits: Number of keywords matched

    Returns:
        Confidence score (0.9 if >1 hit, 0.7 if 1 hit, 0.0 if none)
    """
    if hits > 1:
        return 0.9
    elif hits == 1:
        return 0.7
    else:
        return 0.0


def generate_reply(message: str, conversation_id: str) -> tuple[str, str, float]:
    """
    Generate a reply for the user's message.

    Args:
        message: The user's message
        conversation_id: The conversation ID for context

    Returns:
        Tuple of (reply text, intent name, confidence)
    """
    intent, hits = detect_intent(message)
    confidence = calculate_confidence(hits)

    if intent is None:
        return UNKNOWN_REPLY, "UNKNOWN", 0.0

    reply = intent.reply
    if intent.suggested_followup:
        reply += f"\n\n{intent.suggested_followup}"

    return reply, intent.name, confidence


# --- Freshchat API Client ---


async def send_freshchat_reply(conversation_id: str, message: str) -> bool:
    """
    Send a reply to Freshchat via their API.

    Freshchat API endpoint: POST /v2/conversations/{conversation_id}/messages
    
    Args:
        conversation_id: The Freshchat conversation ID
        message: The message to send

    Returns:
        True if successful, False otherwise
    """
    if not FRESHCHAT_API_URL or not FRESHCHAT_API_KEY:
        logger.warning("Freshchat API not configured - reply not sent")
        return False

    # Build the full URL for sending messages
    # FRESHCHAT_API_URL should be like: https://xxx.freshchat.com/v2
    url = f"{FRESHCHAT_API_URL}/conversations/{conversation_id}/messages"

    # Freshchat message format - replies as Hannah
    payload = {
        "message_parts": [
            {
                "text": {
                    "content": message
                }
            }
        ],
        "actor_type": "agent",
        "actor_id": "c7813965-392f-45a4-a417-bebd90369a02"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {FRESHCHAT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            logger.info(f"Reply sent to Freshchat | conversation_id={conversation_id}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Freshchat API error: {e.response.status_code} - {e.response.text}")
        return False
    except httpx.HTTPError as e:
        logger.error(f"Failed to send Freshchat reply: {e}")
        return False


# --- FastAPI App ---


app = FastAPI(
    title="Design Assistant Bot",
    description="Webhook-based intent detection for Printerpix design support",
    version="2.0.0",
)

# CORS - allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "service": "Design Assistant Bot",
        "version": "2.0.0",
        "endpoint": "POST /webhook",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(request: WebhookRequest) -> WebhookResponse:
    """
    Process incoming message from Freshchat and send reply.

    This endpoint:
    1. Receives a message from Freshchat
    2. Adds it to conversation history
    3. Detects intent and generates reply
    4. Sends reply back via Freshchat API
    5. Logs the interaction
    """
    conversation_id = request.conversation_id
    message = request.message

    # Add user message to conversation history
    add_to_conversation(conversation_id, "user", message)

    # Generate reply
    reply, intent, confidence = generate_reply(message, conversation_id)

    # Add assistant reply to conversation history
    add_to_conversation(conversation_id, "assistant", reply)

    # Log the interaction
    logger.info(
        f"conversation_id={conversation_id} | "
        f"intent={intent} | "
        f"confidence={confidence:.2f} | "
        f"message_preview={message[:50]}..."
    )

    # Send reply to Freshchat
    reply_sent = await send_freshchat_reply(conversation_id, reply)

    return WebhookResponse(
        success=True,
        intent=intent,
        confidence=confidence,
        reply_sent=reply_sent,
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    """
    Get conversation history for debugging/monitoring.

    Args:
        conversation_id: The conversation ID to look up

    Returns:
        Conversation history and metadata
    """
    history = get_conversation_context(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message_count": len(history),
        "messages": history,
    }
