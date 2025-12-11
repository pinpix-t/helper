"""
Design Assistant Bot - FastAPI Backend

A webhook-based service for Printerpix's design editor pages.
Receives messages from Freshchat, detects intent using DeepSeek LLM, and replies via Freshchat API.
"""

import logging
import os
from collections import defaultdict
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from intents import INTENTS, UNKNOWN_REPLY


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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MAX_CONTEXT_MESSAGES = 5

# Log config status on startup
if FRESHCHAT_API_URL and FRESHCHAT_API_KEY:
    logger.info(f"Freshchat API configured: {FRESHCHAT_API_URL}")
else:
    logger.warning("Freshchat API not configured - replies will not be sent")

if DEEPSEEK_API_KEY:
    logger.info("DeepSeek API configured")
else:
    logger.warning("DeepSeek API not configured - falling back to keyword matching")

# DeepSeek client (OpenAI-compatible)
deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


# --- Build intent info for the prompt ---

INTENT_DESCRIPTIONS = "\n".join([
    f"- {intent.name}: {intent.keywords[0]}, {intent.keywords[1] if len(intent.keywords) > 1 else ''}"
    for intent in INTENTS
])

INTENT_NAMES = [intent.name for intent in INTENTS]


# --- Conversation Memory ---

conversations: dict[str, list[dict[str, str]]] = defaultdict(list)


def add_to_conversation(conversation_id: str, role: str, content: str) -> None:
    """Add a message to conversation history, keeping only last N messages."""
    conversations[conversation_id].append({"role": role, "content": content})
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
    user_id: str | None = Field(default=None, description="User identifier")
    timestamp: str | None = Field(default=None, description="Message timestamp")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class WebhookResponse(BaseModel):
    """Response schema for the webhook endpoint."""

    success: bool = Field(..., description="Whether the message was processed successfully")
    intent: str = Field(..., description="Detected intent name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reply_sent: bool = Field(..., description="Whether reply was sent to Freshchat")


# --- Intent Detection with DeepSeek ---


def detect_intent_with_llm(message: str) -> tuple[str, float]:
    """
    Use DeepSeek to classify the user's message into an intent.
    
    Returns:
        Tuple of (intent_name, confidence)
    """
    if not deepseek_client:
        # Fallback to keyword matching if no API key
        return detect_intent_keywords(message)
    
    try:
        prompt = f"""You are a customer support intent classifier for Printerpix, a photo printing company.

Classify the following customer message into ONE of these intents:
- CALENDAR_START_MONTH: Questions about choosing or changing the starting month of a calendar
- CHANGE_BOOK_TYPE_OR_SIZE: Questions about changing photo book type, size, or format after starting a project
- ORDER_PREVIEW: Questions about previewing how an order will look before purchasing
- UPLOAD_CHRONOLOGICAL: Questions about uploading photos in date/chronological order
- UPLOAD_ERROR_GENERAL: Problems with uploading photos, errors, stuck uploads, loading issues
- UNKNOWN: If the message doesn't fit any of the above categories

Customer message: "{message}"

Respond with ONLY the intent name, nothing else. For example: ORDER_PREVIEW"""

        response = deepseek_client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        
        intent = response.choices[0].message.content.strip().upper()
        
        # Validate the intent
        if intent in INTENT_NAMES:
            logger.info(f"DeepSeek classified as: {intent}")
            return intent, 0.9
        elif intent == "UNKNOWN":
            logger.info("DeepSeek classified as: UNKNOWN")
            return "UNKNOWN", 0.0
        else:
            # If LLM returns something unexpected, treat as unknown
            logger.warning(f"DeepSeek returned unexpected intent: {intent}")
            return "UNKNOWN", 0.0
            
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        # Fallback to keyword matching on error
        return detect_intent_keywords(message)


def detect_intent_keywords(message: str) -> tuple[str, float]:
    """
    Fallback: keyword-based intent detection.
    """
    normalized = message.lower()
    
    best_intent = None
    best_hits = 0
    
    for intent in INTENTS:
        hits = sum(1 for keyword in intent.keywords if keyword in normalized)
        if hits > best_hits:
            best_hits = hits
            best_intent = intent
    
    if best_intent:
        confidence = 0.9 if best_hits > 1 else 0.7
        return best_intent.name, confidence
    
    return "UNKNOWN", 0.0


def get_reply_for_intent(intent_name: str) -> str:
    """Get the reply template for a given intent."""
    for intent in INTENTS:
        if intent.name == intent_name:
            reply = intent.reply
            if intent.suggested_followup:
                reply += f"\n\n{intent.suggested_followup}"
            return reply
    return UNKNOWN_REPLY


def generate_reply(message: str, conversation_id: str) -> tuple[str, str, float]:
    """
    Generate a reply for the user's message using DeepSeek for classification.
    """
    intent_name, confidence = detect_intent_with_llm(message)
    reply = get_reply_for_intent(intent_name)
    return reply, intent_name, confidence


# --- Freshchat API Client ---


async def send_freshchat_reply(conversation_id: str, message: str) -> bool:
    """
    Send a reply to Freshchat via their API.
    """
    if not FRESHCHAT_API_URL or not FRESHCHAT_API_KEY:
        logger.warning("Freshchat API not configured - reply not sent")
        return False

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
    description="LLM-powered intent detection for Printerpix design support",
    version="3.0.0",
)

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
        "version": "3.0.0",
        "endpoint": "POST /webhook",
        "health": "/health",
        "llm": "DeepSeek" if deepseek_client else "keyword fallback",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(request: WebhookRequest) -> WebhookResponse:
    """
    Process incoming message from Freshchat and send reply.
    """
    conversation_id = request.conversation_id
    message = request.message

    add_to_conversation(conversation_id, "user", message)

    reply, intent, confidence = generate_reply(message, conversation_id)

    add_to_conversation(conversation_id, "assistant", reply)

    logger.info(
        f"conversation_id={conversation_id} | "
        f"intent={intent} | "
        f"confidence={confidence:.2f} | "
        f"message_preview={message[:50]}..."
    )

    reply_sent = await send_freshchat_reply(conversation_id, reply)

    return WebhookResponse(
        success=True,
        intent=intent,
        confidence=confidence,
        reply_sent=reply_sent,
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Get conversation history for debugging/monitoring."""
    history = get_conversation_context(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message_count": len(history),
        "messages": history,
    }
