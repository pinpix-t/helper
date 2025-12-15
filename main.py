"""
Design Assistant Bot - FastAPI Backend

A conversational chatbot for Printerpix's design editor pages.
Uses DeepSeek LLM to generate helpful responses about the design process.
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
MAX_CONTEXT_MESSAGES = 10

# Log config status on startup
if FRESHCHAT_API_URL and FRESHCHAT_API_KEY:
    logger.info(f"Freshchat API configured: {FRESHCHAT_API_URL}")
else:
    logger.warning("Freshchat API not configured - replies will not be sent")

if DEEPSEEK_API_KEY:
    logger.info("DeepSeek API configured")
else:
    logger.warning("DeepSeek API not configured")

# DeepSeek client (OpenAI-compatible)
deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


# --- System Prompt with Printerpix Knowledge ---

SYSTEM_PROMPT = """You are Hannah, a friendly and helpful customer support assistant for Printerpix, a photo printing company. You help customers with questions about the online design editor for photo books, calendars, and other photo products.

## Your Personality
- Friendly, warm, and professional
- Concise but helpful - don't be overly wordy
- Apologize sincerely when customers have issues
- Always offer to help further at the end

## Key Knowledge

### Calendar Start Month
Customers can choose any month as the starting month for their calendar. This option appears at the beginning of the design process, before editing starts. If they've already started a project with the wrong start month, they need to create a new project and select the correct month at the start.

### Changing Book Type or Size
It is NOT possible to change the book type or size once a project has been created. Each book type and size (leather cover, hardcover, 8x11, 11x14, etc.) has its own unique layout and formatting. To use a different option, customers must start a new project and redesign it.

### Order Preview
Customers can see exactly how their order will look by clicking the "Preview" option next to the "Add to Cart" button in the editor. The final product is printed exactly as shown in this preview, so they should check it carefully before ordering.

### Photo Upload Order (Chronological)
The Printerpix website does NOT currently support automatically uploading photos in chronological order. Each image must be added individually in the order the customer prefers. This can be time-consuming - acknowledge this and apologize for the inconvenience.

### Upload Errors / Issues
For upload problems (stuck uploads, errors, lag), recommend:
1. Open an Incognito/Private window in Google Chrome
2. Go to the Printerpix website, log in, and try uploading again
3. Make sure images are in JPG format - PNG, GIF, and SVG are NOT accepted

If the issue persists after these steps, ask for the specific error message to investigate further.

### Voucher / Groupon Code Issues
If a voucher code isn't working as expected:
1. Check that the code was applied correctly - look in the 'Order Summary' box on the right side of the cart page
2. Make sure they have the correct product in their cart - many codes only apply to specific products and sizes
3. Check if they've selected extra pages or options beyond what the voucher covers (extra pages cost extra)
4. Some vouchers only cover the product, not shipping costs

### Photo Quality / Resolution
For best print quality:
- Use high-resolution images (at least 300 DPI for print size)
- The editor will show a warning icon if an image resolution is too low
- Avoid stretching small images to fill large spaces
- Photos from social media may be compressed and not print well

### Preview Not Displaying Correctly
If the preview looks wrong or isn't loading:
1. Try refreshing the page
2. Clear browser cache or try an incognito window
3. Check if all images have fully uploaded (loading spinner should be gone)
4. The preview shows exactly what will print - if something looks off, fix it before ordering

### Layflat Photo Books
Layflat binding allows pages to lay completely flat when opened - great for panoramic photos across two pages. This option is available during book type selection at the start of a project.

### Safe Area / Bleed
Photos near the edge of pages may get slightly trimmed during production. Keep important elements (faces, text) away from the very edges. The editor shows safe zones - stay within these for best results.

### Saving Projects
Projects are automatically saved to your account as you work. You can find them in 'My Projects' when logged in. Make sure you're logged in before starting to avoid losing work.

### Canvas Products (Simple Editor)
For canvas products (Canvas Print, Framed Canvas, Framed Print, Metal Print, Photo Tile, Aluminium Print, Poster, Board Puzzle, Photo Coaster, Mouse Pad, Premium Cushion, Photo Slate, Beach Towel, Magnet Print), the editor only lets you zoom, adjust position, and replace the image. No text tools, filters, or layouts.

## Response Guidelines
- Keep responses concise (2-4 sentences when possible)
- Be conversational, not robotic
- If you don't know something specific about Printerpix, say so honestly and offer to connect them with a human agent
- Don't make up features or capabilities that weren't mentioned above
- For questions outside design/editor topics, politely explain you're specialized in design editor support

## Example Response Style
"You can preview your order by clicking 'Preview' next to the Add to Cart button - what you see there is exactly how it'll print! Let me know if you need help with anything else."

NOT: "Hello valued customer, thank you for contacting Printerpix support. I would be delighted to assist you with your inquiry regarding the preview functionality..."
"""


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
    reply: str = Field(..., description="The bot's response")
    reply_sent: bool = Field(..., description="Whether reply was sent to Freshchat")


# --- Chat with DeepSeek ---


def generate_reply(message: str, conversation_id: str) -> str:
    """
    Generate a conversational reply using DeepSeek.
    """
    if not deepseek_client:
        return "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
    
    # Build conversation history for context
    history = get_conversation_context(conversation_id)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content.strip()
        logger.info(f"DeepSeek generated reply: {reply[:50]}...")
        return reply
            
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return "I'm sorry, I'm having a bit of trouble right now. Could you try asking again?"


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
    description="Conversational AI assistant for Printerpix design support",
    version="4.0.0",
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
        "version": "4.0.0",
        "endpoint": "POST /webhook",
        "health": "/health",
        "mode": "conversational" if deepseek_client else "offline",
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

    # Add user message to conversation history
    add_to_conversation(conversation_id, "user", message)

    # Generate reply using DeepSeek
    reply = generate_reply(message, conversation_id)

    # Add assistant reply to conversation history
    add_to_conversation(conversation_id, "assistant", reply)

    logger.info(
        f"conversation_id={conversation_id} | "
        f"message={message[:30]}... | "
        f"reply={reply[:30]}..."
    )

    # Send reply to Freshchat
    reply_sent = await send_freshchat_reply(conversation_id, reply)

    return WebhookResponse(
        success=True,
        reply=reply,
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
