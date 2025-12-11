"""
Intent definitions for the Design Assistant Bot.

This module contains all intent definitions with their keywords, reply templates,
and metadata. Designed to be easily extensible - add new intents to the INTENTS list
or load them from external sources (CSV, JSON, etc.) in a future iteration.
"""

from dataclasses import dataclass


@dataclass
class IntentDefinition:
    """Represents a single intent with its matching keywords and response template."""

    name: str
    keywords: list[str]
    reply: str
    suggested_followup: str = ""


# All intent definitions - add new intents here
INTENTS: list[IntentDefinition] = [
    IntentDefinition(
        name="CALENDAR_START_MONTH",
        keywords=[
            "start month",
            "calendar start",
            "calendar from",
            "start from january",
            "start from march",
        ],
        reply=(
            "You can choose any month as the starting month for your calendar. "
            "This option appears at the beginning of the design process, before you start editing. "
            "If you've already started a project and want a different start month, you'll need to "
            "create a new project and select the correct month at the start."
        ),
    ),
    IntentDefinition(
        name="CHANGE_BOOK_TYPE_OR_SIZE",
        keywords=[
            "change book type",
            "change size",
            "switch size",
            "leather to hardcover",
            "8x11",
            "11x14",
        ],
        reply=(
            "It isn't possible to change the book type or size once a project has been created. "
            "Each book type and size has its own layout and formatting, so to use a different option "
            "you'll need to start a new project with the new type or size and redesign it there."
        ),
    ),
    IntentDefinition(
        name="ORDER_PREVIEW",
        keywords=[
            "how will my order look",
            "preview my order",
            "see final product",
            "see how it looks",
        ],
        reply=(
            'You can see exactly how your order will look by clicking the "Preview" option next to '
            'the "Add to Cart" button in the editor. Your final product is printed exactly as shown '
            "in this preview, so please check it carefully before placing the order."
        ),
    ),
    IntentDefinition(
        name="UPLOAD_CHRONOLOGICAL",
        keywords=[
            "chronological",
            "date order",
            "upload in order",
            "upload photos in order",
            "sort by date",
        ],
        reply=(
            "At the moment, the website doesn't support automatically uploading photos in "
            "chronological order. Images need to be added individually in the order you prefer. "
            "We understand this can be time-consuming and are sorry for the inconvenience."
        ),
    ),
    IntentDefinition(
        name="UPLOAD_ERROR_GENERAL",
        keywords=[
            "error uploading",
            "upload stuck",
            "upload is stuck",
            "keeps loading",
            "won't upload",
            "problem uploading",
            "lag while uploading",
            "stuck at",
        ],
        reply=(
            "Let's try a quick fix:\n"
            "1) Open an Incognito or Private window in Google Chrome.\n"
            "2) Go to the Printerpix website, log in, and try uploading your pictures again.\n"
            "3) Make sure your images are in JPG format. PNG, GIF, and SVG are not supported.\n\n"
            "If you still have trouble, please let me know what error message you see."
        ),
        suggested_followup="What exact error message do you see on screen?",
    ),
]


# UNKNOWN intent - used when no other intent matches
UNKNOWN_REPLY = (
    "I'm not sure I understand that question. Could you rephrase it, "
    "or let me know more details about what you're trying to do in the editor?"
)
