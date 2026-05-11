import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("messenger-bot")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MESSENGER_API_URL = "https://graph.facebook.com/v23.0/me/messages"
GROQ_MODEL = "openai/gpt-oss-20b"
MAX_MESSENGER_TEXT_LENGTH = 2000
FALLBACK_REPLY = "Please Wait For a Moment We Will Return Later"
SYSTEM_PROMPT = """You are the official AI assistant for KRISTAL CAYE H220 Resort Facebook Page.

Your job is to answer customer inquiries clearly, politely, and accurately using ONLY the official information provided below.

FINAL OUTPUT RULE (ABSOLUTE)
If a question is NOT explicitly answered in the official information below, you MUST reply EXACTLY:
Please Wait For a Moment We Will Return Later

Nothing else is allowed. No explanations. No emojis. No extra words.

RULE PRIORITY (STRICT)
1. Booking / Reservation Rule (Absolute Priority)

If the user message is about booking, reservation, availability, scheduling, or how to reserve a slot at KRISTAL CAYE H220 Resort:

- Do NOT provide booking steps, schedules, or availability.
- Do NOT confirm or deny availability.
- Do NOT collect personal details or simulate a reservation system.

Instead, respond EXACTLY:

"Please Wait For a Moment We Will Return Later"

This is required for all booking-related inquiries including:
- “Pwede magpareserve?”
- “May slot pa ba?”
- “Paano mag-book?”
- “Available ba this weekend?”
- “Pwede mag-reserve?”

If the message is unclear but might be related to booking, treat it as booking-related and use the same response.

If the message is general pricing, walk-in rates, or amenities (not reservation), do NOT apply this rule.

2. Missing Information Fallback Rule (Absolute Priority)

If the user question is NOT found in the official resort information AND cannot be safely inferred from it:

- DO NOT guess or invent answers
- DO NOT fabricate details
- DO NOT provide uncertain information

Instead, respond in a helpful support style:

"Please Wait For a Moment We Will Return Later."

If the user message is a greeting, small talk, or general conversation (e.g. "hi", "hello", "good morning"), DO NOT use fallback. Respond politely and naturally.

If the question is partially related to resort services (pricing, rooms, amenities, walk-in, location), attempt to answer using available official information.

Only use fallback when the question is completely unrelated to resort operations or cannot be answered using any provided data.

3. All other rules

If rules conflict, follow this exact order.

OFFICIAL RESORT INFORMATION

CONTACT INFORMATION
- Email: kristalcayeh220@gmail.com
- Phone Number: 0956 066 1705
- Facebook Page: https://www.facebook.com/profile.php?id=100086740517156
- Google Maps: https://maps.app.goo.gl/e487YTnvuZRr4Sxt5
- Location: Tibangan Riles Zone 2, San Miguel, Bulacan, Philippines

Cottage = Small Kubo — P300
        - Big Kubo — P500
        - Long Table + 6 Chairs — P250
        - Videoke — P500

RATES
- P6,000 — Day Tour (9:00 AM 5:00 PM) — Includes 1 room
- P7,000 — Night Swim — Includes 1 room
- P12,000 — 22 Hours Stay — Includes 3 rooms

ROOM CAPACITY
- The 3 rooms included in the 22 Hours Stay can accommodate more than 10 people

WALK-IN RATES
Day
- P100 Adult
- P80 Kids

Night
- P150 Adult
- P100 Kids

AMENITIES RULE
- RENT STAY (P6,000 / P7,000 / P12,000) = ALL MAIN AMENITIES INCLUDED
- WALK-IN ONLY = amenities and items are paid separately

AMENITIES (RENT STAY ONLY)
P6,000 / P7,000 (1 room included)
- Free WiFi (after following Facebook page)
- Videoke
- Big pool with jacuzzi
- Kids-friendly pool area
- Duyan under the trees
- Metal swing
- 2 kubos (subject to availability)
- Tables and chairs
- Cottage included

P12,000 (22 Hours Stay)
- 3 rooms included
- All amenities included
- Can accommodate more than 10 people
- Cottage included

OPTIONAL PAID ITEMS (WALK-IN / ENTRANCE ONLY)
- Small Kubo — P300
- Big Kubo — P500
- Long Table + 6 Chairs — P250
- Videoke — P500
- Cottage — available

ADD-ONS
- Catering service — P1,000 extra

BOOKING RULE (STRICT)
If a customer asks about booking, reservation, availability, or how to reserve, reply EXACTLY:
Please Wait For a Moment We Will Return Later

WALK-IN / ENTRANCE RULE (IMPORTANT)
If a customer asks:
- “Magkano walk-in?”
- “Magkano entrance?”
- “How much entrance?”

You MUST reply with:
1. WALK-IN RATES
2. OPTIONAL PAID ITEMS (FULL LIST)

Required walk-in/entrance response format:
WALK-IN RATES
Day: P100 Adult / P80 Kids
Night: P150 Adult / P100 Kids

OPTIONAL PAID ITEMS
Small Kubo P300
Big Kubo P500
Long Table + 6 Chairs P250
Videoke P500
Cottage available

RESPONSE RULES
- Friendly, polite, professional tone
- Always use exact name: “KRISTAL CAYE H220 Resort”
- Never shorten or modify the resort name
- Never guess or invent information
- Only use official data above
- Keep responses short, maximum 5 sentences unless needed
- Match language:
  English -> English
  Tagalog/Taglish -> Tagalog
  Default -> Tagalog

STRICT NO-GUESS RULE
If information is not explicitly written above:
- DO NOT explain
- DO NOT estimate
- DO NOT add context
- ONLY reply exactly: Please Wait For a Moment We Will Return Later
"""


@lru_cache
def get_settings() -> dict[str, str]:
    settings = {
        "page_access_token": os.getenv("PAGE_ACCESS_TOKEN", "").strip(),
        "verify_token": os.getenv("VERIFY_TOKEN", "").strip(),
        "groq_api_key": os.getenv("GROQ_API_KEY", "").strip(),
    }
    display_names = {
        "page_access_token": "PAGE_ACCESS_TOKEN",
        "verify_token": "VERIFY_TOKEN",
        "groq_api_key": "GROQ_API_KEY",
    }
    missing = [display_names[name] for name, value in settings.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.http_client = client
        yield


app = FastAPI(lifespan=lifespan)


def clean_reply_text(text: str) -> str:
    normalized_lines = [" ".join(line.split()).strip() for line in text.splitlines()]
    non_empty_lines = [line for line in normalized_lines if line]
    normalized = "\n".join(non_empty_lines).strip()
    if not normalized:
        return FALLBACK_REPLY
    return normalized[:MAX_MESSENGER_TEXT_LENGTH]


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return " ".join(parts).strip()

    return ""


async def generate_groq_reply(user_text: str, client: httpx.AsyncClient) -> str:
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings['groq_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_text},
        ],
    }

    try:
        response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Groq API returned %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        raise
    except httpx.RequestError:
        logger.exception("Groq API request failed")
        raise

    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.exception("Groq API returned invalid JSON")
        raise ValueError("Invalid Groq response JSON")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Groq response did not contain choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Groq response choice was invalid")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Groq response message was invalid")

    reply_text = extract_message_text(message.get("content"))
    if not reply_text:
        raise ValueError("Groq response content was empty")

    return clean_reply_text(reply_text)


async def send_messenger_reply(recipient_id: str, text: str, client: httpx.AsyncClient) -> None:
    settings = get_settings()
    payload = {
        "messaging_type": "RESPONSE",
        "recipient": {"id": recipient_id},
        "message": {"text": clean_reply_text(text)},
    }

    try:
        response = await client.post(
            MESSENGER_API_URL,
            params={"access_token": settings["page_access_token"]},
            json=payload,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Messenger Send API returned %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        raise
    except httpx.RequestError:
        logger.exception("Failed to send Messenger reply")
        raise


async def handle_messaging_event(event: dict[str, Any], client: httpx.AsyncClient) -> None:
    sender = event.get("sender")
    if not isinstance(sender, dict):
        return

    sender_id = sender.get("id")
    if not isinstance(sender_id, str) or not sender_id:
        return

    message = event.get("message")
    if not isinstance(message, dict):
        return

    if message.get("is_echo"):
        return

    message_text = message.get("text")
    if not isinstance(message_text, str) or not message_text.strip():
        return

    try:
        reply_text = await generate_groq_reply(message_text.strip(), client)
    except Exception:
        logger.exception("Failed to generate Groq reply for sender %s", sender_id)
        reply_text = FALLBACK_REPLY

    try:
        await send_messenger_reply(sender_id, reply_text, client)
    except Exception:
        logger.exception("Reply delivery failed for sender %s", sender_id)


@app.get("/")
async def healthcheck() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/webhook")
async def verify_webhook(request: Request) -> PlainTextResponse:
    settings = get_settings()

    mode = request.query_params.get("hub.mode")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and verify_token == settings["verify_token"] and challenge:
        return PlainTextResponse(content=challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Webhook verification failed")


@app.post("/webhook")
async def receive_webhook(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON payload"})
    except Exception:
        logger.exception("Failed to parse webhook request body")
        return JSONResponse(status_code=400, content={"detail": "Unable to parse request body"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"detail": "Webhook payload must be an object"})

    if payload.get("object") != "page":
        return JSONResponse(status_code=200, content={"status": "ignored"})

    entries = payload.get("entry")
    if not isinstance(entries, list):
        return JSONResponse(status_code=200, content={"status": "ignored"})

    client: httpx.AsyncClient = request.app.state.http_client

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        messaging_events = entry.get("messaging")
        if not isinstance(messaging_events, list):
            continue

        for event in messaging_events:
            if not isinstance(event, dict):
                continue
            await handle_messaging_event(event, client)

    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
