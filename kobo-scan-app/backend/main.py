"""
Kobo Scan App - Backend API
Handles: image upload → Google Vision OCR → field mapping → Kobo API submission
"""

import os
import json
import base64
import httpx
import re
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

app = FastAPI(title="Kobo Scan App")

# CORS — allow all origins so the app works from any phone browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load field map config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "field_map.json"
with open(CONFIG_PATH) as f:
    FORM_CONFIG = json.load(f)

KOBO_TOKEN = os.getenv("KOBO_TOKEN")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
KOBO_BASE_URL = "https://kf.kobotoolbox.org/api/v2"


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "form": FORM_CONFIG["form_title"]}


# ─────────────────────────────────────────────
# STEP 1: OCR — Extract text from scanned image
# ─────────────────────────────────────────────
@app.post("/api/extract")
async def extract_from_image(file: UploadFile = File(...)):
    """
    Receives an image upload, sends it to Google Vision OCR,
    returns raw extracted text for the AI mapping step.
    """
    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="Google Vision API key not configured.")

    # Read and validate image
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a JPG or PNG.")

    # Re-open after verify (verify closes the file pointer)
    img = Image.open(io.BytesIO(contents))

    # Resize if too large (Vision API limit is 20MB; we cap at 4000px to keep it fast)
    max_dim = 4000
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))

    # Convert to bytes for API
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Call Google Vision API
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]  # Best for structured docs
        }]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(vision_url, json=payload)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Google Vision API error: {response.text}"
        )

    vision_data = response.json()
    try:
        raw_text = vision_data["responses"][0]["fullTextAnnotation"]["text"]
    except (KeyError, IndexError):
        raw_text = ""

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be detected in this image. Please ensure the form is clearly visible and well-lit."
        )

    return {"raw_text": raw_text, "char_count": len(raw_text)}


# ─────────────────────────────────────────────
# STEP 2: MAP — Use Claude/AI to map OCR text → Kobo fields
# ─────────────────────────────────────────────
@app.post("/api/map")
async def map_fields(payload: dict):
    """
    Takes raw OCR text and maps it to Kobo field values.
    Uses Anthropic API (Claude) for intelligent field extraction.
    This handles checkboxes, tick marks, handwriting, and context.
    """
    raw_text = payload.get("raw_text", "")
    if not raw_text:
        raise HTTPException(status_code=400, detail="No raw text provided.")

    # Build a structured prompt with the field map
    fields_description = []
    for field in FORM_CONFIG["fields"]:
        if field["type"] == "select_multiple":
            opts = ", ".join([f"'{k}' ({v})" for k, v in field["options"].items()])
            fields_description.append(
                f"- {field['kobo_name']} [{field['type']}]: \"{field['label']}\"\n  Options: {opts}"
            )
        else:
            fields_description.append(
                f"- {field['kobo_name']} [{field['type']}]: \"{field['label']}\""
            )

    fields_str = "\n".join(fields_description)

    prompt = f"""You are a data extraction assistant. A handwritten paper survey form has been scanned and OCR'd. 
Your job is to extract the respondent's answers and map them to the correct Kobo form fields.

IMPORTANT RULES:
1. For select_multiple fields: return a space-separated string of the matching option KEYS (not labels). 
   Example: "mobile_money_account savings_group"
2. For text fields: return the written text exactly as found.
3. For integer fields: return only the number as a string.
4. For date fields: return in YYYY-MM-DD format.
5. If a checkbox or tick is marked next to an option, include that option's key.
6. If a field is blank or unanswered, return null.
7. Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
8. For names: First_Name and Last_Name should be separated from what appears to be a full name.

FORM FIELDS TO EXTRACT:
{fields_str}

OCR TEXT FROM SCANNED FORM:
---
{raw_text}
---

Return a JSON object with kobo field names as keys and extracted values as values."""

    # Call Anthropic API
    anthropic_payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            json=anthropic_payload
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"AI mapping error: {response.text}"
        )

    ai_response = response.json()
    raw_output = ai_response["content"][0]["text"].strip()

    # Strip markdown code fences if present
    if raw_output.startswith("```"):
        raw_output = re.sub(r"^```[a-zA-Z]*\n?", "", raw_output)
        raw_output = re.sub(r"\n?```$", "", raw_output)

    try:
        mapped_fields = json.loads(raw_output)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned malformed JSON. Please try again."
        )

    # Filter out null values and validate field names against our config
    valid_field_names = {f["kobo_name"] for f in FORM_CONFIG["fields"]}
    cleaned = {}
    unknown_fields = []

    for k, v in mapped_fields.items():
        if k not in valid_field_names:
            unknown_fields.append(k)
            continue
        if v is not None and str(v).strip() != "":
            cleaned[k] = v

    return {
        "mapped_fields": cleaned,
        "field_count": len(cleaned),
        "unknown_fields": unknown_fields,  # For debugging
        "form_config": FORM_CONFIG  # Sent back so frontend can build preview
    }


# ─────────────────────────────────────────────
# STEP 3: SUBMIT — POST mapped data to Kobo API
# ─────────────────────────────────────────────
@app.post("/api/submit")
async def submit_to_kobo(payload: dict):
    """
    Receives the reviewed/confirmed field values and submits to Kobo API.
    This is called AFTER the data collector reviews and confirms the preview.
    """
    if not KOBO_TOKEN:
        raise HTTPException(status_code=500, detail="Kobo API token not configured.")

    fields = payload.get("fields", {})
    if not fields:
        raise HTTPException(status_code=400, detail="No field data provided.")

    asset_uid = FORM_CONFIG["asset_uid"]
    url = f"{KOBO_BASE_URL}/assets/{asset_uid}/data/"

    headers = {
        "Authorization": f"Token {KOBO_TOKEN}",
        "Content-Type": "application/json"
    }

    # Add submission metadata
    fields["start"] = datetime.utcnow().isoformat() + "Z"
    fields["end"] = datetime.utcnow().isoformat() + "Z"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=fields)

    if response.status_code in (200, 201):
        resp_data = response.json()
        return {
            "success": True,
            "submission_id": resp_data.get("id") or resp_data.get("_id"),
            "message": "Form submitted successfully to Kobo."
        }
    else:
        raise HTTPException(
            status_code=502,
            detail=f"Kobo API rejected the submission: {response.status_code} — {response.text}"
        )


# ─────────────────────────────────────────────
# Serve frontend static files
# ─────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
