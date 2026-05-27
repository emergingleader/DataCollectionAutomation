"""
Kobo Scan App - Backend API
Handles: image upload → Google Vision OCR → field mapping → Kobo API submission
Supports multi-page forms (up to 5 pages per participant)
"""

import os
import json
import base64
import httpx
import re
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

app = FastAPI(title="Kobo Scan App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# HELPER: OCR one image → raw text string
# ─────────────────────────────────────────────
async def ocr_single_image(contents: bytes, page_label: str = "") -> str:
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid image file{' (' + page_label + ')' if page_label else ''}. Please upload JPG or PNG files only.")

    img = Image.open(io.BytesIO(contents))
    max_dim = 4000
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
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
        return vision_data["responses"][0]["fullTextAnnotation"]["text"]
    except (KeyError, IndexError):
        return ""


# ─────────────────────────────────────────────
# STEP 1: OCR — Accept 1 to 5 pages, merge text
# ─────────────────────────────────────────────
@app.post("/api/extract")
async def extract_from_images(files: List[UploadFile] = File(...)):
    """
    Accepts 1 or more page images for the same participant.
    OCRs each page and merges all text before field mapping.
    Supports up to 5 pages per submission.
    """
    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="Google Vision API key not configured.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 pages allowed per submission.")

    all_text_parts = []
    for i, file in enumerate(files):
        contents = await file.read()
        if not contents:
            continue
        page_label = f"Page {i+1}"
        page_text = await ocr_single_image(contents, page_label)
        if page_text.strip():
            all_text_parts.append(f"--- {page_label.upper()} ---\n{page_text}")

    if not all_text_parts:
        raise HTTPException(
            status_code=422,
            detail="No text could be detected in any of the uploaded images. Please ensure forms are clearly visible and well-lit."
        )

    merged_text = "\n\n".join(all_text_parts)
    return {"raw_text": merged_text, "pages": len(all_text_parts), "char_count": len(merged_text)}


# ─────────────────────────────────────────────
# STEP 2: MAP — OCR text → Kobo fields via Claude AI
# ─────────────────────────────────────────────
@app.post("/api/map")
async def map_fields(payload: dict):
    """
    Takes merged OCR text from all pages and maps to Kobo field values.
    Uses Claude AI for intelligent field extraction across checkbox, text, and numeric fields.
    """
    raw_text = payload.get("raw_text", "")
    if not raw_text:
        raise HTTPException(status_code=400, detail="No raw text provided.")

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

    prompt = f"""You are a data extraction assistant. A handwritten paper survey form has been scanned and OCR'd. The form may span multiple pages — all pages are included below separated by page markers.

Your job is to extract the respondent's answers and map them to the correct Kobo form fields.

IMPORTANT RULES:
1. For select_multiple fields: return a space-separated string of the matching option KEYS (not labels).
   Example: "mobile_money_account savings_group"
2. For text fields: return the written text exactly as found.
3. For integer fields: return only the number as a string.
4. For date fields: return in YYYY-MM-DD format.
5. If a checkbox or tick mark (✓ or √ or V or X) is next to an option, include that option's key.
6. If a field is blank or unanswered, return null.
7. Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
8. For names: separate First_Name and Last_Name if a full name appears.
9. Look across ALL pages for answers — do not stop at page 1.

FORM FIELDS TO EXTRACT:
{fields_str}

OCR TEXT FROM ALL SCANNED PAGES:
---
{raw_text}
---

Return a JSON object with kobo field names as keys and extracted values as values."""

    anthropic_payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}]
    }

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured. Add ANTHROPIC_API_KEY to environment variables.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": anthropic_key
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

    valid_field_names = {f["kobo_name"] for f in FORM_CONFIG["fields"]}
    cleaned = {}
    for k, v in mapped_fields.items():
        if k not in valid_field_names:
            continue
        if v is not None and str(v).strip() != "":
            cleaned[k] = v

    return {
        "mapped_fields": cleaned,
        "field_count": len(cleaned),
        "form_config": FORM_CONFIG
    }


# ─────────────────────────────────────────────
# STEP 3: SUBMIT — POST confirmed data to Kobo API
# ─────────────────────────────────────────────
@app.post("/api/submit")
async def submit_to_kobo(payload: dict):
    """
    Receives reviewed/confirmed field values and submits to Kobo API.
    Only called AFTER data collector reviews and confirms the preview.
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
# Serve frontend
# ─────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
