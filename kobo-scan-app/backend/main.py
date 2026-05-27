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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
KOBO_BASE_URL = "https://kf.kobotoolbox.org/api/v2"


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "form": FORM_CONFIG["form_title"],
        "kobo_token": "set" if KOBO_TOKEN else "MISSING",
        "vision_key": "set" if GOOGLE_VISION_API_KEY else "MISSING",
        "anthropic_key": "set" if ANTHROPIC_API_KEY else "MISSING"
    }


# ─────────────────────────────────────────────
# HELPER: OCR one image → raw text
# ─────────────────────────────────────────────
async def ocr_single_image(contents: bytes, page_label: str = "") -> str:
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid image file{' (' + page_label + ')' if page_label else ''}.")

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
        raise HTTPException(status_code=502, detail=f"Google Vision API error: {response.text}")

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
        raise HTTPException(status_code=422, detail="No text detected. Ensure forms are clearly visible and well-lit.")

    merged_text = "\n\n".join(all_text_parts)
    return {"raw_text": merged_text, "pages": len(all_text_parts), "char_count": len(merged_text)}


# ─────────────────────────────────────────────
# STEP 2: MAP — OCR text → Kobo fields via Claude AI
# ─────────────────────────────────────────────
@app.post("/api/map")
async def map_fields(payload: dict):
    raw_text = payload.get("raw_text", "")
    if not raw_text:
        raise HTTPException(status_code=400, detail="No raw text provided.")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured.")

    # Build detailed field descriptions with explicit option keys
    fields_description = []
    for field in FORM_CONFIG["fields"]:
        if field["type"] == "select_multiple":
            opts = "\n    ".join([f"KEY '{k}' = label '{v}'" for k, v in field["options"].items()])
            fields_description.append(
                f"FIELD: {field['kobo_name']}\n  TYPE: select_multiple\n  QUESTION: {field['label']}\n  OPTIONS:\n    {opts}"
            )
        else:
            fields_description.append(
                f"FIELD: {field['kobo_name']}\n  TYPE: {field['type']}\n  QUESTION: {field['label']}"
            )

    fields_str = "\n\n".join(fields_description)

    prompt = f"""You are an expert data extraction assistant specializing in handwritten survey forms. 

A paper survey has been scanned and the text extracted via OCR. Your job is to read the OCR text carefully and extract the respondent's answers into the correct Kobo form fields.

CRITICAL INSTRUCTIONS:
1. CHECKBOXES & TICKS: OCR renders ticks/checkmarks as √, V, ✓, or similar. If you see these symbols next to an option label, that option IS selected.
2. select_multiple fields: Return ONLY the option KEY(s) as a space-separated string. Example: if "Mobile money account" and "Savings group" are ticked, return "mobile_money_account savings_group"
3. text fields: Return the handwritten text exactly as written.
4. integer fields: Return only the number. No currency symbols, no text.
5. date fields: Return in YYYY-MM-DD format. Example: "19/04/2026" becomes "2026-04-19"
6. If a field has NO answer or is blank, return null — do not guess.
7. Look through ALL pages carefully — answers span multiple pages.
8. Return ONLY a valid JSON object. Absolutely no explanation, markdown, or extra text.

CHECKBOX DETECTION RULE — VERY IMPORTANT:
- Look for √ or V or ✓ symbols IMMEDIATELY before or after an option label
- Example OCR text: "√Self-employed" means Self-employed IS selected
- Example OCR text: "□Single √Married □Divorced" means Married IS selected
- Do NOT select options that only have □ or empty boxes next to them

FORM FIELDS:
{fields_str}

OCR TEXT FROM SCANNED FORM (all pages):
---
{raw_text}
---

Return a single JSON object mapping kobo field names to extracted values. Nothing else."""

    anthropic_payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": ANTHROPIC_API_KEY
            },
            json=anthropic_payload
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AI mapping error: {response.text}")

    ai_response = response.json()
    raw_output = ai_response["content"][0]["text"].strip()

    # Strip markdown fences if present
    if raw_output.startswith("```"):
        raw_output = re.sub(r"^```[a-zA-Z]*\n?", "", raw_output)
        raw_output = re.sub(r"\n?```$", "", raw_output)

    try:
        mapped_fields = json.loads(raw_output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed JSON. Please try again.")

    # Validate against known field names
    valid_field_names = {f["kobo_name"] for f in FORM_CONFIG["fields"]}
    cleaned = {}
    for k, v in mapped_fields.items():
        if k not in valid_field_names:
            continue
        if v is not None and str(v).strip() not in ("", "null", "None"):
            cleaned[k] = v

    return {
        "mapped_fields": cleaned,
        "field_count": len(cleaned),
        "form_config": FORM_CONFIG
    }


# ─────────────────────────────────────────────
# STEP 3: SUBMIT — POST to Kobo API
# ─────────────────────────────────────────────
@app.post("/api/submit")
async def submit_to_kobo(payload: dict):
    if not KOBO_TOKEN:
        raise HTTPException(status_code=500, detail="Kobo API token not configured.")

    fields = payload.get("fields", {})
    if not fields:
        raise HTTPException(status_code=400, detail="No field data provided.")

    # Remove null/empty values before submission
    clean_fields = {k: v for k, v in fields.items() if v is not None and str(v).strip() != ""}

    asset_uid = FORM_CONFIG["asset_uid"]

    # Kobo v2 correct submission endpoint
    url = f"{KOBO_BASE_URL}/assets/{asset_uid}/submissions/"

    headers = {
        "Authorization": f"Token {KOBO_TOKEN}",
        "Content-Type": "application/json"
    }

    now = datetime.utcnow().isoformat() + "Z"
    clean_fields["start"] = now
    clean_fields["end"] = now

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=clean_fields)

    if response.status_code in (200, 201):
        try:
            resp_data = response.json()
            sub_id = resp_data.get("id") or resp_data.get("_id") or "confirmed"
        except Exception:
            sub_id = "confirmed"
        return {"success": True, "submission_id": sub_id, "message": "Submitted successfully to Kobo."}

    raise HTTPException(
        status_code=502,
        detail=f"Kobo submission failed ({response.status_code}): {response.text[:500]}"
    )


# ─────────────────────────────────────────────
# DEBUG: Test Kobo connection and see raw response
# ─────────────────────────────────────────────
@app.get("/api/debug-kobo")
async def debug_kobo():
    """Test endpoint — shows raw Kobo API response to diagnose submission issues."""
    if not KOBO_TOKEN:
        return {"error": "KOBO_TOKEN not set"}

    asset_uid = FORM_CONFIG["asset_uid"]
    url = f"{KOBO_BASE_URL}/assets/{asset_uid}/submissions/"
    headers = {"Authorization": f"Token {KOBO_TOKEN}", "Content-Type": "application/json"}

    # Send a minimal test submission
    test_payload = {"First_Name": "TEST_DEBUG", "Last_Name": "DELETE_ME"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # First check if token works by listing asset
        asset_resp = await client.get(
            f"{KOBO_BASE_URL}/assets/{asset_uid}/",
            headers={"Authorization": f"Token {KOBO_TOKEN}"}
        )
        # Try submission
        sub_resp = await client.post(url, headers=headers, json=test_payload)
        sub_resp2 = await client.post(url, headers=headers, json={"submission": test_payload})

    return {
        "asset_check": {"status": asset_resp.status_code, "body": asset_resp.text[:300]},
        "flat_submission": {"status": sub_resp.status_code, "body": sub_resp.text[:500]},
        "wrapped_submission": {"status": sub_resp2.status_code, "body": sub_resp2.text[:500]},
    }


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
