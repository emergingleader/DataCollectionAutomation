"""
Kobo Scan App - Backend API
Handles: image upload → Google Vision OCR → field mapping → Kobo API submission

IMPROVEMENTS IN THIS VERSION:
  - Multi-form support via ?form= URL parameter
  - Submission counter endpoint (pulls live from Kobo)
  - Accuracy: stronger Claude prompt for checkboxes, text, numbers
  - Speed: image pre-compression before OCR, parallel-ready structure
  - Review: mapped fields returned with confidence flags for UI review
"""

import os
import json
import base64
import httpx
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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

# ─────────────────────────────────────────────
# CONFIG LOADING — Multi-form support
# ─────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
FORMS_REGISTRY_PATH = CONFIG_DIR / "forms.json"

with open(FORMS_REGISTRY_PATH) as f:
    FORMS_REGISTRY = json.load(f)["forms"]

def load_form_config(form_slug: str) -> dict:
    """Load config for a specific form by its URL slug."""
    if form_slug not in FORMS_REGISTRY:
        available = list(FORMS_REGISTRY.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Form '{form_slug}' not found. Available forms: {available}"
        )
    form_meta = FORMS_REGISTRY[form_slug]
    config_path = CONFIG_DIR / form_meta["config_file"]
    if not config_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Config file missing for form '{form_slug}': {form_meta['config_file']}"
        )
    with open(config_path) as f:
        return json.load(f)

# Env vars
KOBO_TOKEN = os.getenv("KOBO_TOKEN")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
KOBO_BASE_URL = "https://kf.kobotoolbox.org/api/v2"
KOBO_SUBMISSION_URL = "https://kc.kobotoolbox.org/api/v1/submissions"


# ─────────────────────────────────────────────
# HELPER: Build ODK-compatible XML
# ─────────────────────────────────────────────
def build_submission_xml(fields: dict, asset_uid: str) -> tuple:
    now = datetime.utcnow().isoformat() + "Z"
    instance_id = str(uuid.uuid4())

    field_lines = []
    for key, value in fields.items():
        if value is None or str(value).strip() in ("", "null", "None"):
            continue
        safe_value = (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
        field_lines.append(f"  <{key}>{safe_value}</{key}>")

    fields_xml = "\n".join(field_lines)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<{asset_uid} id="{asset_uid}">
  <formhub>
    <uuid>{asset_uid}</uuid>
  </formhub>
  <start>{now}</start>
  <end>{now}</end>
{fields_xml}
  <meta>
    <instanceID>uuid:{instance_id}</instanceID>
  </meta>
</{asset_uid}>"""

    return xml, instance_id


# ─────────────────────────────────────────────
# HELPER: Compress image for faster OCR
# ─────────────────────────────────────────────
def compress_image_for_ocr(contents: bytes) -> bytes:
    """
    Resize and compress image before sending to Vision API.
    Targets ~1600px max dimension — enough for clear text, much faster upload.
    """
    img = Image.open(io.BytesIO(contents))

    # Convert to RGB if needed (handles PNG with alpha, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize if larger than 1600px on longest side
    max_dim = 1600
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # Save as JPEG with good quality
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "forms_available": list(FORMS_REGISTRY.keys()),
        "kobo_token": "set" if KOBO_TOKEN else "MISSING",
        "vision_key": "set" if GOOGLE_VISION_API_KEY else "MISSING",
        "anthropic_key": "set" if ANTHROPIC_API_KEY else "MISSING"
    }


# ─────────────────────────────────────────────
# FORMS LIST — for UI dropdown
# ─────────────────────────────────────────────
@app.get("/api/forms")
def list_forms():
    """Returns all available forms for the UI to build a selector."""
    return {
        "forms": [
            {
                "slug": slug,
                "title": meta["title"],
                "description": meta["description"]
            }
            for slug, meta in FORMS_REGISTRY.items()
        ]
    }


# ─────────────────────────────────────────────
# SUBMISSION COUNTER — live from Kobo
# ─────────────────────────────────────────────
@app.get("/api/count")
async def get_submission_count(form: str = Query(..., description="Form slug")):
    """Returns total submission count for a form, pulled live from Kobo."""
    if not KOBO_TOKEN:
        raise HTTPException(status_code=500, detail="Kobo API token not configured.")

    form_config = load_form_config(form)
    asset_uid = form_config["asset_uid"]

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{KOBO_BASE_URL}/assets/{asset_uid}/submissions/?format=json&limit=1",
            headers={"Authorization": f"Token {KOBO_TOKEN}"}
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch count from Kobo: {response.status_code}"
        )

    data = response.json()
    total = data.get("count", 0)

    return {
        "form": form,
        "form_title": form_config["form_title"],
        "total_submissions": total
    }


# ─────────────────────────────────────────────
# STEP 1: OCR — Accept 1 to 5 pages, merge text
# ─────────────────────────────────────────────
async def ocr_single_image(contents: bytes, page_label: str = "") -> str:
    # Validate image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file{' (' + page_label + ')' if page_label else ''}."
        )

    # Compress before sending — speeds up upload to Vision API significantly
    compressed = compress_image_for_ocr(contents)
    img_b64 = base64.b64encode(compressed).decode("utf-8")

    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {
                "languageHints": ["en"]  # Hint: English form — improves OCR accuracy
            }
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


@app.post("/api/extract")
async def extract_from_images(
    files: List[UploadFile] = File(...),
    form: str = Query(..., description="Form slug")
):
    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="Google Vision API key not configured.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 pages allowed per submission.")

    # Validate form exists early
    load_form_config(form)

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
            detail="No text detected. Ensure forms are clearly visible and well-lit."
        )

    merged_text = "\n\n".join(all_text_parts)
    return {
        "raw_text": merged_text,
        "pages": len(all_text_parts),
        "char_count": len(merged_text),
        "form": form
    }


# ─────────────────────────────────────────────
# STEP 2: MAP — OCR text → Kobo fields via Claude
# ─────────────────────────────────────────────
@app.post("/api/map")
async def map_fields(payload: dict):
    raw_text = payload.get("raw_text", "")
    form_slug = payload.get("form", "")

    if not raw_text:
        raise HTTPException(status_code=400, detail="No raw text provided.")
    if not form_slug:
        raise HTTPException(status_code=400, detail="No form slug provided.")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured.")

    form_config = load_form_config(form_slug)

    # Build field descriptions with full option details
    fields_description = []
    for field in form_config["fields"]:
        ftype = field["type"]
        fname = field["kobo_name"]
        flabel = field["label"]

        if ftype in ("select_one", "select_multiple"):
            opts = "\n    ".join([
                f"KEY='{k}' → LABEL='{v}'" for k, v in field["options"].items()
            ])
            fields_description.append(
                f"FIELD: {fname}\n"
                f"  TYPE: {ftype}\n"
                f"  QUESTION: {flabel}\n"
                f"  VALID OPTIONS (use KEY not label):\n    {opts}"
            )
        elif ftype == "integer":
            fields_description.append(
                f"FIELD: {fname}\n"
                f"  TYPE: integer\n"
                f"  QUESTION: {flabel}\n"
                f"  RULE: Return digits only. Strip currency symbols, commas, spaces. "
                f"If unclear or blank, return null."
            )
        elif ftype == "date":
            fields_description.append(
                f"FIELD: {fname}\n"
                f"  TYPE: date\n"
                f"  QUESTION: {flabel}\n"
                f"  RULE: Return in YYYY-MM-DD format only. "
                f"Convert '19/04/2026' → '2026-04-19'. If blank, return null."
            )
        else:
            fields_description.append(
                f"FIELD: {fname}\n"
                f"  TYPE: text\n"
                f"  QUESTION: {flabel}\n"
                f"  RULE: Copy handwritten text exactly as written. "
                f"Do NOT autocorrect names, places, or spellings."
            )

    fields_str = "\n\n".join(fields_description)

    prompt = f"""You are a specialist data entry assistant for handwritten survey forms. A paper form has been scanned and OCR-processed. Extract answers accurately into the correct fields.

═══════════════════════════════════════════
CHECKBOX & TICK DETECTION — READ CAREFULLY
═══════════════════════════════════════════
OCR converts physical tick marks into these characters: √  ✓  V  ✔  /
A ticked option means it IS selected. An empty box □ or ☐ means NOT selected.

EXAMPLES:
  "□Male  √Female  □Prefer not to say"  →  selected: female
  "√Agree  □Disagree"                   →  selected: agree
  "√Employed  √Casual Worker  □Other"   →  selected (multiple): employed casual_work
  "V Married"                           →  selected: married

For select_one: return exactly ONE key (the ticked option).
For select_multiple: return space-separated keys for ALL ticked options.
If you see tick marks but cannot match them to a valid KEY, return null — do not guess.

═══════════════════════════════════════════
HANDWRITTEN TEXT — READ CAREFULLY
═══════════════════════════════════════════
- Copy names EXACTLY as written. Do not autocorrect spelling.
- Copy addresses, areas, counties exactly as written.
- If text is illegible or ambiguous, return null rather than guessing.
- Do not add punctuation or reformat the answer.

═══════════════════════════════════════════
NUMBERS — READ CAREFULLY
═══════════════════════════════════════════
- Return digits only. Strip: KES, Ksh, R, $, commas, spaces.
- "KES 15,000" → 15000
- "R 500" → 500
- "1 500" → 1500
- If blank or clearly illegible: return null.

═══════════════════════════════════════════
GENERAL RULES
═══════════════════════════════════════════
1. Scan ALL pages — answers may be spread across pages.
2. For select_one and select_multiple, ONLY return valid KEYs from the options listed.
3. Never invent or guess option keys not listed below.
4. Return ONLY a valid JSON object — no explanation, no markdown, no extra text.
5. If a field is blank, skipped, or illegible: return null.

═══════════════════════════════════════════
FORM FIELDS FOR: {form_config['form_title']}
═══════════════════════════════════════════
{fields_str}

═══════════════════════════════════════════
OCR TEXT (all pages)
═══════════════════════════════════════════
{raw_text}

Return a single JSON object with kobo field names as keys. Nothing else."""

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

    # Validate: only keep known fields, clean empties
    valid_field_names = {f["kobo_name"] for f in form_config["fields"]}
    cleaned = {}
    for k, v in mapped_fields.items():
        if k not in valid_field_names:
            continue
        if v is not None and str(v).strip() not in ("", "null", "None"):
            cleaned[k] = v

    # Build field metadata for review UI
    # Flags fields that are null (not captured) so data collector can review
    field_review = []
    for field in form_config["fields"]:
        fname = field["kobo_name"]
        value = cleaned.get(fname)
        field_review.append({
            "kobo_name": fname,
            "label": field["label"],
            "type": field["type"],
            "value": value,
            "captured": value is not None,
            "options": field.get("options", {})
        })

    return {
        "mapped_fields": cleaned,
        "field_count": len(cleaned),
        "total_fields": len(form_config["fields"]),
        "uncaptured_count": len(form_config["fields"]) - len(cleaned),
        "form_config": form_config,
        "field_review": field_review
    }


# ─────────────────────────────────────────────
# STEP 3: SUBMIT — POST to Kobo via ODK XML
# ─────────────────────────────────────────────
@app.post("/api/submit")
async def submit_to_kobo(payload: dict):
    if not KOBO_TOKEN:
        raise HTTPException(status_code=500, detail="Kobo API token not configured.")

    fields = payload.get("fields", {})
    form_slug = payload.get("form", "")

    if not fields:
        raise HTTPException(status_code=400, detail="No field data provided.")
    if not form_slug:
        raise HTTPException(status_code=400, detail="No form slug provided.")

    form_config = load_form_config(form_slug)
    asset_uid = form_config["asset_uid"]

    # Clean nulls/empties
    clean_fields = {
        k: v for k, v in fields.items()
        if v is not None and str(v).strip() != ""
    }

    xml_str, instance_id = build_submission_xml(clean_fields, asset_uid)

    files_payload = {
        "xml_submission_file": ("submission.xml", xml_str.encode("utf-8"), "text/xml")
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            KOBO_SUBMISSION_URL,
            headers={"Authorization": f"Token {KOBO_TOKEN}"},
            files=files_payload
        )

    if response.status_code in (200, 201):
        return {
            "success": True,
            "submission_id": instance_id,
            "form": form_slug,
            "form_title": form_config["form_title"],
            "message": f"Submitted successfully to {form_config['form_title']}."
        }

    raise HTTPException(
        status_code=502,
        detail=f"Kobo submission failed ({response.status_code}): {response.text[:500]}"
    )


# ─────────────────────────────────────────────
# DEBUG: Test any form's connection
# ─────────────────────────────────────────────
@app.get("/api/debug-kobo")
async def debug_kobo(form: str = Query("el-baseline", description="Form slug to test")):
    """
    Tests auth and submission for any form.
    Usage: /api/debug-kobo?form=rtf-baseline
    """
    if not KOBO_TOKEN:
        return {"error": "KOBO_TOKEN not set"}

    form_config = load_form_config(form)
    asset_uid = form_config["asset_uid"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        asset_resp = await client.get(
            f"{KOBO_BASE_URL}/assets/{asset_uid}/",
            headers={"Authorization": f"Token {KOBO_TOKEN}"}
        )

    test_fields = {"First_Name": "TEST_DEBUG", "Last_Name": "DELETE_ME"}
    xml_str, instance_id = build_submission_xml(test_fields, asset_uid)
    files_payload = {
        "xml_submission_file": ("submission.xml", xml_str.encode("utf-8"), "text/xml")
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        sub_resp = await client.post(
            KOBO_SUBMISSION_URL,
            headers={"Authorization": f"Token {KOBO_TOKEN}"},
            files=files_payload
        )

    return {
        "form_tested": form,
        "form_title": form_config["form_title"],
        "asset_uid": asset_uid,
        "asset_check": {
            "status": asset_resp.status_code,
            "note": "200 = token valid ✅  |  401/403 = bad token ❌"
        },
        "xml_submission": {
            "status": sub_resp.status_code,
            "note": "201 = submission works ✅  |  anything else = broken ❌",
            "body": sub_resp.text[:500],
            "instance_id_used": instance_id
        }
    }


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
