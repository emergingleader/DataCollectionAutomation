"""
Kobo Scan App - Backend API
ACCURACY IMPROVEMENTS (no new APIs, no extra cost):
  - Image preprocessing: sharpen + contrast boost before Vision OCR
  - Claude returns confidence per field (high/low)
  - Only low-confidence fields flagged for review
  - High-confidence fields auto-accepted, shown collapsed
"""

import os
import json
import base64
import httpx
import re
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageEnhance
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

CONFIG_DIR = Path(__file__).parent.parent / "config"
FORMS_REGISTRY_PATH = CONFIG_DIR / "forms.json"
with open(FORMS_REGISTRY_PATH) as f:
    FORMS_REGISTRY = json.load(f)["forms"]

def load_form_config(form_slug: str) -> dict:
    if form_slug not in FORMS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Form '{form_slug}' not found. Available: {list(FORMS_REGISTRY.keys())}")
    form_meta = FORMS_REGISTRY[form_slug]
    config_path = CONFIG_DIR / form_meta["config_file"]
    if not config_path.exists():
        raise HTTPException(status_code=500, detail=f"Config file missing: {form_meta['config_file']}")
    with open(config_path) as f:
        return json.load(f)

KOBO_TOKEN = os.getenv("KOBO_TOKEN")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
KOBO_BASE_URL = "https://kf.kobotoolbox.org/api/v2"
KOBO_SUBMISSION_URL = "https://kc.kobotoolbox.org/api/v1/submissions"


def build_submission_xml(fields: dict, asset_uid: str) -> tuple:
    now = datetime.utcnow().isoformat() + "Z"
    instance_id = str(uuid.uuid4())
    field_lines = []
    for key, value in fields.items():
        if value is None or str(value).strip() in ("", "null", "None"):
            continue
        safe_value = str(value).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&apos;")
        field_lines.append(f"  <{key}>{safe_value}</{key}>")
    fields_xml = "\n".join(field_lines)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<{asset_uid} id="{asset_uid}">
  <formhub><uuid>{asset_uid}</uuid></formhub>
  <start>{now}</start><end>{now}</end>
{fields_xml}
  <meta><instanceID>uuid:{instance_id}</instanceID></meta>
</{asset_uid}>"""
    return xml, instance_id


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING — improves OCR accuracy
# Sharpen + contrast boost before sending to Vision
# Especially helps with dim photos and faint handwriting
# ─────────────────────────────────────────────
def preprocess_image_for_ocr(contents: bytes) -> bytes:
    img = Image.open(io.BytesIO(contents))

    # Convert to RGB
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize to 1200px max — optimal for text OCR
    max_dim = 1200
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # Convert to grayscale for OCR — removes colour noise that confuses Vision
    img = img.convert("L")

    # Sharpen edges — makes handwriting crisper
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)  # Apply twice for stronger effect

    # Boost contrast — makes dark ink stand out against paper
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)  # 1.8x contrast (1.0 = no change)

    # Boost sharpness one more time after contrast
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)

    # Convert back to RGB for JPEG
    img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()


# ─────────────────────────────────────────────
# CLAUDE PROMPT — now asks for confidence per field
# ─────────────────────────────────────────────
def build_claude_prompt(raw_text: str, form_config: dict) -> str:
    fields_description = []
    for field in form_config["fields"]:
        ftype, fname, flabel = field["type"], field["kobo_name"], field["label"]
        if ftype in ("select_one", "select_multiple"):
            opts = "\n    ".join([f"KEY='{k}' → LABEL='{v}'" for k, v in field["options"].items()])
            fields_description.append(
                f"FIELD: {fname}\n  TYPE: {ftype}\n  QUESTION: {flabel}\n  OPTIONS:\n    {opts}")
        elif ftype == "integer":
            fields_description.append(
                f"FIELD: {fname}\n  TYPE: integer\n  QUESTION: {flabel}\n"
                f"  RULE: Digits only. Strip KES/Ksh/R/$/commas. Blank=null.")
        elif ftype == "date":
            fields_description.append(
                f"FIELD: {fname}\n  TYPE: date\n  QUESTION: {flabel}\n"
                f"  RULE: YYYY-MM-DD. '19/04/2026'→'2026-04-19'. Blank=null.")
        else:
            fields_description.append(
                f"FIELD: {fname}\n  TYPE: text\n  QUESTION: {flabel}\n"
                f"  RULE: Copy exactly as written. No autocorrect.")

    return f"""You are a data entry specialist for handwritten survey forms.

CHECKBOX DETECTION:
OCR renders ticks as: √ ✓ V ✔ / — these mean SELECTED.
Empty box □ ☐ = NOT selected.
Examples: "□Male √Female" → female | "√Employed √Casual" → employed casual_work
select_one: ONE key. select_multiple: ALL ticked keys space-separated. No match → null.

TEXT: Copy exactly. No autocorrect. Illegible → null.
NUMBERS: Digits only. "KES 15,000"→15000 "R 500"→500. Blank→null.
DATES: YYYY-MM-DD. "19/04/2026"→"2026-04-19". Blank→null.

RULES:
1. Scan ALL pages.
2. Only valid KEYs from options. Never invent keys.
3. Return ONLY valid JSON. No markdown, no explanation.
4. Blank/illegible → null.

CONFIDENCE SCORING — IMPORTANT:
For each field, also return a confidence score:
  "high" = you can clearly see the answer in the OCR text
  "low"  = the answer is ambiguous, partially visible, or you had to infer it

Return a JSON object where each field has TWO keys:
  "value": the extracted value (or null)
  "confidence": "high" or "low"

Example output format:
{{
  "First_Name": {{"value": "Jane", "confidence": "high"}},
  "Age": {{"value": "18__29", "confidence": "high"}},
  "County": {{"value": null, "confidence": "low"}},
  "Approximately_how_mu_you_earn_in_a_month": {{"value": "15000", "confidence": "low"}}
}}

FORM: {form_config['form_title']}

FIELDS:
{chr(10).join(fields_description)}

OCR TEXT:
{raw_text}

Return JSON only (with value + confidence per field):"""


@app.get("/health")
def health():
    return {"status": "ok", "forms_available": list(FORMS_REGISTRY.keys()),
            "kobo_token": "set" if KOBO_TOKEN else "MISSING",
            "vision_key": "set" if GOOGLE_VISION_API_KEY else "MISSING",
            "anthropic_key": "set" if ANTHROPIC_API_KEY else "MISSING"}


@app.get("/api/forms")
def list_forms():
    return {"forms": [{"slug": slug, "title": meta["title"], "description": meta["description"]}
                      for slug, meta in FORMS_REGISTRY.items()]}


@app.get("/api/count")
async def get_submission_count(form: str = Query(...)):
    if not KOBO_TOKEN:
        raise HTTPException(status_code=500, detail="Kobo token not configured.")
    form_config = load_form_config(form)
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{KOBO_BASE_URL}/assets/{form_config['asset_uid']}/submissions/?format=json&limit=1",
            headers={"Authorization": f"Token {KOBO_TOKEN}"}
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Could not fetch count: {response.status_code}")
    return {"form": form, "form_title": form_config["form_title"],
            "total_submissions": response.json().get("count", 0)}


# ─────────────────────────────────────────────
# STEP 1: OCR — PARALLEL + preprocessed images
# ─────────────────────────────────────────────
async def ocr_one_page(img_b64: str, page_label: str) -> tuple:
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    payload = {"requests": [{"image": {"content": img_b64},
                              "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                              "imageContext": {"languageHints": ["en"]}}]}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(vision_url, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Vision API error ({page_label}): {response.text[:200]}")
    try:
        text = response.json()["responses"][0]["fullTextAnnotation"]["text"]
    except (KeyError, IndexError):
        text = ""
    return page_label, text


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
        raise HTTPException(status_code=400, detail="Maximum 5 pages allowed.")

    load_form_config(form)

    tasks = []
    for i, file in enumerate(files):
        contents = await file.read()
        if not contents:
            continue
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image on page {i+1}.")
        # ✅ Preprocess: sharpen + contrast before OCR
        processed = preprocess_image_for_ocr(contents)
        img_b64 = base64.b64encode(processed).decode("utf-8")
        tasks.append(ocr_one_page(img_b64, f"Page {i+1}"))

    if not tasks:
        raise HTTPException(status_code=400, detail="No valid images found.")

    # ✅ Parallel OCR
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_text_parts = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        page_label, page_text = result
        if page_text.strip():
            all_text_parts.append(f"--- {page_label.upper()} ---\n{page_text}")

    if not all_text_parts:
        raise HTTPException(status_code=422,
            detail="No text detected. Ensure forms are clearly visible and well-lit.")

    merged_text = "\n\n".join(all_text_parts)
    return {"raw_text": merged_text, "pages": len(all_text_parts),
            "char_count": len(merged_text), "form": form}


# ─────────────────────────────────────────────
# STEP 2: MAP — Streaming + confidence scoring
# ─────────────────────────────────────────────
def recover_partial_json(text: str) -> dict:
    """
    Attempts to salvage fields from a truncated JSON response.
    Extracts all complete "field": {"value": ..., "confidence": ...} pairs
    that appeared before the response was cut off.
    Returns a dict of whatever was successfully parsed.
    """
    recovered = {}
    # Match complete field entries: "field_name": {"value": ..., "confidence": "..."}
    pattern = r'"([^"]+)"\s*:\s*\{[^}]*"value"\s*:\s*([^,}]+)[^}]*"confidence"\s*:\s*"(high|low)"[^}]*\}'
    for match in re.finditer(pattern, text):
        field_name = match.group(1)
        raw_value = match.group(2).strip().strip('"')
        confidence = match.group(3)
        if raw_value.lower() in ("null", "none", ""):
            recovered[field_name] = {"value": None, "confidence": confidence}
        else:
            recovered[field_name] = {"value": raw_value, "confidence": confidence}
    return recovered


async def stream_claude(raw_text: str, form_config: dict) -> AsyncGenerator[str, None]:
    prompt = build_claude_prompt(raw_text, form_config)
    payload = {
        "model": "claude-sonnet-4-5",
        # ✅ FIX 1: Increased from 4000 to 8000
        # EL Baseline has 43 fields × ~50 tokens each (with confidence) = ~2150 tokens
        # Plus prompt overhead. 8000 gives plenty of headroom for all forms.
        "max_tokens": 8000,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}]
    }
    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json",
                         "anthropic-version": "2023-06-01",
                         "x-api-key": ANTHROPIC_API_KEY},
                json=payload
            ) as response:
                if response.status_code != 200:
                    err = await response.aread()
                    yield f"data: {json.dumps({'error': f'AI error: {err[:200].decode()}'})}\n\n"
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        chunk = event.get("delta", {}).get("text", "")
                        if chunk:
                            full_response += chunk
                            if len(full_response) % 50 == 0:
                                yield f"data: {json.dumps({'type': 'progress', 'chars': len(full_response)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    # Clean response
    cleaned = full_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    # ✅ FIX 2: Try full JSON parse first; fall back to partial recovery if truncated
    raw_mapped = None
    try:
        raw_mapped = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to recover whatever fields were parsed before truncation
        raw_mapped = recover_partial_json(cleaned)
        if not raw_mapped:
            # Nothing salvageable — genuinely malformed
            yield f"data: {json.dumps({'error': 'AI response was cut short. Please try again — this usually resolves on retry.'})}\n\n"
            return
        # Partial recovery succeeded — continue with what we have
        # Fields not in raw_mapped will appear as missing (red) in the review screen

    # ─────────────────────────────────────────────
    # Parse confidence scores from Claude's response
    # Claude returns: {"field": {"value": x, "confidence": "high"/"low"}}
    # We split this into mapped_fields + confidence_map
    # ─────────────────────────────────────────────
    valid_names = {f["kobo_name"] for f in form_config["fields"]}
    mapped_fields = {}
    confidence_map = {}

    for k, v in raw_mapped.items():
        if k not in valid_names:
            continue
        # Handle both formats: {value, confidence} dict or plain value
        if isinstance(v, dict) and "value" in v:
            field_value = v.get("value")
            confidence = v.get("confidence", "low")
        else:
            field_value = v
            confidence = "high"  # Plain value = treat as high confidence

        if field_value is not None and str(field_value).strip() not in ("", "null", "None"):
            mapped_fields[k] = field_value
            confidence_map[k] = confidence
        else:
            confidence_map[k] = "low"  # Missing = low confidence

    # Build field review with confidence flags
    field_review = []
    for field in form_config["fields"]:
        fname = field["kobo_name"]
        value = mapped_fields.get(fname)
        confidence = confidence_map.get(fname, "low")
        needs_review = (value is None) or (confidence == "low")
        field_review.append({
            "kobo_name": fname,
            "label": field["label"],
            "type": field["type"],
            "value": value,
            "captured": value is not None,
            "confidence": confidence,
            "needs_review": needs_review,
            "options": field.get("options", {})
        })

    # Stats for the UI summary
    high_conf = sum(1 for f in field_review if not f["needs_review"])
    needs_review_count = sum(1 for f in field_review if f["needs_review"])

    yield f"data: {json.dumps({'type': 'done', 'mapped_fields': mapped_fields, 'field_count': len(mapped_fields), 'total_fields': len(form_config['fields']), 'high_confidence_count': high_conf, 'needs_review_count': needs_review_count, 'form_config': form_config, 'field_review': field_review})}\n\n"


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
    return StreamingResponse(
        stream_claude(raw_text, form_config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


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
    clean_fields = {k: v for k, v in fields.items() if v is not None and str(v).strip() != ""}
    xml_str, instance_id = build_submission_xml(clean_fields, form_config["asset_uid"])
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            KOBO_SUBMISSION_URL,
            headers={"Authorization": f"Token {KOBO_TOKEN}"},
            files={"xml_submission_file": ("submission.xml", xml_str.encode("utf-8"), "text/xml")}
        )
    if response.status_code in (200, 201):
        return {"success": True, "submission_id": instance_id, "form": form_slug,
                "form_title": form_config["form_title"]}
    raise HTTPException(status_code=502,
        detail=f"Kobo submission failed ({response.status_code}): {response.text[:500]}")


@app.get("/api/debug-kobo")
async def debug_kobo(form: str = Query("el-baseline")):
    if not KOBO_TOKEN:
        return {"error": "KOBO_TOKEN not set"}
    form_config = load_form_config(form)
    asset_uid = form_config["asset_uid"]
    async with httpx.AsyncClient(timeout=30.0) as client:
        asset_resp = await client.get(f"{KOBO_BASE_URL}/assets/{asset_uid}/",
                                      headers={"Authorization": f"Token {KOBO_TOKEN}"})
    xml_str, instance_id = build_submission_xml({"First_Name": "TEST_DEBUG", "Last_Name": "DELETE_ME"}, asset_uid)
    async with httpx.AsyncClient(timeout=30.0) as client:
        sub_resp = await client.post(KOBO_SUBMISSION_URL,
                                     headers={"Authorization": f"Token {KOBO_TOKEN}"},
                                     files={"xml_submission_file": ("submission.xml", xml_str.encode("utf-8"), "text/xml")})
    return {"form_tested": form,
            "asset_check": {"status": asset_resp.status_code},
            "xml_submission": {"status": sub_resp.status_code, "body": sub_resp.text[:300]}}


frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
