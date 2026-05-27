# Kobo Scan App
### Emerging Leaders — Handwritten Form → KoboToolbox Automation

---

## What This Does
Data collectors take a photo of a completed paper form on their phone.
The app scans it, extracts all fields using AI, shows a review screen,
and submits directly to KoboToolbox. No manual data entry needed.

---

## Architecture
```
Phone Camera / Photo
       ↓
   Web App (this app — opens in any browser via link)
       ↓
   Google Vision OCR (extracts all text from the image)
       ↓
   Claude AI (maps OCR text → Kobo field values)
       ↓
   Data Collector Reviews & Confirms
       ↓
   KoboToolbox API (submission)
```

---

## Project Structure
```
kobo-scan-app/
├── backend/
│   ├── main.py              # FastAPI backend (3 endpoints)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-file web app (works on any phone)
├── config/
│   └── field_map.json       # Kobo form field definitions
├── .env.example             # Environment variables template
├── render.yaml              # Render.com deployment config
└── README.md
```

---

## Setup — Step by Step

### 1. Get Your Google Vision API Key
1. Go to https://console.cloud.google.com
2. Create a new project (or use existing)
3. Enable **Cloud Vision API**
4. Go to Credentials → Create Credentials → API Key
5. Copy the key

### 2. Get Your Kobo API Token
1. Log into https://kf.kobotoolbox.org
2. Go to Account Settings (top right)
3. Scroll to **API Token**
4. Copy the token

### 3. Deploy to Render.com
1. Push this project to a GitHub repository
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set environment variables in Render dashboard:
   - `KOBO_TOKEN` = your Kobo token
   - `GOOGLE_VISION_API_KEY` = your Google Vision key
5. Deploy — Render will give you a URL like `https://kobo-scan-app.onrender.com`

### 4. Share the Link
Share `https://your-app-name.onrender.com` with data collectors.
That's it. No app installation. Works on any phone browser.

---

## Adding a New Project/Form
1. Update `config/field_map.json` with new `asset_uid` and fields
2. Or create separate config files per project and use `?form=project_code`
   (multi-form support can be added in a future version)

---

## Local Development
```bash
cd kobo-scan-app
cp .env.example .env
# Fill in .env values

pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# Open http://localhost:8000
```

---

## Known Limitations
- Requires internet connection (no offline mode)
- OCR accuracy depends on handwriting clarity and photo quality
- The preview/review step is mandatory — collectors must confirm before submitting
- Free tier on Render may sleep after 15 minutes of inactivity (first request takes ~30s)
  - Upgrade to $7/month paid tier to avoid this in production
