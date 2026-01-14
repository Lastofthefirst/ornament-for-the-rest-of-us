"""
OCR Review Backend Server

Serves OCR page data, images, and proxies LLM analysis with tool calls.
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

app = FastAPI(title="OCR Review Server")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-20b")

# Current working directory - can be changed via API
current_ocr_dir: Path | None = None


class PageUpdate(BaseModel):
    text: str


class AnalyzeRequest(BaseModel):
    pageNumber: int
    text: str


class SetFolderRequest(BaseModel):
    path: str


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "suggest_fix",
            "description": "Suggest a fix for an OCR error or text issue. Call this for each issue found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": [
                            "garbled_text",
                            "ocr_confusion",
                            "missing_space",
                            "extra_space",
                            "broken_word",
                            "header_issue",
                            "formatting",
                            "spelling",
                            "repetition",
                            "truncation"
                        ],
                        "description": "Type of issue detected"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the issue"
                    },
                    "original_text": {
                        "type": "string",
                        "description": "The problematic text exactly as it appears"
                    },
                    "suggested_text": {
                        "type": "string",
                        "description": "The corrected text"
                    },
                    "start_offset": {
                        "type": "integer",
                        "description": "Character offset where the issue starts"
                    },
                    "end_offset": {
                        "type": "integer",
                        "description": "Character offset where the issue ends"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0-1 that this is a real issue"
                    }
                },
                "required": ["issue_type", "description", "original_text", "suggested_text", "start_offset", "end_offset", "confidence"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an OCR quality reviewer. Analyze the provided text for OCR errors and quality issues.

Common OCR issues to look for:
1. Garbled text - nonsensical characters or corrupted words
2. OCR confusion - common substitutions like 0/O, 1/l/I, rn/m, cl/d
3. Missing spaces - words run together (e.g., "theword" should be "the word")
4. Extra spaces - spurious spaces in words (e.g., "wo rd" should be "word")
5. Broken words - words split incorrectly across lines with hyphens
6. Header issues - running headers/footers mixed into body text
7. Formatting - markdown formatting problems
8. Spelling errors that are likely OCR misreads
9. Repetition - repeated words or phrases (hallucination artifacts)
10. Truncation - text that appears cut off

For EACH issue found, call the suggest_fix tool with:
- The exact original text (matching character positions)
- Your suggested correction
- Start and end character offsets (0-indexed)
- Your confidence level

Focus on clear errors. Don't flag stylistic choices or archaic spellings unless clearly wrong.
If the text looks correct, you don't need to suggest anything."""


def find_ocr_pages_dir(base_path: Path) -> Path:
    """Find the directory containing OCR page files."""
    # Check if base_path directly contains page files
    if list(base_path.glob("page_*.md")) or list(base_path.glob("*_page_*.md")):
        return base_path

    # Check common subdirectory names
    for subdir in ["ocr_pages", "pages", "output"]:
        candidate = base_path / subdir
        if candidate.exists() and (list(candidate.glob("page_*.md")) or list(candidate.glob("*_page_*.md"))):
            return candidate

    # Fallback to base path
    return base_path


def get_page_pattern(ocr_dir: Path) -> tuple[str, callable]:
    """Detect the page naming pattern and return regex + number extractor."""
    # Check for page_0000 pattern (zero-padded)
    if list(ocr_dir.glob("page_0*.md")):
        return r'page_(\d+)\.md$', lambda m: int(m.group(1))

    # Check for prefix_page_N pattern
    prefixed = list(ocr_dir.glob("*_page_*.md"))
    if prefixed:
        # Extract prefix from first file
        sample = prefixed[0].name
        match = re.match(r'(.+)_page_(\d+)\.md$', sample)
        if match:
            prefix = match.group(1)
            return rf'{re.escape(prefix)}_page_(\d+)\.md$', lambda m: int(m.group(1))

    # Default pattern
    return r'page_(\d+)\.md$', lambda m: int(m.group(1))


def get_pages():
    """Load all pages from the current OCR output directory."""
    if not current_ocr_dir:
        return []

    ocr_dir = find_ocr_pages_dir(current_ocr_dir)
    pattern, num_extractor = get_page_pattern(ocr_dir)

    pages = []
    md_files = []

    # Find all .md files (not _nohf.md)
    for f in ocr_dir.iterdir():
        if f.suffix == '.md' and not f.name.endswith('_nohf.md'):
            match = re.search(pattern, f.name)
            if match:
                md_files.append((f, num_extractor(match)))

    # Sort by page number
    md_files.sort(key=lambda x: x[1])

    for md_file, page_num in md_files:
        base_name = md_file.stem

        # Find corresponding files
        json_file = ocr_dir / f"{base_name}.json"
        img_file = ocr_dir / f"{base_name}.jpg"

        # Also check for images in processed_images folder
        if not img_file.exists():
            # Try page_NNNN.jpg format in processed_images
            processed_dir = current_ocr_dir / "processed_images"
            if processed_dir.exists():
                img_file = processed_dir / f"page_{page_num:04d}.jpg"
                if not img_file.exists():
                    img_file = processed_dir / f"page_{page_num}.jpg"

        # Load text
        try:
            text = md_file.read_text(encoding='utf-8')
        except Exception:
            text = ""

        # Load blocks from JSON if exists
        blocks = []
        if json_file.exists():
            try:
                blocks = json.loads(json_file.read_text(encoding='utf-8'))
            except Exception:
                pass

        pages.append({
            "pageNumber": page_num,
            "imagePath": str(img_file) if img_file.exists() else None,
            "markdownPath": str(md_file),
            "jsonPath": str(json_file) if json_file.exists() else None,
            "blocks": blocks,
            "text": text,
            "hasIssues": False,
            "issues": []
        })

    return pages


@app.get("/status")
async def get_status():
    """Get current folder status."""
    if current_ocr_dir:
        pages = get_pages()
        return {
            "folder": str(current_ocr_dir),
            "pageCount": len(pages),
            "ready": True
        }
    return {
        "folder": None,
        "pageCount": 0,
        "ready": False
    }


@app.post("/folder")
async def set_folder(request: SetFolderRequest):
    """Set the OCR output folder to load."""
    global current_ocr_dir

    path = Path(request.path).expanduser().resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {path}")

    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    current_ocr_dir = path
    pages = get_pages()

    if not pages:
        raise HTTPException(status_code=400, detail=f"No OCR pages found in: {path}")

    return {
        "folder": str(path),
        "pageCount": len(pages),
        "ready": True
    }


@app.get("/pages")
async def list_pages():
    """Get all pages with their text content."""
    if not current_ocr_dir:
        return {"pages": [], "folder": None}

    pages = get_pages()
    return {"pages": pages, "folder": str(current_ocr_dir)}


@app.get("/image/{page_num}")
async def get_image(page_num: int):
    """Serve a page image."""
    if not current_ocr_dir:
        raise HTTPException(status_code=400, detail="No folder selected")

    pages = get_pages()

    # Find the page
    for page in pages:
        if page["pageNumber"] == page_num:
            if page["imagePath"] and Path(page["imagePath"]).exists():
                return FileResponse(page["imagePath"], media_type="image/jpeg")
            break

    raise HTTPException(status_code=404, detail=f"Image not found for page {page_num}")


@app.put("/pages/{page_num}")
async def update_page(page_num: int, update: PageUpdate):
    """Save updated text for a page."""
    if not current_ocr_dir:
        raise HTTPException(status_code=400, detail="No folder selected")

    pages = get_pages()

    for page in pages:
        if page["pageNumber"] == page_num:
            md_path = Path(page["markdownPath"])
            md_path.write_text(update.text, encoding='utf-8')
            return {"success": True, "pageNumber": page_num}

    raise HTTPException(status_code=404, detail=f"Page {page_num} not found")


@app.post("/analyze")
async def analyze_page(request: AnalyzeRequest):
    """Analyze page text with LLM and return suggested fixes."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please analyze this OCR text for errors and suggest fixes:\n\n{request.text}"}
    ]

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                LLM_API_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "temperature": 0.1,
                    "max_tokens": 4096
                }
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    # Extract tool calls from response
    issues = []
    message = result.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    for tc in tool_calls:
        if tc.get("function", {}).get("name") == "suggest_fix":
            try:
                args = json.loads(tc["function"]["arguments"])
                severity = "low"
                if args.get("confidence", 0) > 0.8:
                    severity = "high"
                elif args.get("confidence", 0) > 0.5:
                    severity = "medium"

                issues.append({
                    "id": str(uuid.uuid4()),
                    "type": args.get("issue_type", "unknown"),
                    "severity": severity,
                    "description": args.get("description", ""),
                    "startOffset": args.get("start_offset", 0),
                    "endOffset": args.get("end_offset", 0),
                    "suggestion": args.get("suggested_text", ""),
                    "status": "pending"
                })
            except (json.JSONDecodeError, KeyError):
                continue

    return {"issues": issues}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "folder": str(current_ocr_dir) if current_ocr_dir else None,
        "llm_url": LLM_API_URL
    }


if __name__ == "__main__":
    import uvicorn
    import sys

    # Allow passing folder as command line arg
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1]).expanduser().resolve()
        if folder.exists():
            current_ocr_dir = folder
            print(f"Loaded folder: {current_ocr_dir}")

    print(f"Starting server on port 8787...")
    print(f"LLM API URL: {LLM_API_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8787)
