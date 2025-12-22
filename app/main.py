import asyncio
import base64
import io
import time
import uuid
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, HttpUrl
from starlette.responses import StreamingResponse, Response
from PIL import Image
import cv2
import numpy as np


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
FASTAPI_APP_TITLE = "Manuscript Translator"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 240.0
MAX_CONCURRENCY = 20

TRANSCRIBE_MODEL = "google/gemini-3-pro-preview"
TRANSLATE_MODEL = "google/gemini-3-pro-preview"
CROP_MODEL = "google/gemini-3-flash-preview"

# In-memory cache for cropped images (job-scoped).
_CROP_CACHE_TTL_S = 60 * 60
_crop_cache: Dict[str, Dict[str, Any]] = {}


app = FastAPI(title=FASTAPI_APP_TITLE)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)


class ProcessRequest(BaseModel):
    iiif_url: HttpUrl = Field(..., description="IIIF manifest URL (Presentation 2 or 3).")


class PageResult(BaseModel):
    page_number: int
    label: str
    image_url: str
    original_image_url: str = ""
    enhanced_image_url: str = ""
    transcript_md: str
    translation_md: str
    raw_model_output: str


class ProcessResponse(BaseModel):
    manifest_url: str
    total_pages: int
    pages: List[PageResult]


def _first_present(*vals: Optional[str]) -> str:
    for v in vals:
        if v:
            return v
    return ""


def _iiif_label_to_text(label_obj: Any) -> str:
    """
    IIIF v3 label: {"none":["foo"]} or {"en":["foo"]}. v2 label: string.
    """
    if label_obj is None:
        return ""
    if isinstance(label_obj, str):
        return label_obj
    if isinstance(label_obj, dict):
        for _, arr in label_obj.items():
            if isinstance(arr, list) and arr:
                return str(arr[0])
    return str(label_obj)


def _image_url_from_service(service_id: str) -> str:
    # Standard IIIF Image API pattern
    service_id = service_id.rstrip("/")
    # Prefer "max" sizing (commonly supported and avoids requesting unbounded "full" size).
    return f"{service_id}/full/max/0/default.jpg"


def _extract_canvases_v2(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    sequences = manifest.get("sequences") or []
    if not sequences:
        return []
    first_seq = sequences[0] or {}
    return first_seq.get("canvases") or []


def _extract_canvases_v3(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    # v3: manifest.items = canvases
    items = manifest.get("items") or []
    return items if isinstance(items, list) else []


def _get_canvas_image_url(canvas: Dict[str, Any]) -> str:
    """
    Try IIIF Presentation 2 and 3 shapes.
    """
    # v2: canvas.images[0].resource.service["@id"] or canvas.images[0].resource["@id"]
    images = canvas.get("images")
    if isinstance(images, list) and images:
        anno = images[0] or {}
        resource = anno.get("resource") or {}
        service = resource.get("service") or {}
        if isinstance(service, list) and service:
            service = service[0]
        service_id = _first_present(service.get("@id"), service.get("id"))
        if service_id:
            return _image_url_from_service(service_id)
        direct = _first_present(resource.get("@id"), resource.get("id"))
        if direct:
            return direct

    # v3: canvas.items[0].items[0].body.id (preferred) or body.service[0].id
    items = canvas.get("items")
    if isinstance(items, list) and items:
        anno_page = items[0] or {}
        annos = anno_page.get("items")
        if isinstance(annos, list) and annos:
            anno = annos[0] or {}
            body = anno.get("body") or {}
            # Prefer the concrete image URL if present (many manifests provide a good /full/max/ URL here).
            direct = _first_present(body.get("id"), body.get("@id"))
            if direct:
                return direct
            service = body.get("service") or []
            if isinstance(service, dict):
                service = [service]
            if isinstance(service, list) and service:
                s0 = service[0] or {}
                service_id = _first_present(s0.get("id"), s0.get("@id"))
                if service_id:
                    return _image_url_from_service(service_id)

    return ""


def _parse_manifest(manifest: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns list of (label, image_url) in reading order, from the first sequence (v2)
    or manifest.items (v3).
    """
    canvases = _extract_canvases_v2(manifest)
    if not canvases:
        canvases = _extract_canvases_v3(manifest)
    out: List[Tuple[str, str]] = []
    for c in canvases:
        label = _iiif_label_to_text(c.get("label")) or _first_present(c.get("@id"), c.get("id"), "Canvas")
        img = _get_canvas_image_url(c)
        if img:
            out.append((label, img))
    return out


TAG_RE_TRANSCRIPT = re.compile(r"<transcript>\s*(.*?)\s*</transcript>", re.DOTALL | re.IGNORECASE)
TAG_RE_TRANSLATION = re.compile(r"<translation>\s*(.*?)\s*</translation>", re.DOTALL | re.IGNORECASE)
TAG_RE_PAGE_BLOCK = re.compile(
    r"<page[_-]?(\d+)>\s*([\s\S]*?)\s*</page[_-]?\1>",
    re.IGNORECASE,
)
TAG_RE_PAGE_START = re.compile(r"<page[_-]?(\d+)>", re.IGNORECASE)


def _openrouter_client() -> AsyncOpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT_S,
    )


def _cleanup_crop_cache() -> None:
    now = time.time()
    dead: List[str] = []
    for job_id, blob in _crop_cache.items():
        created = float(blob.get("created", 0))
        if now - created > _CROP_CACHE_TTL_S:
            dead.append(job_id)
    for job_id in dead:
        _crop_cache.pop(job_id, None)


def _cache_cropped(job_id: str, page_number: int, img_bytes: bytes, media_type: str) -> None:
    _crop_cache.setdefault(job_id, {"created": time.time(), "pages": {}})
    _crop_cache[job_id]["pages"].setdefault(int(page_number), {})
    _crop_cache[job_id]["pages"][int(page_number)]["enhanced"] = {"bytes": img_bytes, "media_type": media_type}


def _get_cached_cropped(job_id: str, page_number: int) -> Optional[Dict[str, Any]]:
    blob = _crop_cache.get(job_id)
    if not blob:
        return None
    page = (blob.get("pages") or {}).get(int(page_number)) or {}
    return page.get("enhanced")


def _cache_original(job_id: str, page_number: int, img_bytes: bytes, media_type: str) -> None:
    _crop_cache.setdefault(job_id, {"created": time.time(), "pages": {}})
    _crop_cache[job_id]["pages"].setdefault(int(page_number), {})
    _crop_cache[job_id]["pages"][int(page_number)]["original"] = {"bytes": img_bytes, "media_type": media_type}


def _get_cached_original(job_id: str, page_number: int) -> Optional[Dict[str, Any]]:
    blob = _crop_cache.get(job_id)
    if not blob:
        return None
    page = (blob.get("pages") or {}).get(int(page_number)) or {}
    return page.get("original")


async def _fetch_image_bytes(url: str) -> Tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, follow_redirects=True) as client:
        r = await client.get(url, headers={"Accept": "image/*"})
        r.raise_for_status()
        media_type = r.headers.get("content-type", "image/jpeg").split(";")[0].strip() or "image/jpeg"
        return r.content, media_type


def _to_data_url(img_bytes: bytes, media_type: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{media_type};base64,{b64}"


def _parse_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(s[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("could not parse JSON object")


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(n)))


def _crop_with_box(img_bytes: bytes, box_2d: List[int], *, pad_px: int = 8) -> bytes:
    """
    box_2d: [ymin,xmin,ymax,xmax] normalized 0-1000.
    Returns cropped JPEG bytes.
    """
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    if w <= 1 or h <= 1:
        return img_bytes
    if not isinstance(box_2d, list) or len(box_2d) != 4:
        return img_bytes

    y1n, x1n, y2n, x2n = [int(v) for v in box_2d]
    x1 = int(x1n / 1000 * w)
    y1 = int(y1n / 1000 * h)
    x2 = int(x2n / 1000 * w)
    y2 = int(y2n / 1000 * h)

    x1 = _clamp(x1 - pad_px, 0, w - 1)
    y1 = _clamp(y1 - pad_px, 0, h - 1)
    x2 = _clamp(x2 + pad_px, x1 + 1, w)
    y2 = _clamp(y2 + pad_px, y1 + 1, h)

    crop = im.crop((x1, y1, x2, y2))
    out = io.BytesIO()
    crop.save(out, format="JPEG", quality=98, optimize=True)
    return out.getvalue()


def _apply_clahe_lab_l(jpeg_bytes: bytes) -> bytes:
    """
    Soft enhancement + CLAHE on the LAB L channel:
    - Background normalization to reduce uneven illumination
    - Soft ink/paper separation (Sauvola-like) blended to darken ink and slightly lift paper
    Input/output are JPEG bytes.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jpeg_bytes

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 1) Background normalization (smooth illumination field) — keep gentle
    l_f = l_chan.astype(np.float32)
    bg = cv2.GaussianBlur(l_f, (0, 0), sigmaX=35.0)
    bg = np.clip(bg, 1.0, 255.0)
    ln = (l_f / bg) * 255.0
    ln = np.clip(ln, 0.0, 255.0)

    # 2) Sauvola-like local threshold => soft ink mask (0..1)
    win = 45
    mean = cv2.boxFilter(ln, ddepth=-1, ksize=(win, win), normalize=True)
    mean2 = cv2.boxFilter(ln * ln, ddepth=-1, ksize=(win, win), normalize=True)
    var = np.maximum(0.0, mean2 - mean * mean)
    std = np.sqrt(var)
    k = 0.12
    R = 128.0
    thresh = mean * (1.0 + k * (std / R - 1.0))

    d = (thresh - ln)  # positive => likely ink
    softness = 20.0
    mask = 1.0 / (1.0 + np.exp(-d / softness))

    # 3) Blend (soft enhancement)
    ink_dark = 12.0
    paper_lift = 2.0
    out_l = ln - ink_dark * mask + paper_lift * (1.0 - mask)
    # Blend back toward original L to keep the effect subtle.
    alpha = 0.55
    out_l = (1.0 - alpha) * l_f + alpha * out_l
    out_l = np.clip(out_l, 0.0, 255.0).astype(np.uint8)

    # 4) Very strong sharpening AFTER soft enhancement (luminance-only).
    # Warning: this can introduce halos/noise; reduce weights/sigma if it looks too harsh.
    out_f2 = out_l.astype(np.float32)
    blur2 = cv2.GaussianBlur(out_f2, (0, 0), sigmaX=6.0)
    out_sharp = cv2.addWeighted(out_f2, 1.55, blur2, -0.55, 0.0)
    out_l = np.clip(out_sharp, 0.0, 255.0).astype(np.uint8)

    # 5) Gentle CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(64, 64))
    out_l = clahe.apply(out_l)

    lab2 = cv2.merge((out_l, a_chan, b_chan))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr2, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    if not ok:
        return jpeg_bytes
    return enc.tobytes()


def _crop_prompt() -> str:
    return (
        "Return ONE bounding box that tightly contains the main body text.\n"
        "Ignore decorative margins, page borders, stains, and minor marginal notes if they are small compared to the main text.\n"
        "Output MUST be valid JSON with exactly this shape:\n"
        "{\"box_2d\":[ymin,xmin,ymax,xmax]}\n"
        "Where box_2d values are integers normalized to 0-1000 in [ymin,xmin,ymax,xmax] order.\n"
        "Do not include any other keys or any extra text."
    )


async def _detect_crop_box(
    client: AsyncOpenAI, *, image_data_url: str, semaphore: asyncio.Semaphore
) -> List[int]:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=CROP_MODEL,
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _crop_prompt()},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        obj = _parse_json_object(content)
        box = obj.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid box_2d: {box}")
        return [int(v) for v in box]


def _transcribe_messages(image_url: str) -> List[Dict[str, Any]]:
    system = (
        "Task: Transcribe the page faithfully.\n"
        "Output MUST be ONLY the transcription as Markdown.\n"
        "Do not add any preamble, commentary, or extra sections.\n"
        "Preserve formatting as best as possible.\n"
        "For text that you can see but is ambiguous, make your best guesses like [guess 1/guess 2].\n"
        "For text sections or words that are entirely illegible, use [illegible] as a placeholder and continue.\n"
        "Make inferences when appropriate, but put them in [brackets].\n"
        "You should use context clues to help you, as many of the letters will be difficult to make out.\n"
        "Your objective is to capture as much of the text as possible."
    )
    user_text = (
        "Transcribe this page."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def _translate_batch_messages(tagged_transcripts: str, total_pages: int) -> List[Dict[str, Any]]:
    system = (
        "Task: Translate each page transcript to English.\n"
        "Input is a set of XML-like blocks: <page_1>...</page_1>, <page_2>...</page_2>, etc.\n"
        "Output MUST contain translations for EVERY page in the SAME format: <page_1>...</page_1> ...\n"
        "Inside each <page_N> tag, use Markdown to format based on the author's intent (rather than matching OCR line for line).\n"
        "Do not include any other text outside the <page_N> tags.\n"
        "If something is unclear, keep uncertainty in [brackets].\n"
        "OCR was used for the transcription and may have errors. Use context clues to correct them.\n"
        "Translate as faithfully and use [brackets] to provide modern context when appropriate.\n"
        "You should attempt to translate as much of the text as possible. Heavy annotation is desired when parts are unclear or missing."
    )
    user_text = (
        f"Translate all pages to English. There are {total_pages} pages.\n\n"
        f"{tagged_transcripts}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


async def _transcribe_one(
    client: AsyncOpenAI, *, image_url: str, semaphore: asyncio.Semaphore
) -> Tuple[str, str]:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=TRANSCRIBE_MODEL,
            messages=_transcribe_messages(image_url),
            temperature=1.0,
            extra_body={"reasoning": {"effort": "low"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        return content, content


def _pop_completed_page_blocks_incremental(
    buffer: str, *, total_pages: int, emitted: set[int]
) -> Tuple[Dict[int, str], str]:
    """
    Robust incremental extraction of <page_N>...</page_N> blocks.

    - Maps strictly by the numeric N in the tag (NOT by output order).
    - Avoids the "wrong page" issue when the model temporarily emits malformed/out-of-order tags while streaming.
    """
    out: Dict[int, str] = {}
    if not buffer:
        return out, buffer

    while True:
        m = TAG_RE_PAGE_START.search(buffer)
        if not m:
            # Keep buffer bounded
            if len(buffer) > 200_000:
                buffer = buffer[-200_000:]
            return out, buffer

        start_idx = m.start()
        tag_end = m.end()
        try:
            n = int(m.group(1))
        except Exception:
            # Skip malformed tag
            buffer = buffer[tag_end:]
            continue

        end_tag = f"</page_{n}>"
        end_tag_alt = f"</page-{n}>"

        end_idx = buffer.lower().find(end_tag, tag_end)
        end_len = len(end_tag)
        if end_idx < 0:
            end_idx = buffer.lower().find(end_tag_alt, tag_end)
            end_len = len(end_tag_alt)

        # If we can't find the matching end tag yet, but there is another page start after this one,
        # assume this tag was malformed/unclosed and skip forward to the next start to avoid blocking.
        if end_idx < 0:
            next_start = TAG_RE_PAGE_START.search(buffer, tag_end)
            if next_start:
                buffer = buffer[next_start.start():]
                continue
            # Otherwise, wait for more chunks; keep from this start tag
            buffer = buffer[start_idx:]
            return out, buffer

        content = buffer[tag_end:end_idx].strip()
        if 1 <= n <= total_pages and n not in emitted:
            out[n] = content
            emitted.add(n)

        # Consume through end tag and continue scanning
        buffer = buffer[end_idx + end_len :]


async def _translate_batch_stream(
    client: AsyncOpenAI, *, tagged_transcripts: str, total_pages: int
):
    """
    Streams translation output and yields (page_number, translation_md) as soon as each
    <page_N>...</page_N> block is complete.
    """
    stream = await client.chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=_translate_batch_messages(tagged_transcripts, total_pages),
        temperature=1.0,
        extra_body={"reasoning": {"effort": "high"}},
        stream=True,
    )

    buffer = ""
    emitted: set[int] = set()

    async for event in stream:
        # ChatCompletionChunk shape: choices[0].delta.content
        try:
            delta = event.choices[0].delta
            chunk = getattr(delta, "content", None) or ""
        except Exception:
            chunk = ""

        if not chunk:
            continue

        buffer += chunk
        new_blocks, buffer = _pop_completed_page_blocks_incremental(
            buffer, total_pages=total_pages, emitted=emitted
        )
        for page_number, translation_md in new_blocks.items():
            yield page_number, translation_md

    # Flush anything remaining (in case the final tag closed right at the end)
    new_blocks, _ = _pop_completed_page_blocks_incremental(
        buffer, total_pages=total_pages, emitted=emitted
    )
    for page_number, translation_md in new_blocks.items():
        yield page_number, translation_md


async def _fetch_json(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, follow_redirects=True) as client:
        r = await client.get(
            url,
            headers={
                # IIIF manifests are sometimes served as JSON-LD.
                "Accept": "application/ld+json, application/json;q=0.9, */*;q=0.1",
                "User-Agent": "Manuscript-Translator/1.0 (+FastAPI)",
            },
        )
        r.raise_for_status()
        return r.json()


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    template = templates_env.get_template("index.html")
    html = template.render()
    return HTMLResponse(html)


@app.get("/api/crop/{job_id}/{page_number}")
async def get_crop(job_id: str, page_number: int) -> Response:
    _cleanup_crop_cache()
    item = _get_cached_cropped(job_id, page_number)
    if not item:
        raise HTTPException(status_code=404, detail="Cropped image not ready")
    return Response(content=item["bytes"], media_type=item["media_type"])


@app.get("/api/original/{job_id}/{page_number}")
async def get_original(job_id: str, page_number: int) -> Response:
    _cleanup_crop_cache()
    item = _get_cached_original(job_id, page_number)
    if not item:
        raise HTTPException(status_code=404, detail="Original image not ready")
    return Response(content=item["bytes"], media_type=item["media_type"])


def _sse_event(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    # Basic SSE format
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _sse_comment(text: str) -> bytes:
    # SSE comment line (keeps connection alive through proxies)
    return f": {text}\n\n".encode("utf-8")


@app.get("/api/stream")
async def stream(iiif_url: HttpUrl, request: Request) -> StreamingResponse:
    """
    Streams:
    - event: manifest  data: { manifest_url, total_pages, pages:[{page_number,label,image_url}] }
    - event: page      data: {page_number,label,image_url,transcript_md,translation_md,raw_model_output}
    - event: page_error data: {page_number,label,image_url,error}
    - event: done      data: {total_pages, completed}
    """

    async def gen():
        # Keep-alive + retry hint to encourage proxies/browsers to keep streaming
        yield b"retry: 3000\n"
        yield _sse_comment("connected")

        try:
            manifest = await _fetch_json(str(iiif_url))
        except Exception as e:
            yield _sse_event("error", {"detail": f"Failed to fetch IIIF manifest: {e}"})
            return

        pages = _parse_manifest(manifest)
        if not pages:
            yield _sse_event(
                "error",
                {
                    "detail": "Could not find canvases/images in the manifest. Supported: IIIF Presentation v2 sequences[0].canvases or v3 items."
                },
            )
            return

        _cleanup_crop_cache()
        job_id = uuid.uuid4().hex

        # Send all raw images ASAP; we'll stream per-page updates with cropped image URLs as they become ready.
        manifest_payload = {
            "manifest_url": str(iiif_url),
            "total_pages": len(pages),
            "pages": [
                {
                    "page_number": i + 1,
                    "label": label or f"Page {i+1}",
                    "image_url": img,
                    "original_image_url": img,
                    "enhanced_image_url": "",
                }
                for i, (label, img) in enumerate(pages)
            ],
            "job_id": job_id,
        }
        yield _sse_event("manifest", manifest_payload)

        try:
            client = _openrouter_client()
        except Exception as e:
            yield _sse_event("error", {"detail": str(e)})
            return

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        completed = 0

        async def run_transcribe(i: int, label: str, image_url: str) -> PageResult:
            # Preprocess: crop to main body text (ignoring minor margins), then transcribe cropped image.
            raw_bytes, raw_media = await _fetch_image_bytes(image_url)
            raw_data_url = _to_data_url(raw_bytes, raw_media)

            cropped_bytes: bytes = raw_bytes
            try:
                box = await _detect_crop_box(client, image_data_url=raw_data_url, semaphore=semaphore)
                cropped_bytes = _crop_with_box(raw_bytes, box)
            except Exception:
                # If crop detection fails, fall back to full image.
                cropped_bytes = raw_bytes

            # Cache original + enhanced (CLAHE on cropped)
            _cache_original(job_id, i + 1, raw_bytes, raw_media)
            enhanced_bytes = _apply_clahe_lab_l(cropped_bytes)
            _cache_cropped(job_id, i + 1, enhanced_bytes, "image/jpeg")
            cropped_data_url = _to_data_url(enhanced_bytes, "image/jpeg")

            transcript_md, raw = await _transcribe_one(client, image_url=cropped_data_url, semaphore=semaphore)
            return PageResult(
                page_number=i + 1,
                label=label or f"Page {i+1}",
                image_url=f"/api/crop/{job_id}/{i+1}",
                original_image_url=f"/api/original/{job_id}/{i+1}",
                enhanced_image_url=f"/api/crop/{job_id}/{i+1}",
                transcript_md=transcript_md,
                translation_md="",
                raw_model_output=raw,
            )

        tasks = [asyncio.create_task(run_transcribe(i, label, img)) for i, (label, img) in enumerate(pages)]
        results: List[Optional[PageResult]] = [None] * len(tasks)

        try:
            pending = set(tasks)
            # Emit a keepalive at least this often while work is in progress
            keepalive_s = 15.0
            while pending:
                if await request.is_disconnected():
                    for t in tasks:
                        t.cancel()
                    return

                done, pending = await asyncio.wait(pending, timeout=keepalive_s, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    yield _sse_comment("ping")
                    continue

                for fut in done:
                    try:
                        res: PageResult = await fut
                        results[res.page_number - 1] = res
                        completed += 1
                        # Stage 1: stream transcript (and enhanced/original urls) as soon as each page is done
                        yield _sse_event("page", res.model_dump())
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        yield _sse_event("page_error", {"error": str(e)})
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

        # Stage 2: batch translate once we have all transcripts (stream per-page as tags complete)
        if await request.is_disconnected():
            return

        transcripts = [r for r in results if r is not None]
        if len(transcripts) != len(pages):
            yield _sse_event(
                "error",
                {"detail": f"Only transcribed {len(transcripts)}/{len(pages)} pages; skipping batch translation."},
            )
            yield _sse_event("done", {"total_pages": len(pages), "completed": completed})
            return

        tagged = "\n\n".join(
            f"<page_{p.page_number}>\n{p.transcript_md}\n</page_{p.page_number}>"
            for p in transcripts
        )
        try:
            page_map: Dict[int, PageResult] = {p.page_number: p for p in transcripts}
            async for page_number, translation_md in _translate_batch_stream(
                client, tagged_transcripts=tagged, total_pages=len(pages)
            ):
                if await request.is_disconnected():
                    return
                p = page_map.get(page_number)
                if not p:
                    continue
                p.translation_md = translation_md
                yield _sse_event(
                    "page",
                    {
                        "page_number": p.page_number,
                        "label": p.label,
                        "image_url": p.image_url,
                        "original_image_url": p.original_image_url,
                        "enhanced_image_url": p.enhanced_image_url,
                        "transcript_md": p.transcript_md,
                        "translation_md": p.translation_md,
                        "raw_model_output": p.raw_model_output,
                    },
                )

            missing = [n for n in range(1, len(pages) + 1) if not page_map.get(n) or not page_map[n].translation_md]
            if missing:
                yield _sse_event("error", {"detail": f"Missing translations for pages: {missing}"})
        except Exception as e:
            yield _sse_event("error", {"detail": f"Batch translation failed: {e}"})

        yield _sse_event("done", {"total_pages": len(pages), "completed": completed})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Note: Connection header is ignored under HTTP/2/3, but harmless under HTTP/1.1.
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


