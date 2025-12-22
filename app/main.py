import asyncio
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, HttpUrl
from starlette.responses import StreamingResponse


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
FASTAPI_APP_TITLE = "Manuscript Translator"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 240.0
MAX_CONCURRENCY = 20

TRANSCRIBE_MODEL = "google/gemini-3-flash-preview"
TRANSLATE_MODEL = "google/gemini-3-pro-preview"


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


def _openrouter_client() -> AsyncOpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT_S,
    )


def _transcribe_messages(image_url: str) -> List[Dict[str, Any]]:
    system = (
        "Task: Transcribe the page EXACTLY as seen.\n"
        "Output MUST be ONLY the transcription as Markdown.\n"
        "Do not add any preamble, commentary, or extra sections.\n"
        "Preserve line breaks as best as possible.\n"
        "If a character/word is illegible, keep position with [?] and continue.\n"
        "Make inferences when appropriate, but put them in [brackets].\n"
    )
    user_text = (
        "Transcribe exactly."
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
        "Translate as faithfully as possible and [brackets] to provide modern context when needed."
    )
    user_text = (
        f"Translate all pages to English. There are {total_pages} pages.\n\n"
        f"{tagged_transcripts}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def _parse_page_blocks(text: str, total_pages: int) -> Dict[int, str]:
    """
    Parses <page_1>...</page_1> blocks. Returns mapping of 1-based page_number -> content.
    """
    out: Dict[int, str] = {}
    for m in TAG_RE_PAGE_BLOCK.finditer(text or ""):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        out[n] = (m.group(2) or "").strip()

    # Fallback: if model didn't use tags and there's only one page, treat whole output as page_1
    if not out and total_pages == 1 and (text or "").strip():
        out[1] = (text or "").strip()
    return out


async def _transcribe_one(
    client: AsyncOpenAI, *, image_url: str, semaphore: asyncio.Semaphore
) -> Tuple[str, str]:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=TRANSCRIBE_MODEL,
            messages=_transcribe_messages(image_url),
            temperature=0.2
        )
        content = (resp.choices[0].message.content or "").strip()
        return content, content


async def _translate_batch(
    client: AsyncOpenAI, *, tagged_transcripts: str, total_pages: int
) -> str:
    resp = await client.chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=_translate_batch_messages(tagged_transcripts, total_pages),
        temperature=1.0,
        extra_body={"reasoning": {"effort": "high"}}
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_completed_page_blocks(
    buffer: str, *, total_pages: int, emitted: set[int]
) -> Tuple[Dict[int, str], str]:
    """
    Finds any fully-closed <page_N>...</page_N> blocks in buffer that haven't been emitted yet.
    Returns (new_blocks, trimmed_buffer).
    """
    new: Dict[int, str] = {}
    last_end = 0
    for m in TAG_RE_PAGE_BLOCK.finditer(buffer or ""):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 1 <= n <= total_pages and n not in emitted:
            new[n] = (m.group(2) or "").strip()
            emitted.add(n)
        if m.end() > last_end:
            last_end = m.end()

    # Trim buffer to keep tail (in case the model is still writing the next page)
    if last_end > 0:
        buffer = buffer[last_end:]
    return new, buffer


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
        new_blocks, buffer = _extract_completed_page_blocks(buffer, total_pages=total_pages, emitted=emitted)
        for page_number, translation_md in new_blocks.items():
            yield page_number, translation_md

    # Flush anything remaining (in case the final tag closed right at the end)
    new_blocks, _ = _extract_completed_page_blocks(buffer, total_pages=total_pages, emitted=emitted)
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


@app.post("/api/process", response_model=ProcessResponse)
async def process(req: ProcessRequest) -> ProcessResponse:
    try:
        manifest = await _fetch_json(str(req.iiif_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch IIIF manifest: {e}")

    pages = _parse_manifest(manifest)
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="Could not find canvases/images in the manifest. Supported: IIIF Presentation v2 sequences[0].canvases or v3 items.",
        )

    try:
        client = _openrouter_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run_one(i: int, label: str, image_url: str) -> PageResult:
        transcript_md, raw = await _transcribe_one(client, image_url=image_url, semaphore=semaphore)
        return PageResult(
            page_number=i + 1,
            label=label or f"Page {i+1}",
            image_url=image_url,
            transcript_md=transcript_md,
            translation_md="",
            raw_model_output=raw,
        )

    tasks = [run_one(i, label, img) for i, (label, img) in enumerate(pages)]
    results = await asyncio.gather(*tasks)

    # Batch translate all transcripts in one request.
    tagged = "\n\n".join(
        f"<page_{p.page_number}>\n{p.transcript_md}\n</page_{p.page_number}>"
        for p in results
    )
    batch_out = await _translate_batch(client, tagged_transcripts=tagged, total_pages=len(results))
    translations = _parse_page_blocks(batch_out, total_pages=len(results))
    for p in results:
        p.translation_md = translations.get(p.page_number, "")

    return ProcessResponse(
        manifest_url=str(req.iiif_url),
        total_pages=len(results),
        pages=results,
    )


def _sse_event(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    # Basic SSE format
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


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
        # Keep-alive comment to encourage proxies to start streaming
        yield b": connected\n\n"

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

        # Send all images ASAP so frontend can render immediately
        manifest_payload = {
            "manifest_url": str(iiif_url),
            "total_pages": len(pages),
            "pages": [
                {"page_number": i + 1, "label": label or f"Page {i+1}", "image_url": img}
                for i, (label, img) in enumerate(pages)
            ],
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
            transcript_md, raw = await _transcribe_one(client, image_url=image_url, semaphore=semaphore)
            return PageResult(
                page_number=i + 1,
                label=label or f"Page {i+1}",
                image_url=image_url,
                transcript_md=transcript_md,
                translation_md="",
                raw_model_output=raw,
            )

        tasks = [asyncio.create_task(run_transcribe(i, label, img)) for i, (label, img) in enumerate(pages)]
        results: List[Optional[PageResult]] = [None] * len(tasks)

        try:
            for fut in asyncio.as_completed(tasks):
                if await request.is_disconnected():
                    for t in tasks:
                        t.cancel()
                    return
                try:
                    res: PageResult = await fut
                    results[res.page_number - 1] = res
                    completed += 1
                    # Stage 1: stream transcript as soon as each page is done
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
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


