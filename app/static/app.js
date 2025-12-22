/* global marked, DOMPurify */

const els = {
  form: document.getElementById("iiif-form"),
  url: document.getElementById("iiif-url"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  tpl: document.getElementById("page-template"),
  btn: document.getElementById("submit-btn"),
  pdfBtn: document.getElementById("pdf-btn"),
  modeTranslation: document.getElementById("mode-translation"),
  modeTranscript: document.getElementById("mode-transcript"),
};

let lastPages = [];
let viewMode = "translation"; // default
let evtSource = null;
let streamDone = false;

marked.setOptions({
  gfm: true,
  breaks: true, // preserve single line breaks (common for transcripts)
  headerIds: false,
  mangle: false,
});

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function stripCodeFromHtml(html) {
  const doc = new DOMParser().parseFromString(html || "", "text/html");

  // Replace fenced/indented code blocks with plain text (preserve line breaks).
  doc.querySelectorAll("pre").forEach((pre) => {
    const text = pre.textContent || "";
    const div = doc.createElement("div");
    div.className = "md-plain";
    div.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
    pre.replaceWith(div);
  });

  // Replace inline code with plain text.
  doc.querySelectorAll("code").forEach((code) => {
    const span = doc.createElement("span");
    span.textContent = code.textContent || "";
    code.replaceWith(span);
  });

  return doc.body.innerHTML;
}

function setStatus(text, kind = "info") {
  els.status.textContent = text || "";
  els.status.className = `status status-${kind}`;
}

function setMode(mode) {
  viewMode = mode;
  if (mode === "translation") {
    els.modeTranslation.classList.add("is-active");
    els.modeTranscript.classList.remove("is-active");
  } else {
    els.modeTranslation.classList.remove("is-active");
    els.modeTranscript.classList.add("is-active");
  }
  renderPages(lastPages);
}

function mdToHtml(md) {
  // Convert markdown -> HTML then sanitize. Do NOT HTML-escape first, or it breaks markdown syntax like '>' blockquotes.
  const html = marked.parse(md || "");
  if (typeof DOMPurify !== "undefined") {
    const safe = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    return stripCodeFromHtml(safe);
  }
  // Fallback (shouldn't happen since we load DOMPurify via CDN):
  return stripCodeFromHtml(html);
}

function renderPages(pages) {
  els.results.innerHTML = "";
  if (!pages || pages.length === 0) return;

  for (const p of pages) {
    const node = els.tpl.content.cloneNode(true);
    node.querySelector(".page-title").textContent = `Page ${p.page_number}: ${p.label || ""}`.trim();
    node.querySelector(".page-subtitle").textContent = p.image_url;
    const img = node.querySelector(".page-img");
    img.src = p.image_url;
    img.alt = `Page ${p.page_number}`;

    const content = viewMode === "translation" ? p.translation_md : p.transcript_md;
    node.querySelector(".content-inner").innerHTML = mdToHtml(content || "");

    els.results.appendChild(node);
  }
}

function closeStream() {
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
}

function allTranslationsReady() {
  return (
    Array.isArray(lastPages) &&
    lastPages.length > 0 &&
    lastPages.every((p) => (p.translation_md || "").trim().length > 0)
  );
}

function updatePdfButtonState() {
  if (!els.pdfBtn) return;
  els.pdfBtn.disabled = !(streamDone && allTranslationsReady());
}

function htmlToPlainText(html) {
  const doc = new DOMParser().parseFromString(html || "", "text/html");
  // Preserve some line breaks for readability in PDF
  doc.querySelectorAll("br").forEach((br) => br.replaceWith("\n"));
  doc.querySelectorAll("p, div, li, h1, h2, h3, h4, h5, h6, blockquote, pre").forEach((el) => {
    el.appendChild(doc.createTextNode("\n"));
  });
  return (doc.body.textContent || "").replace(/\n{3,}/g, "\n\n").trim();
}

function mdToPlainText(md) {
  const html = marked.parse(md || "");
  return htmlToPlainText(html);
}

async function fetchImageDataUrl(url) {
  const res = await fetch(url, { mode: "cors" });
  if (!res.ok) throw new Error(`Image fetch failed (${res.status})`);
  const blob = await res.blob();

  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read image blob"));
    reader.readAsDataURL(blob);
  });

  const dims = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth || 1, h: img.naturalHeight || 1 });
    img.onerror = () => reject(new Error("Failed to decode image"));
    img.src = dataUrl;
  });

  const fmt = String(dataUrl).startsWith("data:image/png") ? "PNG" : "JPEG";
  return { dataUrl, fmt, ...dims };
}

function mdToSafeHtml(md) {
  const html = marked.parse(md || "");
  if (typeof DOMPurify !== "undefined") {
    const safe = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    return stripCodeFromHtml(safe);
  }
  return stripCodeFromHtml(html);
}

function createPdfPageElement({ pagePxW, pagePxH, p, imageDataUrl }) {
  const el = document.createElement("div");
  el.style.width = `${pagePxW}px`;
  el.style.height = `${pagePxH}px`;
  el.style.background = "#ffffff";
  el.style.color = "#0f172a";
  el.style.fontFamily = "Helvetica, Arial, sans-serif";
  el.style.boxSizing = "border-box";

  // Exact 3 equal-width columns across full page width
  el.style.display = "grid";
  el.style.gridTemplateColumns = "1fr 1fr 1fr";
  el.style.gap = "0px";

  const mkCol = (title) => {
    const col = document.createElement("div");
    col.style.boxSizing = "border-box";
    col.style.padding = "18px";
    col.style.borderRight = "1px solid #e2e8f0";
    col.style.overflow = "hidden"; // clip vertically to keep 1 manuscript page = 1 PDF page
    col.style.display = "flex";
    col.style.flexDirection = "column";
    col.style.minHeight = "0";

    const h = document.createElement("div");
    h.textContent = title;
    h.style.fontSize = "12px";
    h.style.fontWeight = "700";
    h.style.marginBottom = "10px";
    col.appendChild(h);

    const body = document.createElement("div");
    body.className = "pdf-md";
    body.style.flex = "1";
    body.style.minHeight = "0";
    // Prevent horizontal overflow
    body.style.overflowWrap = "anywhere";
    body.style.wordBreak = "break-word";
    body.style.maxWidth = "100%";
    body.style.fontSize = "10px";
    body.style.lineHeight = "1.35";
    body.style.overflow = "hidden";
    col.appendChild(body);
    return { col, body };
  };

  const c1 = mkCol(`Image — Page ${p.page_number}`);
  const c2 = mkCol("Transcript");
  const c3 = mkCol("Translation");
  c3.col.style.borderRight = "none";

  // Image column content
  const imgWrap = document.createElement("div");
  imgWrap.style.width = "100%";
  imgWrap.style.height = "100%";
  imgWrap.style.display = "flex";
  imgWrap.style.alignItems = "center";
  imgWrap.style.justifyContent = "center";
  imgWrap.style.border = "1px solid #e2e8f0";
  imgWrap.style.borderRadius = "10px";
  imgWrap.style.background = "#f8fafc";
  imgWrap.style.overflow = "hidden";
  imgWrap.style.boxSizing = "border-box";
  imgWrap.style.minHeight = "0";

  if (imageDataUrl) {
    const img = document.createElement("img");
    img.src = imageDataUrl;
    img.alt = `Page ${p.page_number}`;
    img.style.maxWidth = "100%";
    img.style.maxHeight = "100%";
    img.style.objectFit = "contain";
    imgWrap.appendChild(img);
  } else {
    const msg = document.createElement("div");
    msg.textContent = "Image could not be embedded (CORS).";
    msg.style.color = "#475569";
    msg.style.fontSize = "11px";
    imgWrap.appendChild(msg);
  }
  c1.body.appendChild(imgWrap);

  // Markdown bodies
  c2.body.innerHTML = mdToSafeHtml(p.transcript_md || "");
  c3.body.innerHTML = mdToSafeHtml(p.translation_md || "");

  // Make code blocks wrap
  // (No code blocks are rendered; they are stripped into plain text above.)

  el.appendChild(c1.col);
  el.appendChild(c2.col);
  el.appendChild(c3.col);
  return el;
}

async function exportPdf() {
  if (!lastPages || lastPages.length === 0) return;
  if (!streamDone || !allTranslationsReady()) {
    setStatus("Translations are still in progress. Please wait until processing finishes.", "info");
    return;
  }
  const jspdf = window.jspdf;
  if (!jspdf || !jspdf.jsPDF) {
    setStatus("PDF library failed to load.", "error");
    return;
  }

  els.pdfBtn.disabled = true;
  setStatus("Building PDF…", "info");

  const doc = new jspdf.jsPDF({ orientation: "landscape", unit: "pt", format: "a4" });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();

  // Build each page as HTML (with exact 3 columns), render to a canvas of fixed pixel size,
  // then embed that image to fill the PDF page. This guarantees 1/3 columns and no clipping.
  const pagePxW = 1123; // A4 landscape @ ~96dpi
  const pagePxH = 794;

  const scratch = document.createElement("div");
  scratch.style.position = "fixed";
  scratch.style.left = "-100000px";
  scratch.style.top = "0";
  scratch.style.width = `${pagePxW}px`;
  scratch.style.height = `${pagePxH}px`;
  scratch.style.pointerEvents = "none";
  scratch.style.opacity = "0";
  document.body.appendChild(scratch);

  try {
    for (let i = 0; i < lastPages.length; i++) {
      const p = lastPages[i];
      if (i > 0) doc.addPage();

      let imageDataUrl = null;
      try {
        const img = await fetchImageDataUrl(p.image_url);
        imageDataUrl = img.dataUrl;
      } catch {
        imageDataUrl = null;
      }

      scratch.innerHTML = "";
      const pageEl = createPdfPageElement({ pagePxW, pagePxH, p, imageDataUrl });
      scratch.appendChild(pageEl);

      // Render HTML to canvas, then embed as a full-page image in the PDF.
      // eslint-disable-next-line no-await-in-loop
      const canvas = await window.html2canvas(pageEl, {
        backgroundColor: "#ffffff",
        width: pagePxW,
        height: pagePxH,
        windowWidth: pagePxW,
        windowHeight: pagePxH,
        scale: 2, // higher DPI for readability
        useCORS: true,
        allowTaint: false,
        logging: false,
      });
      const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
      doc.addImage(dataUrl, "JPEG", 0, 0, pageW, pageH);
    }
  } finally {
    scratch.remove();
  }

  doc.save("manuscript_translation.pdf");
  setStatus("PDF saved.", "ok");
  updatePdfButtonState();
}

els.modeTranslation.addEventListener("click", () => setMode("translation"));
els.modeTranscript.addEventListener("click", () => setMode("transcript"));
if (els.pdfBtn) els.pdfBtn.addEventListener("click", exportPdf);

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const iiifUrl = (els.url.value || "").trim();
  if (!iiifUrl) return;

  closeStream();
  setStatus("Connecting…", "info");
  els.btn.disabled = true;
  streamDone = false;
  updatePdfButtonState();
  els.results.innerHTML = "";
  lastPages = [];

  try {
    const streamUrl = `/api/stream?iiif_url=${encodeURIComponent(iiifUrl)}`;
    evtSource = new EventSource(streamUrl);

    evtSource.addEventListener("manifest", (ev) => {
      const data = JSON.parse(ev.data);
      lastPages = (data.pages || []).map((p) => ({
        ...p,
        transcript_md: "",
        translation_md: "",
        raw_model_output: "",
      }));
      setStatus(`Loaded ${data.total_pages} page(s). Processing...`, "info");
      renderPages(lastPages); // show all images ASAP
      updatePdfButtonState();
    });

    evtSource.addEventListener("page", (ev) => {
      const data = JSON.parse(ev.data);
      const idx = (data.page_number || 1) - 1;
      if (idx >= 0 && idx < lastPages.length) {
        lastPages[idx] = { ...lastPages[idx], ...data };
      } else {
        lastPages.push(data);
      }
      renderPages(lastPages);
      updatePdfButtonState();
    });

    evtSource.addEventListener("page_error", (ev) => {
      // Keep going; just surface the error.
      const data = JSON.parse(ev.data);
      setStatus(`A page failed: ${data.error || "unknown error"}`, "error");
    });

    evtSource.addEventListener("done", (ev) => {
      const data = JSON.parse(ev.data);
      setStatus(`Done. Completed ${data.completed}/${data.total_pages} page(s).`, "ok");
      closeStream();
      els.btn.disabled = false;
      streamDone = true;
      updatePdfButtonState();
    });

    evtSource.addEventListener("error", (ev) => {
      // EventSource sends generic error events; try to extract server-provided details if present.
      // If the server emitted an "error" SSE event, it will be handled by addEventListener("error") too,
      // but `ev.data` may be empty for network errors.
      if (ev && ev.data) {
        try {
          const data = JSON.parse(ev.data);
          setStatus(data.detail || "Stream error", "error");
        } catch {
          setStatus("Stream error", "error");
        }
      } else {
        setStatus("Stream connection lost.", "error");
      }
      closeStream();
      els.btn.disabled = false;
      streamDone = false;
      updatePdfButtonState();
    });
  } catch (err) {
    setStatus(err.message || String(err), "error");
    closeStream();
    els.btn.disabled = false;
    streamDone = false;
    updatePdfButtonState();
  }
});


