/* global marked, DOMPurify */

const els = {
  form: document.getElementById("iiif-form"),
  url: document.getElementById("iiif-url"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  tpl: document.getElementById("page-template"),
  btn: document.getElementById("submit-btn"),
  modeTranslation: document.getElementById("mode-translation"),
  modeTranscript: document.getElementById("mode-transcript"),
};

let lastPages = [];
let viewMode = "translation"; // default
let evtSource = null;

marked.setOptions({
  gfm: true,
  breaks: true, // preserve single line breaks (common for transcripts)
  headerIds: false,
  mangle: false,
});

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
    return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  }
  // Fallback (shouldn't happen since we load DOMPurify via CDN):
  return html;
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

els.modeTranslation.addEventListener("click", () => setMode("translation"));
els.modeTranscript.addEventListener("click", () => setMode("transcript"));

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const iiifUrl = (els.url.value || "").trim();
  if (!iiifUrl) return;

  closeStream();
  setStatus("Connecting…", "info");
  els.btn.disabled = true;
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
    });
  } catch (err) {
    setStatus(err.message || String(err), "error");
    closeStream();
    els.btn.disabled = false;
  }
});


