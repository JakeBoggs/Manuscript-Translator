## Manuscript Translator

Single-page web app: paste a **IIIF manifest URL**, the server extracts the **first sequence/items** canvases, pulls each canvas image, then (in parallel) asks the model to:

- **Transcribe exactly** (markdown) inside `<transcript>...</transcript>`
- **Translate to English** (markdown) inside `<translation>...</translation>`

The UI shows the **image on the left** and **transcript/translation on the right** with a toggle (default: translation).

### Streaming behavior (SSE)

The UI uses **Server-Sent Events**:

- The server sends a `manifest` event immediately with **all page image URLs** (so images render right away).
- Then it sends `page` events as each page finishes.
- Finally it sends `done`.

### Run with Docker

1) Start:

```
docker compose up --build
```

2) Open:

- `http://localhost:8000`

### Environment variables

- **OPENROUTER_API_KEY**: required
