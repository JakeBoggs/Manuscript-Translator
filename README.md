# Manuscript Translator

A web application that transcribes and translates historical manuscripts using Gemini 3. Simply provide a IIIF manifest URL, and the app will automatically extract pages, enhance images, transcribe the text, and translate it to English.

## Features

- **IIIF Support** — Works with IIIF Presentation API v2 and v3 manifests from libraries, archives, and museums worldwide
- **Intelligent Cropping** — Automatically detects and crops the main text area, ignoring margins and decorative elements
- **Image Enhancement** — Applies CLAHE-based contrast enhancement and sharpening to improve legibility of faded manuscripts
- **AI Translation** — Translates transcribed text to English with contextual annotations
- **Real-time Streaming** — Results stream to the browser as they complete via Server-Sent Events
- **PDF Export** — Save the complete manuscript with images, transcriptions, and translations side-by-side as a PDF

## Quick Start

### Running with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Manuscript-Translator.git
   cd Manuscript-Translator
   ```

2. Create a `.env` file with your OpenRouter API key:
   ```bash
   echo "OPENROUTER_API_KEY=your_api_key_here" > .env
   ```

3. Start the application:
   ```bash
   docker-compose up --build
   ```

4. Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Find a IIIF manifest URL from a library or archive. Examples:
   - [Bodleian Libraries](https://digital.bodleian.ox.ac.uk/)
   - [British Library](https://www.bl.uk/manuscripts/)
   - [Gallica (BnF)](https://gallica.bnf.fr/)

2. Paste the manifest URL into the input field and click **Process**.

3. Watch as pages are processed in real-time:
   - Images are fetched and enhanced
   - Transcriptions appear as they complete
   - Translations stream in progressively

4. Toggle between **Translation** and **Transcript** views using the buttons in the header.

5. Toggle between **Enhanced** and **Original** images for each page.

6. Click **Save PDF** to export the complete document.
