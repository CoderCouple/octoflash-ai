# Octoflash

AI-powered YouTube-to-Manim animation generator. Paste a YouTube URL, get a 3Blue1Brown-quality educational animation with voiceover.

## What It Does

1. **Analyze** — Downloads a YouTube video, extracts frames, auto-fetches transcript, and generates a structured description using Claude's vision API
2. **Generate** — Claude API writes a production-quality Manim script with graphs, equations, diagrams, MCQs, and ValueTracker animations
3. **Render** — Manim CE renders the scene to MP4 with optional ElevenLabs voiceover
4. **Self-correct** — Evaluates rendered output, scores quality, and iteratively improves the script
5. **Brand** — Automatically adds OCTOFLASH watermark intro/outro to every video

## Features

- **Claude-powered script generation** with 3b1b-inspired animation patterns
- **Vision analysis** — sends source video frames to Claude for context-aware animations
- **ElevenLabs voiceover** integration via manim-voiceover
- **Iterative improvement loop** — render → evaluate → regenerate until quality > 7/10
- **Smart fallback chain** — Claude+voice → Claude-no-voice → simple generator
- **40+ auto-fix rules** in `sanitize_script()` for manimgl→CE compatibility
- **Rich helper libraries**: visual effects, diagram patterns, ML visuals, math animations
- **Web UI** — single-page app with video preview, frame gallery, script viewer

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed

### Manim System Dependencies

```bash
# macOS
brew install ffmpeg cairo pango

# Ubuntu/Debian
sudo apt install ffmpeg libcairo2-dev libpango1.0-dev
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and set:
#   ANTHROPIC_API_KEY=your_key_here
#   ELEVEN_API_KEY=your_key_here  (optional, for voiceover)
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## Architecture

```
app/
├── main.py                          # FastAPI endpoints + static UI
├── services/
│   ├── script_generator.py          # Claude API script generation + evaluation
│   ├── transcriber.py               # YouTube transcript fetching
│   ├── describer.py                 # Claude vision frame analysis
│   └── job_manager.py               # Background job tracking
├── manim_pipeline/
│   ├── renderer.py                  # Render pipeline + fallback chain + watermark
│   ├── styles.py                    # OctoflashScene base class, colors, helpers
│   ├── scene_generator.py           # Simple fallback text-cycling generator
│   ├── visual_effects.py            # 30+ transition/animation helpers
│   ├── diagram_patterns.py          # Flowcharts, tables, timelines, Venn diagrams
│   ├── ml_visuals.py                # Neural nets, gradient descent, loss curves
│   └── math_animations.py           # Fourier, Riemann sums, parametric curves
├── models/
│   └── schemas.py                   # Pydantic request/response models
└── static/
    └── index.html                   # Single-page web UI
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Analyze YouTube video (URL + optional transcript) |
| `POST` | `/generate` | Start render job |
| `GET` | `/generate/{job_id}/status` | Poll job status |
| `GET` | `/generate/{job_id}/video` | Download rendered MP4 |
| `GET` | `/scripts/{video_id}` | View generated Manim script |
| `POST` | `/youtube-metadata` | Fetch YouTube video metadata |

## Workflow

```
YouTube URL → Analyze → Generate → Render → Watermark → MP4
                ↓                    ↓
          Frame extraction    Claude script gen
          Transcript fetch    ↓
          Vision analysis     Manim CE render
                              ↓
                         Self-evaluation
                         (iterate if score < 7/10)
```

## License

Private — All rights reserved.
