# 🎭 Avatar Video Generator
### Internal Corporate Video Tool — South Indian Avatars + Telugu/English + Emotion-Aware

---

## What this tool does

Paste a script, drop a YouTube URL, or paste any article → Get a realistic talking-head video with:
- South Indian photorealistic avatar (AI-generated, unique to your org)
- Telugu or English voice with per-sentence emotion detection
- Frame-perfect lip sync (MuseTalk)
- Natural head movement and expressions (LivePortrait)
- 4K face enhancement (GFPGAN)
- Auto-generated captions (Whisper)
- Office/studio background

Content types: KT sessions, tool reviews, news digests, movie reviews, event summaries, notifications

---

## What it costs

| Component | Cost |
|---|---|
| All AI models | **$0** — 100% open source |
| EC2 g4dn.2xlarge (spot) | ~$0.22/hr → ~$50/month (8hrs/day) |
| EC2 g4dn.xlarge (minimum) | ~$0.16/hr → ~$35/month (8hrs/day) |
| Storage (150GB EBS gp3) | ~$12/month |
| **Total** | **~$62/month** |

---

## Server requirements

**Minimum:** AWS EC2 g4dn.xlarge
- GPU: NVIDIA T4 (16 GB VRAM)
- RAM: 16 GB
- vCPU: 4
- Storage: 150 GB EBS gp3

**Recommended:** AWS EC2 g4dn.2xlarge
- GPU: NVIDIA T4 (16 GB VRAM)
- RAM: 32 GB
- vCPU: 8
- Storage: 150 GB EBS gp3

**AMI:** AWS Deep Learning AMI GPU PyTorch (Ubuntu 22.04)
→ Pre-installed: CUDA 11.8, PyTorch, Python 3.10

---

## Deploy in 3 commands

### Step 1: Launch EC2
```
AWS Console → EC2 → Launch Instance
- AMI: "Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04)"
- Instance: g4dn.2xlarge (or g4dn.xlarge for minimum)
- Storage: 150 GB gp3
- Security group: Allow inbound TCP 80, 22
- Use spot instance → saves 60-70%
```

### Step 2: SSH in and clone
```bash
ssh -i your-key.pem ubuntu@YOUR-EC2-IP
git clone https://github.com/YOUR-ORG/avatar-tool.git
cd avatar-tool
```

### Step 3: Run setup (one time, ~30 minutes)
```bash
bash setup.sh
```

That's it. After setup completes, open: `http://YOUR-EC2-IP`

---

## Manual start/stop (if not using supervisor)

```bash
# Start
bash start.sh

# Stop
bash stop.sh

# Check logs
tail -f logs/app.out.log
```

---

## Run tests before first use

```bash
source venv/bin/activate
python test_all.py
```

All critical tests must pass before generating videos.

---

## Add your own avatars (optional but recommended)

After setup, generate custom South Indian avatars:
```bash
source venv/bin/activate
python scripts/generate_avatars.py
```

Or place your own PNG images (512×768px, front-facing, neutral background) in the `avatars/` folder.

---

## Environment variables (optional)

Create `.env` in root:
```
# Groq free API — fallback if Ollama unavailable
GROQ_API_KEY=your_groq_key_here

# HuggingFace token — if using private models
HF_TOKEN=your_hf_token_here
```

---

## Architecture

```
User inputs script / YouTube URL / article text
        ↓
[Emotion Brain]  Ollama Gemma3 4B → per-sentence emotion tags
        ↓
[Voice Layer]    Svara TTS (Telugu) / Chatterbox (English) → WAV
        ↓
[Animation]      LivePortrait → head movement + expressions
        ↓
[Lip Sync]       MuseTalk (diffusion) → frame-perfect mouth sync
        ↓
[Enhancement]    GFPGAN → face restore → 4x upscale
        ↓
[Composition]    ffmpeg → avatar + background + captions + logo
        ↓
Output: MP4 video (720p/1080p) in outputs/ folder
```

---

## Fallback chain (resilience)

Every component has fallbacks — tool never hard-fails:

| Component | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| Emotion brain | Ollama Gemma3 | Groq API (free) | Rule-based keywords |
| Telugu TTS | Svara TTS | AI4Bharat Indic | gTTS |
| English TTS | Chatterbox | Coqui TTS | gTTS |
| Lip sync | MuseTalk | LatentSync | Static avatar |
| Expression | LivePortrait | SadTalker | Static loop |
| Face enhance | GFPGAN | Basic upscale | None (skip) |

---

## Copyright and usage policy

**INTERNAL USE ONLY**

When using YouTube or external content as source:
- Content is **fully rewritten** by LLM — no verbatim copying
- Source is attributed in script metadata
- Videos are for internal corporate distribution only
- Always attribute original creator in video description

---

## File structure

```
avatar_tool/
├── app.py                    # Main Gradio application
├── setup.sh                  # One-time EC2 setup
├── start.sh / stop.sh        # Run/stop
├── test_all.py               # Component test suite
├── requirements.txt
├── avatars/                  # Avatar PNG images
├── outputs/                  # Generated videos
├── logs/                     # Application logs
├── models/
│   ├── MuseTalk/             # Lip sync model
│   ├── LivePortrait/         # Expression model
│   ├── SadTalker/            # Fallback animation
│   └── weights/              # Downloaded model weights
├── web/
│   └── assets/               # Backgrounds, logo
└── scripts/
    ├── emotion_tagger.py     # Ollama emotion detection
    ├── tts_engine.py         # Voice generation
    ├── video_pipeline.py     # Video generation
    ├── content_ingester.py   # YouTube + text ingestion
    ├── generate_avatars.py   # Avatar generation
    └── download_weights.py   # Model weight downloader
```

---

## Troubleshooting

**"No GPU" warning:**
→ g4dn instances have NVIDIA T4. If you see "no GPU", check CUDA drivers: `nvidia-smi`

**Ollama not reachable:**
→ Run: `ollama serve &` then `ollama pull gemma3:4b`

**TTS generates silence:**
→ Run `python test_all.py` — check TTS smoke test results

**Video too slow to generate:**
→ Use g4dn.2xlarge instead of xlarge. Ensure spot instance is running.

**Telugu voice sounds robotic:**
→ Ensure Svara TTS is installed: `pip install git+https://github.com/Kenpath/svara-tts-inference.git`
→ Check GROQ_API_KEY is set for fallback

---

## License

All model licenses:
- MuseTalk: MIT
- LivePortrait: MIT
- SadTalker: MIT
- Svara TTS: Apache 2.0
- Chatterbox: MIT
- Coqui TTS: Mozilla Public License 2.0
- GFPGAN: Apache 2.0
- Whisper: MIT
- Ollama + Gemma3: Apache 2.0
- Gradio: Apache 2.0

This tool itself: Internal use license — not for external distribution.
