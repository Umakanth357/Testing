# Avatar Studio v2.0 — Full EC2 Setup Guide

Telugu AI Video Generator | Instagram Reels + YouTube + Audio + Text → Professional Video

---

## EC2 Instance — What to Launch

| Setting | Value |
|---|---|
| Instance type | `g4dn.2xlarge` |
| vCPU / RAM / VRAM | 8 vCPU · 32GB RAM · T4 16GB VRAM |
| AMI | **Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)** |
| Storage | **120 GB gp3** (minimum — MuseTalk weights need space) |
| Security Group | Port **22** (SSH) · Port **80** (HTTP) — **never open 7860** |

---

## Step 1 — Launch EC2

AWS Console → EC2 → Launch Instance

- Name: `AvatarStudio`
- AMI: search **"Deep Learning OSS Nvidia Driver AMI"** → pick Ubuntu 22.04
- Instance type: `g4dn.2xlarge`
- Key pair: use existing or create new `.pem`
- Storage: change to **120 GB gp3**
- Security Group: allow SSH (22) + HTTP (80) from your IP (or 0.0.0.0/0 for HTTP)

Click **Launch Instance**.

---

## Step 2 — SSH into EC2

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

If permission denied:
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Step 3 — Clone the repo

```bash
git clone https://github.com/Umakanth357/Testing.git AvatarStudio
cd AvatarStudio
```

---

## Step 4 — Create .env file

```bash
cat > .env << 'EOF'
HF_TOKEN=hf_YOUR_TOKEN_HERE
EOF
```

Get your HuggingFace token: https://huggingface.co/settings/tokens
(Read access is enough — needed for model downloads)

---

## Step 5 — Run setup (one command, ~25 min)

```bash
chmod +x setup.sh
./setup.sh 2>&1 | tee setup.log
```

**What this does automatically:**
- Installs all Python packages in correct dependency order
- Pins numpy==1.26.4 (critical — prevents conflicts)
- Installs Gradio 4.44.1 + patches bool schema bug
- Installs edge-tts (Microsoft natural Telugu voices)
- Installs Ollama → pulls **llama3.1:8b** (primary) + gemma3:4b (fallback)
- Installs Nginx → configures port 80 → 127.0.0.1:7860 proxy
- Generates 24 background images (PIL, no internet)
- Generates 6 avatar images via SDXL (~5 min on T4)
- Clones MuseTalk + downloads ~3.5GB lip sync weights
- Creates all avatar directory structure

**Watch the log while it runs:**
```bash
tail -f setup.log
```

---

## Step 6 — Start the app

```bash
source venv/bin/activate
python app.py
```

Expected output:
```
INFO — MuseTalk: available  ← means lip sync is working
INFO — LLM: llama3.1:8b
Running on local URL: http://127.0.0.1:7860
```

Open in browser: `http://<EC2_PUBLIC_IP>`

---

## How to Use Avatar Studio v2.0

### Tab 1 · Content — Get your source material

**Option A — Instagram Reel (recommended, works perfectly on EC2)**
- Paste a public Instagram Reel URL
- e.g. `https://www.instagram.com/reel/XXXXXXXXXXX/`
- Click **Extract Content** → Whisper transcribes the audio automatically

**Option B — Upload Audio (for YouTube videos)**
- Download the audio on your local PC:
  ```bash
  pip install yt-dlp
  yt-dlp -x --audio-format mp3 -o "video.mp3" "YOUTUBE_URL"
  ```
- Upload via the **Upload Audio** field in the UI

**Option C — Type a topic directly**
- Type in the Topic box: e.g. `Bigg Boss Telugu Season 8 Day 10 — eviction results and house drama`
- Best option for original content, no copyright concerns

### Tab 2 · Script

1. Select character: **Navya Reddy** (Telugu F) or **Arjun Varma** (Telugu M)
2. Click **Generate Script** → llama3.1:8b generates in Telugu anchor style
3. Read and edit the script if needed
4. Click **Approve Script**

### Tab 3 · Avatar & Scene

- Choose avatar pose (Half Body recommended)
- Choose attire (Professional default)
- Choose background scene (auto-selected based on content)
- For debate: select second speaker

### Tab 4 · Generate

- Check output options (subtitles, 9:16 Reels export)
- Click **Generate Video**
- Progress shown live
- Download 16:9 (YouTube) and 9:16 (Reels/Shorts)

---

## Generation Times (T4 GPU)

| Video Duration | With MuseTalk (lip sync) | Static fallback |
|---|---|---|
| 60 second short | ~3 min | ~1 min |
| 3 min KT video | ~6 min | ~2 min |
| 8 min review | ~12 min | ~4 min |
| 15 min debate | ~25 min | ~8 min |

---

## Output Formats

Each generated video produces:
- `avatar_studio_JOBID.mp4` — 16:9 (1920×1080) for YouTube/LinkedIn
- `avatar_studio_JOBID_reels.mp4` — 9:16 (1080×1920) for Instagram Reels/YouTube Shorts
- `thumbnail.jpg` — YouTube-style thumbnail
- `script_JOBID.txt` — script for reference

---

## TTS Voice Chain

| Priority | Engine | Quality | Notes |
|---|---|---|---|
| **1st** | **edge-tts** | **Natural** | Microsoft ShrutiNeural (F) / MohanNeural (M). Works immediately. |
| 2nd | Indic Parler-TTS | Best | Needs HF gated access at huggingface.co/ai4bharat/indic-parler-tts (~4GB) |
| Last | gTTS | Robotic | Google fallback, always works |

---

## Known Issues — Already Fixed

| Issue | Fix Applied |
|---|---|
| `gradio_client TypeError: argument of type 'bool'` | Auto-patched in setup.sh |
| `starlette 1.x TemplateResponse API break` | Pinned starlette<1.0.0 |
| `TTS profile 'indic_f5' not found` | Fixed — uses AVATARS voice_profile |
| `Avatar not found: navya_telugu_f/half_body/professional` | Directory structure created in setup.sh |
| YouTube blocked on EC2 IPs | Audio upload + Whisper workaround in UI |

---

## Stop / Restart

**Stop for the day (keep data — costs ~$0.10/GB/month storage):**
- AWS Console → Instances → **Stop**
- Restart next time: Start → SSH → `cd AvatarStudio && source venv/bin/activate && python app.py`

**Keep app running after SSH disconnect:**
```bash
nohup python app.py > logs/app.log 2>&1 &
echo $! > app.pid
```

Stop it:
```bash
kill $(cat app.pid)
```

**Terminate (fresh start, deletes volume):**
- AWS Console → Terminate → follow this guide again

---

## Security

- Port 7860 is **NEVER** opened in Security Group — Nginx proxies port 80 only
- `server_name="127.0.0.1"` in app.py — never 0.0.0.0
- `share=False` in Gradio launch
- .env file with HF_TOKEN is in .gitignore — never committed

---

## File Structure

```
AvatarStudio/
├── app.py                         # Main UI (Gradio 4.44.1)
├── config.py                      # Central config (model IDs, scenes, avatars)
├── setup.sh                       # One-command EC2 setup
├── requirements.txt               # Python dependencies
├── .env                           # HF_TOKEN (never commit)
├── pipeline/
│   ├── content_engine.py          # Instagram/YouTube/audio/text extraction
│   ├── tts_engine.py              # edge-tts + Parler-TTS + gTTS chain
│   ├── script_engine.py           # llama3.1:8b script generation
│   ├── musetalk_engine.py         # MuseTalk lip sync wrapper + GFPGAN
│   ├── video_engine.py            # FFmpeg compositing, subtitles, formats
│   ├── avatar_engine.py           # SDXL avatar loading
│   └── character_bible.py        # Navya + Arjun personality config
├── scripts/
│   ├── generate_avatars.py        # SDXL avatar generation
│   └── download_backgrounds.py   # Background image generation
├── models/
│   ├── avatars/                   # Avatar PNGs by persona
│   ├── backgrounds/               # Background PNGs
│   └── MuseTalk/                  # MuseTalk lip sync (cloned by setup.sh)
└── outputs/
    └── <job_id>/                  # Per-job: video + reels + thumbnail + script
```
