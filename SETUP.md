# Avatar Studio v3.0 — Complete EC2 Setup Guide

**Telugu AI Anchor Platform | Beat HeyGen on Telugu | Fully Human-Authentic Video**

Pipeline: `SadTalker (head motion + eye blink) → MuseTalk (lip sync) → GFPGAN (face quality) → FFmpeg (broadcast compose)`

---

## EC2 Instance

| Setting | Value |
|---|---|
| Instance type | **`g4dn.2xlarge`** |
| vCPU / RAM / VRAM | 8 vCPU · 32GB RAM · T4 16GB VRAM |
| AMI | **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04)** |
| Storage | **150 GB gp3** (SadTalker 700MB + MuseTalk 3.5GB + models) |
| Security Group | Port **22** (SSH) · Port **80** (HTTP) — **NEVER open 7860** |

---

## Step 1 — Launch EC2

AWS Console → EC2 → Launch Instance

- Name: `AvatarStudio`
- AMI: search `Deep Learning OSS Nvidia Driver AMI` → Ubuntu 22.04
- Instance type: `g4dn.2xlarge`
- Storage: change to **150 GB gp3**
- Security Group: allow SSH (22) + HTTP (80) from your IP

---

## Step 2 — SSH into EC2

```bash
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
PEXELS_API_KEY=your_pexels_key_here
EOF
```

- **HF_TOKEN**: Get at https://huggingface.co/settings/tokens (Read access)
- **PEXELS_API_KEY**: Get free at https://www.pexels.com/api/ (optional — for real stock backgrounds)

---

## Step 5 — Run setup (~35 min)

```bash
chmod +x setup.sh
./setup.sh 2>&1 | tee setup.log
```

**What setup.sh installs:**

| Step | What | Size | Time |
|---|---|---|---|
| System packages | ffmpeg, git-lfs, libgl | — | 2 min |
| PyTorch + CUDA | (skip if AMI has it) | — | 2 min |
| Core ML | transformers, diffusers | ~2GB | 5 min |
| GFPGAN stack | basicsr, facexlib, gfpgan | ~500MB | 3 min |
| TTS stack | edge-tts, TTS, gTTS | ~200MB | 2 min |
| Gradio + UI | gradio 4.44.1, patched | ~300MB | 2 min |
| **SadTalker** (NEW) | head motion + blink | **~700MB** | **5 min** |
| **MuseTalk** | lip sync | **~3.5GB** | **10 min** |
| Ollama + llama3.1:8b | script generation | ~5GB | 8 min |
| Nginx config | port 80 proxy | — | 1 min |

Watch log: `tail -f setup.log`

---

## Step 6 — Start the app

```bash
source venv/bin/activate
python app.py
```

Expected output:
```
INFO — Pipeline: SadTalker=True MuseTalk=True GFPGAN=True
Running on local URL: http://127.0.0.1:7860
```

Open in browser: `http://<EC2_PUBLIC_IP>`

---

## How to Use — Step by Step

### Tab 1 · Content

**Source A — Instagram Reel (best for EC2, always works):**
- Paste public Reel URL: `https://www.instagram.com/reel/XXXXXXXXX/`
- Click Extract Content
- Whisper transcribes the audio automatically

**Source B — YouTube with Cookies (works with cookies.txt):**
1. On your local PC — install Chrome extension: *Get cookies.txt LOCALLY*
2. Log in to youtube.com
3. Click extension → Export → saves `cookies.txt`
4. Upload `cookies.txt` in the "YouTube Cookies" field
5. Paste YouTube URL → Extract Content
6. Cookies valid for ~30 days — re-upload when expired

**Source C — Upload Audio (for any YouTube video, always works):**
```bash
# Run on your local PC, not EC2
pip install yt-dlp
yt-dlp -x --audio-format mp3 -o "video.mp3" "YOUTUBE_URL"
```
Upload the MP3 file via "Upload Audio" field.

**Source D — Type topic directly:**
- Type in the Topic box — most copyright-safe option

---

### Tab 2 · Script

1. Select anchor: **Navya Reddy** (warm, professional) or **Arjun Varma** (authoritative)
2. Select duration: 30s → 10min
3. Click **Generate Script** — llama3.1:8b generates in Telugu
4. Review + edit the script
5. Click **Approve Script** → enables Tab 4

---

### Tab 3 · Avatar & Scene

- **Pose**: Half Body (recommended) or Standing
- **Attire**: Professional, Traditional, Casual
- **Scene**: Studio (default), Outdoor, Parliament, Sports, Entertainment

---

### Tab 4 · Generate

Options:
- **Subtitles**: burn Telugu subtitles into video (recommended)
- **9:16 Reels**: auto-exports vertical version for Instagram/YouTube Shorts
- **GFPGAN**: face quality enhancement (adds ~8 min for 10-min video)
- **Indic Parler-TTS**: best voice quality (needs HF gated access first)

Click **Generate Video** → wait 6-40 min depending on video length.

---

## Generation Time Reference (T4 GPU)

| Video Length | Full Pipeline | Without GFPGAN | Static (no lip sync) |
|---|---|---|---|
| 60 seconds | ~6 min | ~4 min | ~1 min |
| 3 min | ~14 min | ~9 min | ~2 min |
| 5 min | ~22 min | ~14 min | ~3 min |
| 10 min | ~40 min | ~25 min | ~5 min |

**Why these times?**
- SadTalker: ~0.8× real-time (head motion from audio)
- MuseTalk: ~1× real-time (lip sync)
- GFPGAN: ~0.8× real-time (per-frame face quality)
- FFmpeg compose: ~0.2× real-time

---

## Output Files

Each generation creates a job folder `outputs/<job_id>/`:

```
outputs/abc123de/
├── avatar_studio_abc123de.mp4          ← 16:9 (YouTube/LinkedIn)
├── avatar_studio_abc123de_reels.mp4    ← 9:16 (Instagram Reels / YouTube Shorts)
├── avatar_studio_abc123de_square.mp4   ← 1:1 (Instagram feed) — if enabled
├── thumbnail.jpg                        ← YouTube-style thumbnail
├── subtitles.ass                        ← Telugu subtitles (if enabled)
├── script_abc123de.txt                  ← approved script
└── avatar.png                           ← avatar image used
```

---

## Voice Quality Ladder

| Tier | Engine | When Active | Quality |
|---|---|---|---|
| **Best** | **Indic Parler-TTS** | Enabled + HF gated access approved | ⭐⭐⭐⭐⭐ Natural emotional delivery |
| **Default** | **edge-tts ShrutiNeural** | Always available | ⭐⭐⭐⭐ Natural Telugu prosody |
| Fallback | gTTS | edge-tts fails | ⭐⭐ Works but robotic |

**To unlock Indic Parler-TTS:**
1. Go to: https://huggingface.co/ai4bharat/indic-parler-tts
2. Request gated access (approved within 24-48h)
3. Then: `pip install git+https://github.com/huggingface/parler-tts`
4. Enable "Indic Parler-TTS" checkbox in Tab 4

---

## vs The Competition

| Tool | Their strength | Our advantage |
|---|---|---|
| HeyGen | Proprietary neural rendering | We support Telugu (they don't) |
| Synthesia | 60+ languages, corporate polish | We have genuine Telugu voice + proper script |
| D-ID | Quick turnaround | Better face quality + SadTalker motion |
| ElevenLabs | English voice quality | We **beat them on Telugu** (ShrutiNeural is more natural) |
| Fliki | Indian market | We have far better lip sync + authentic Telugu |

**Blue ocean: Zero commercial tools have genuine Telugu support with proper lip sync.**

---

## Optional Enhancements

### Pexels Real Backgrounds (highly recommended)
Real cinematic stock video backgrounds instead of PIL-generated ones.

1. Sign up free: https://www.pexels.com/api/
2. Get API key (free, 200 req/hr)
3. Add to `.env`: `PEXELS_API_KEY=your_key`
4. Backgrounds auto-download and cache on first use

### Indic Parler-TTS Emotion Control
Best Telugu TTS with 6 emotion parameters.

1. Request HF gated access: https://huggingface.co/ai4bharat/indic-parler-tts
2. `pip install git+https://github.com/huggingface/parler-tts`
3. Enable in UI Tab 4

---

## Keep App Running After SSH Disconnect

```bash
nohup python app.py > logs/app.log 2>&1 &
echo $! > app.pid
```

Check logs:
```bash
tail -f logs/app.log
```

Stop:
```bash
kill $(cat app.pid)
```

---

## Stop / Restart Instance

**Stop for the night (data preserved, ~$0.10/GB/month storage):**
- AWS Console → EC2 → Stop instance

**Resume next day:**
- AWS Console → Start instance
- SSH → `cd AvatarStudio && source venv/bin/activate && python app.py`

**Terminate (deletes everything):**
- AWS Console → Terminate → run this setup guide again from Step 1

---

## Security Checklist

- ✅ Port 7860 NEVER opened in Security Group
- ✅ `server_name="127.0.0.1"` in app.py (never 0.0.0.0)
- ✅ `share=False` in Gradio launch
- ✅ `.env` in `.gitignore` (HF_TOKEN never committed)
- ✅ `cookies.txt` stored locally only (never committed)
- ✅ Nginx proxies port 80 → 127.0.0.1:7860 only

---

## Known Issues — Fixed

| Issue | Fix |
|---|---|
| `gradio_client TypeError: bool` | Auto-patched in setup.sh |
| `starlette 1.x TemplateResponse` | Pinned `starlette<1.0.0` |
| `TTS profile 'indic_f5' not found` | Uses `AVATARS[persona_id]["voice_profile"]` |
| Avatar not found: `navya/half_body/professional` | Directory structure created in setup.sh |
| YouTube blocked on EC2 IPs | Cookies + audio upload workarounds |
| numpy version conflicts | Pin 1.26.4 FIRST before any ML package |

---

## File Structure v3.0

```
AvatarStudio/
├── app.py                          # Main UI — 4-tab Gradio (v3.0)
├── config.py                       # AVATARS, SCENES, model IDs
├── setup.sh                        # One-command EC2 setup (v3.0)
├── requirements.txt                # Python dependencies
├── .env                            # HF_TOKEN + PEXELS_API_KEY (never commit)
├── pipeline/
│   ├── content_engine.py           # Instagram / YouTube / audio / text
│   ├── script_engine.py            # llama3.1:8b + emotion tagging (v3.0)
│   ├── tts_engine.py               # edge-tts + Parler-TTS + emotion (v3.0)
│   ├── audio_post.py               # Broadcast audio chain — NEW v3.0
│   ├── sadtalker_engine.py         # SadTalker head motion + blink — NEW v3.0
│   ├── musetalk_engine.py          # MuseTalk lip sync (v3.0 — uses SadTalker)
│   ├── video_engine.py             # FFmpeg compose + Pexels + color grade (v3.0)
│   ├── avatar_engine.py            # Avatar image loading
│   └── character_bible.py         # Navya + Arjun personality config
├── scripts/
│   ├── generate_avatars.py         # SDXL avatar generation
│   └── download_backgrounds.py    # Fallback PIL background generation
├── assets/fonts/                   # Noto Sans Telugu TTF
├── models/
│   ├── avatars/                    # Avatar PNGs by persona/pose/attire
│   ├── backgrounds/                # Background images (PIL + Pexels cache)
│   ├── SadTalker/                  # SadTalker (cloned by setup.sh)
│   ├── MuseTalk/                   # MuseTalk (cloned by setup.sh)
│   └── yt_cookies.txt              # YouTube cookies (uploaded via UI, never committed)
└── outputs/
    └── <job_id>/                   # Per-job outputs
```
