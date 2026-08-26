# Avatar Studio — EC2 Setup Guide

Telugu AI digital personality platform with AI anchors Navya Reddy and Arjun Varma.

---

## EC2 Instance Requirements

| Setting | Value |
|---|---|
| Instance type | `g4dn.2xlarge` (T4 16GB VRAM, 32GB RAM) |
| AMI | Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04) |
| Storage | 100GB gp3 (minimum) |
| Security Group | Port 22 (SSH), Port 80 (HTTP) — **never open 7860** |

---

## Step 1 — Launch EC2 and SSH in

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## Step 2 — Clone the repo

```bash
git clone https://github.com/Umakanth357/Testing.git AvatarStudio
cd AvatarStudio
```

---

## Step 3 — Create .env file

```bash
cat > .env << 'EOF'
HF_TOKEN=hf_your_token_here
EOF
```

Get your token from: https://huggingface.co/settings/tokens (read access is enough)

---

## Step 4 — Run setup (one command, ~20 min)

```bash
chmod +x setup.sh
./setup.sh 2>&1 | tee setup.log
```

This automatically:
- Installs all Python packages in correct order
- Pins numpy==1.26.4 (critical — prevents package conflicts)
- Installs Gradio 4.44.1 + patches gradio_client bool schema bug
- Installs edge-tts (natural Telugu voice, Microsoft)
- Installs Ollama + pulls gemma3:4b (script generation LLM)
- Configures Nginx (port 80 → 127.0.0.1:7860)
- Generates 24 background images (pure PIL, no internet)
- Generates 6 avatar images using SDXL (navya × 3, arjun × 3, ~5 min on T4)
- Creates avatar directory structure the app expects

---

## Step 5 — Start the app

```bash
source venv/bin/activate
python app.py
```

Expected output:
```
Running on local URL: http://127.0.0.1:7860
HTTP Request: HEAD http://127.0.0.1:7860/ "HTTP/1.1 200 OK"
```

Open `http://<EC2_PUBLIC_IP>` in your browser.

---

## How to use Avatar Studio

### Tab 1 — Content

**Input options:**
- **YouTube URL** — paste a YouTube video URL (see note below on EC2 limitations)
- **Topic text** — type what the video should be about (recommended)
- **Paste transcript** — paste text directly

**Recommended workflow:**
1. Type your topic in Telugu or English, e.g.:
   > `Bigg Boss Telugu Season 8 Day 6 voting results and analysis`
2. Select Language: Telugu
3. Select Video Format: Monologue / News Anchor / Short
4. Select Character: Navya or Arjun
5. Set Target Duration (seconds): 180 = 3 min video
6. Click **Analyse & Generate Script**

### Tab 2 — Review Script
- Edit the generated Telugu script before generating video
- Check for factual accuracy — AI generates based on your topic, not live data

### Tab 3 — Avatar
- Choose pose: Half Body (recommended)
- Choose attire: Professional / Traditional / Casual

### Tab 4 — Scene
- Choose background scene

### Tab 5 — Generate
- Click Generate — video is created
- Download when complete

---

## YouTube URL limitations on EC2

AWS EC2 IPs are blacklisted by YouTube for automated access. Two workarounds:

**Option A — Paste topic text directly** (simplest)
Type the key points from the video. The LLM generates a full script from your summary.

**Option B — Whisper transcription**
1. Download audio on your local PC:
   ```
   pip install yt-dlp
   yt-dlp -x --audio-format mp3 -o "video.%(ext)s" "YOUTUBE_URL"
   ```
2. Upload to EC2:
   ```
   scp -i key.pem video.mp3 ubuntu@<EC2_IP>:/tmp/audio.mp3
   ```
3. Transcribe on EC2:
   ```bash
   source venv/bin/activate
   python3 -c "
   import whisper
   m = whisper.load_model('small')
   r = m.transcribe('/tmp/audio.mp3', language='te')
   print(r['text'])
   "
   ```
4. Paste the transcript into the app input box

---

## TTS Voice Quality

The app uses a fallback chain for Telugu TTS:

| Priority | Engine | Quality | Notes |
|---|---|---|---|
| 1st | Indic Parler-TTS | Best | Needs `ai4bharat/indic-parler-tts` model download |
| 2nd | IndicF5 | Very good | Gated HF repo — requires manual approval at huggingface.co/ai4bharat/IndicF5 |
| 3rd | Coqui XTTS | Good | English-quality voice cloning |
| 4th | **edge-tts** | **Natural** | Microsoft Telugu voices (ShrutiNeural F, MohanNeural M) — works out of box |
| 5th | gTTS | Robotic | Google fallback, always works |

On a fresh EC2 setup, edge-tts (4th) will be used — natural Microsoft Telugu voice.

To enable Indic Parler-TTS (best quality):
1. Accept terms at: https://huggingface.co/ai4bharat/indic-parler-tts
2. Model downloads automatically on first use (~4GB)

---

## Known Issues and Fixes Already Applied

| Issue | Fix Applied |
|---|---|
| `gradio_client TypeError: argument of type 'bool' is not iterable` | Auto-patched in setup.sh |
| `starlette 1.x TemplateResponse API break` | Pinned `starlette<1.0.0` |
| `app.py TTS profile 'indic_f5' not found` | Fixed in app.py — uses `AVATARS[persona_id]["voice_profile"]` |
| `Avatar not found: navya_telugu_f/half_body/professional` | Avatar symlinks created in setup.sh |
| `FLUX OOM on T4` | Switched to SDXL (6.5GB VRAM, equivalent portrait quality) |
| `huggingface_hub 1.x missing get_cached_repo_tree` | Shims added |
| `torchvision functional_tensor removed` | Compatibility shim added |

---

## File Structure

```
AvatarStudio/
├── app.py                    # Main Gradio UI
├── config.py                 # Avatars, voices, scenes config
├── setup.sh                  # One-command EC2 setup
├── requirements.txt          # Python dependencies
├── .env                      # HF_TOKEN (never commit)
├── pipeline/
│   ├── tts_engine.py         # TTS: Parler → IndicF5 → Coqui → edge-tts → gTTS
│   ├── avatar_engine.py      # Avatar image loading + SDXL generation
│   ├── script_engine.py      # Ollama script generation
│   ├── video_engine.py       # Video assembly (FFmpeg)
│   └── character_bible.py    # Navya + Arjun personality/voice config
├── scripts/
│   ├── generate_avatars.py   # One-time SDXL avatar generation
│   └── download_backgrounds.py # One-time PIL background generation
├── models/
│   ├── avatars/              # Generated avatar PNGs + persona subdirs
│   └── backgrounds/          # Generated background PNGs
└── outputs/                  # Generated videos
```

---

## Security

- Port 7860 is NEVER opened in Security Group — Nginx proxies only
- `server_name="127.0.0.1"` in app.py — never `0.0.0.0`
- `share=False` in gradio launch
- IndicF5 license must be verified at huggingface.co/ai4bharat/IndicF5 before production use

---

## Stopping and Restarting

**Stop for the day (keep data):**
- AWS Console → Instances → **Stop** (NOT Terminate)
- Volume persists, you pay ~$0.10/GB/month for storage
- Restart: Start instance, SSH in, `cd AvatarStudio && source venv/bin/activate && python app.py`

**Terminate (fresh start):**
- AWS Console → Terminate
- Volume deleted by default
- Next time: follow this guide from Step 1
