# Avatar Video Generator — Deployment Guide

**Target:** AWS EC2 g4dn.2xlarge (or g4dn.xlarge minimum)  
**OS:** Ubuntu 22.04 (Deep Learning AMI)  
**Time to deploy:** ~45 minutes on first setup  
**Monthly cost:** ~$50/month spot pricing at 8 hrs/day

---

## Table of Contents

1. [EC2 Launch Steps](#1-ec2-launch-steps)
2. [Connect and Clone](#2-connect-and-clone)
3. [Run setup.sh](#3-run-setupsh)
4. [Run test_all.py and Expected Output](#4-run-test_allpy-and-expected-output)
5. [Access the Gradio UI](#5-access-the-gradio-ui)
6. [Stop and Restart the Service](#6-stop-and-restart-the-service)
7. [Troubleshooting: Top 5 Failure Points](#7-troubleshooting-top-5-failure-points)

---

## 1. EC2 Launch Steps

### Step 1.1 — Open EC2 Console

Go to: **AWS Console → EC2 → Launch Instance**

```
Region: Choose closest to your users (e.g. ap-south-1 for India)
```

*[Screenshot: AWS EC2 dashboard showing "Launch Instance" orange button in top right]*

---

### Step 1.2 — Name the Instance

```
Name: avatar-tool-prod
```

---

### Step 1.3 — Choose AMI

Click **Browse more AMIs** → search for:

```
Deep Learning OSS Nvidia Driver AMI GPU PyTorch
```

Select the result that shows:
- **Name contains:** `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2` (pick latest version)
- **OS:** Ubuntu 22.04
- **Architecture:** x86_64

*[Screenshot: AMI search results showing "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 22.04)" as top result with "Select" button]*

> **Why this AMI?** It comes with CUDA 11.8, cuDNN, and PyTorch pre-installed. Saves 20 minutes of driver setup and avoids CUDA version mismatch issues.

---

### Step 1.4 — Choose Instance Type

| Use Case | Instance | vCPU | RAM | GPU VRAM | Spot Price |
|---|---|---|---|---|---|
| **Minimum** | `g4dn.xlarge` | 4 | 16 GB | T4 16 GB | ~$0.16/hr |
| **Recommended** | `g4dn.2xlarge` | 8 | 32 GB | T4 16 GB | ~$0.22/hr |

Select: `g4dn.2xlarge`

*[Screenshot: Instance type selection table with g4dn.2xlarge row highlighted in blue]*

> **Why g4dn.2xlarge over xlarge?** The T4 GPU is the same. The extra RAM (32 vs 16 GB) prevents OOM when Whisper large-v3 + MuseTalk + Chatterbox are all loaded simultaneously for a YouTube job.

---

### Step 1.5 — Key Pair

Select an existing key pair or create a new one:
- Click **Create new key pair**
- Name: `avatar-tool-key`
- Type: RSA
- Format: `.pem`
- Click **Create key pair** (downloads automatically)

*[Screenshot: "Create key pair" dialog with RSA selected and .pem format]*

```bash
# Save your key and restrict permissions
chmod 400 ~/Downloads/avatar-tool-key.pem
```

---

### Step 1.6 — Network Settings / Security Group

Click **Edit** on Network Settings. Create a new security group named `avatar-tool-sg` with these rules:

| Type | Protocol | Port | Source | Reason |
|---|---|---|---|---|
| SSH | TCP | 22 | My IP | SSH access only |
| HTTP | TCP | 80 | 0.0.0.0/0 | Gradio UI via Nginx |

**Do NOT open port 7860.** Gradio runs on 7860 internally but Nginx proxies it on port 80. Exposing 7860 directly bypasses Nginx and has no rate limiting.

*[Screenshot: Security group rules table showing port 22 (My IP) and port 80 (0.0.0.0/0) only]*

---

### Step 1.7 — Storage

Under **Configure Storage**:

```
Volume 1 (root): 200 GB  gp3  3000 IOPS
```

> **Why 200 GB?**  
> - MuseTalk weights: ~8 GB  
> - LivePortrait weights: ~6 GB  
> - SadTalker weights: ~5 GB  
> - Whisper large-v3: ~3 GB  
> - SD/FLUX model for avatar generation: ~14 GB  
> - Python venv + packages: ~10 GB  
> - Output videos (buffer): ~30 GB  
> - OS + swap: ~20 GB  
> - **Total: ~96 GB + 100 GB headroom**

*[Screenshot: Storage configuration showing "200 GiB gp3" with 3000 IOPS]*

---

### Step 1.8 — Spot Instance (saves 60-70%)

Expand **Advanced Details** → scroll to **Purchasing option**:

```
☑  Request Spot Instances
Maximum price: leave blank (use current spot price)
Interruption behavior: Stop
```

*[Screenshot: Advanced Details section with "Request Spot Instances" checkbox ticked]*

> Spot instances can be reclaimed by AWS with 2-minute notice. Supervisor auto-restarts the app after stop/start. For production use, consider On-Demand or a Spot interruption handler.

---

### Step 1.9 — Launch

Click **Launch Instance**.

*[Screenshot: Green "Launch Instance" confirmation screen showing instance ID i-0abc123...]*

Wait 2-3 minutes for the instance to show **Running** state in the EC2 console, then note your **Public IPv4 address**.

---

## 2. Connect and Clone

### Step 2.1 — SSH into EC2

```bash
ssh -i ~/Downloads/avatar-tool-key.pem ubuntu@YOUR-EC2-PUBLIC-IP
```

*[Screenshot: Terminal showing ubuntu@ip-xxx-xxx connected prompt with GPU driver info visible]*

### Step 2.2 — Verify GPU

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx    Driver Version: 525.xx    CUDA Version: 12.x          |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla T4          Off   |  00000000:00:1E.0 Off |                  0   |
+-----------------------------------------------------------------------------+
```

If you see `command not found`, the Deep Learning AMI did not load correctly — re-launch with the correct AMI.

### Step 2.3 — Clone the Repository

```bash
git clone https://github.com/Umakanth357/Testing.git avatar_tool
cd avatar_tool
```

### Step 2.4 — (Optional) Set API Keys

```bash
cp .env.example .env 2>/dev/null || touch .env
nano .env
```

Add your keys (both optional — tool works without them):
```bash
# Groq free API — fallback if Ollama unavailable
GROQ_API_KEY=your_groq_key_here

# HuggingFace token — only needed for private/gated models
HF_TOKEN=your_hf_token_here
```

---

## 3. Run setup.sh

### Step 3.1 — Make Executable and Run

```bash
chmod +x setup.sh
bash setup.sh 2>&1 | tee logs/setup.log
```

> Piping to `tee` saves the full setup log for debugging. The `logs/` folder is created automatically.

### Step 3.2 — What setup.sh Does (9 Steps)

| Step | What Happens | Duration |
|---|---|---|
| 1/9 System packages | apt-get: ffmpeg, nginx, supervisor, libsndfile | ~2 min |
| 2/9 Python venv | Creates `venv/` with Python 3.10 | ~1 min |
| 3/9 Python deps | `pip install -r requirements.txt` + PyTorch CUDA | ~8 min |
| 4/9 Clone model repos | MuseTalk, LivePortrait, SadTalker from GitHub | ~5 min |
| 5/9 Download weights | GFPGANv1.4.pth, MuseTalk, LivePortrait weights | ~15 min |
| 6/9 Ollama + Gemma3 | Installs Ollama, pulls gemma3:4b model | ~5 min |
| 7/9 Generate avatars | FLUX/SDXL → 6 South Indian avatar PNGs | ~5 min |
| 8/9 Nginx config | Reverse proxy on port 80 → 7860 | <1 min |
| 9/9 Supervisor | Auto-start and auto-restart config | <1 min |

### Step 3.3 — Success Output

Setup ends with:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SETUP COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Open browser: http://3.110.xx.xx
  Logs: tail -f /home/ubuntu/avatar_tool/logs/app.out.log
  Start: bash start.sh
  Stop:  bash stop.sh
```

---

## 4. Run test_all.py and Expected Output

### Step 4.1 — Run the Test Suite

```bash
cd ~/avatar_tool
source venv/bin/activate
python test_all.py
```

### Step 4.2 — Full Expected Output (Passing)

```
============================================================
  AVATAR TOOL — FULL COMPONENT TEST SUITE
============================================================

🔧 Environment & Infrastructure
  Testing Python 3.10+...          ✅ Python 3.10.12 (0.0s)
  Testing ffmpeg...                ✅ ffmpeg version 4.4.2-0ubuntu0.22.04.1 (0.1s)
  Testing GPU / CUDA...            ✅ Tesla T4 — 15109 MB VRAM (0.2s)
  Testing Disk space...            ✅ 152 GB free / 196 GB total (0.0s)

🧠 AI Models
  Testing Ollama (local LLM)...    ✅ Models available: ['gemma3:4b'] (0.3s)
  Testing Emotion tagger...        ✅ Tagged 2 sentences: ['excited', 'urgent'] (2.1s)
  Testing Whisper (STT)...         ✅ whisper available (0.1s)
  Testing yt-dlp (YouTube)...      ✅ 2024.11.18 (0.1s)

🎙️ TTS Engines
  Testing Svara TTS (Telugu)...    ✅ svara not installed but HuggingFace API reachable (1.2s)
  Testing Chatterbox (EN)...       ✅ chatterbox importable (0.2s)
  Testing Coqui (EN fallback)...   ✅ Coqui TTS available as EN fallback (0.3s)
  Testing TTS smoke test...        ✅ Audio generated: 284,160 bytes (8.4s)

🎬 Video Models
  Testing MuseTalk (lip sync)...   ✅ Repo and weights present (0.0s)
  Testing LivePortrait (anim)...   ✅ 5 weight files found (0.0s)
  Testing SadTalker (fallback)...  ✅ SadTalker repo present (0.0s)
  Testing GFPGAN (enhance)...      ✅ Weights: 332MB | pkg OK (0.1s)

🎭 Assets
  Testing Avatar images...         ✅ 6 avatar(s) found: ['si_male_30.png', ...] (0.0s)

🔗 Integration
  Testing Content ingester...      ✅ Script generated: 312 words (4.8s)

============================================================
  RESULTS: 18/18 passed
  ✅ All tests passed — tool is ready!
============================================================

Detailed results saved: logs/test_results.json
```

### Step 4.3 — Acceptable vs Critical Failures

**These tests MUST pass before using the tool:**

| Test | Why Critical |
|---|---|
| `Python 3.10+` | Walrus operator and match syntax used |
| `ffmpeg` | Every audio concat and video compose depends on it |
| `GPU / CUDA` | MuseTalk and Chatterbox won't run on CPU in reasonable time |
| `TTS smoke test` | End-to-end audio chain proof |
| `Avatar images` | Tool can't generate video without an avatar face |
| `Emotion tagger` | Powers the per-sentence voice modulation |

**These tests can fail — fallbacks handle them:**

| Test | Fallback |
|---|---|
| `Ollama (local LLM)` | Groq API → rule-based keywords |
| `Svara TTS (Telugu)` | AI4Bharat HF API → gTTS |
| `Chatterbox (EN)` | Coqui TTS → gTTS |
| `SadTalker (fallback)` | Static avatar loop |

### Step 4.4 — If Tests Fail

```bash
# See full details
cat logs/test_results.json | python3 -m json.tool

# Re-run just the critical tests
python test_all.py 2>&1 | grep -E "(✅|❌|RESULTS)"
```

---

## 5. Access the Gradio UI

### Step 5.1 — Get Your EC2 Public IP

```bash
# From inside EC2
curl -s ifconfig.me
```

Or find it in AWS Console: **EC2 → Instances → your instance → Public IPv4 address**

### Step 5.2 — Open in Browser

```
http://YOUR-EC2-PUBLIC-IP
```

*[Screenshot: Gradio UI loading with "🎭 Avatar Video Generator" header, two-panel layout with Content Input on left and Voice & Avatar settings on right]*

> **Important:** Use `http://` not `https://`. HTTPS is not configured by default. If your organization requires HTTPS, add an ACM certificate via AWS Load Balancer or configure Nginx with Let's Encrypt.

### Step 5.3 — First Video Test

1. Select source type: **script**
2. Paste a short test script (3-4 sentences)
3. Language: **en**, Voice: **en_female**
4. Avatar: select any from dropdown
5. Background: **Office**
6. Target duration: **1 minute**
7. Click **🚀 Generate Avatar Video**

Expected time for a 1-minute video on g4dn.2xlarge:
- Emotion tagging: ~2s
- TTS audio: ~5-10s
- LivePortrait animation: ~30-60s
- MuseTalk lip sync: ~60-90s
- GFPGAN enhancement: ~30-60s
- Compose + captions: ~20s
- **Total: 3-5 minutes**

### Step 5.4 — Health Check

In the Gradio UI, click the **🔧 System Status** tab → click **Run Health Check**.

All components should show ✅. Any ❌ indicates a setup issue.

---

## 6. Stop and Restart the Service

### Via Supervisor (Recommended)

```bash
# Status
sudo supervisorctl status avatar_tool

# Stop
bash stop.sh
# or: sudo supervisorctl stop avatar_tool

# Start
bash start.sh
# or: sudo supervisorctl start avatar_tool

# Restart (e.g. after code changes)
sudo supervisorctl restart avatar_tool

# View live logs
tail -f ~/avatar_tool/logs/app.out.log
tail -f ~/avatar_tool/logs/app.err.log
```

### After Pulling Code Updates

```bash
cd ~/avatar_tool
git pull origin main
sudo supervisorctl restart avatar_tool
# Wait 15-20 seconds for models to reload
tail -f logs/app.out.log
```

### EC2 Instance Start/Stop (Cost Saving)

When you stop the EC2 instance, supervisor and all processes stop. When you restart:

```bash
# After EC2 restart — Ollama may need to be kicked
sudo systemctl start ollama
ollama list  # verify gemma3:4b is still there

# Check if supervisor auto-started app
sudo supervisorctl status avatar_tool
# If STOPPED: sudo supervisorctl start avatar_tool
```

### Managing Output Videos (Disk Space)

```bash
# Check disk usage
df -h ~/avatar_tool/outputs/

# List videos by size
ls -lhS ~/avatar_tool/outputs/*.mp4 | tail -20

# Delete videos older than 30 days
find ~/avatar_tool/outputs/ -name "*.mp4" -mtime +30 -delete
```

---

## 7. Troubleshooting: Top 5 Failure Points

---

### Failure 1 — Ollama not running / Gemma3 not responding

**Symptom:** Emotion tagger test fails. Emotions all come back as `neutral` on every sentence. Health check shows `❌ ollama`.

**Cause:** Ollama service stopped (common after EC2 stop/start).

**Fix:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If connection refused:
sudo systemctl start ollama
sleep 3
ollama list  # should show gemma3:4b

# If gemma3:4b is missing:
ollama pull gemma3:4b

# Verify emotion tagging works:
source venv/bin/activate
python -c "
from scripts.emotion_tagger import tag_script
r = tag_script('This is amazing! This is a critical issue.')
print([x['emotion'] for x in r])
"
# Expected: ['excited', 'urgent']  (not ['neutral', 'neutral'])
```

**Prevention:** Add Ollama to supervisor or systemd so it starts on boot.

---

### Failure 2 — TTS produces silence or no audio file

**Symptom:** `TTS smoke test` fails. Generation status shows "Audio generation failed." Log shows `No audio segments generated`.

**Cause A:** Chatterbox failed to load (usually CUDA OOM or import error).  
**Cause B:** All TTS engines including gTTS failed (no internet).  
**Cause C:** ffmpeg concat step failed.

**Diagnose:**
```bash
source venv/bin/activate
python -c "
from scripts.tts_engine import TTSEngine
e = TTSEngine()
print('Chatterbox:', e.chatterbox.is_available())
print('Coqui:', e.coqui.is_available())
print('Svara:', e.svara.is_available())
"
```

**Fix A — Chatterbox CUDA OOM:**
```bash
# Check VRAM usage
nvidia-smi
# Kill any hanging processes using GPU memory
sudo fuser -k /dev/nvidia0

# Then retry — Chatterbox will load fresh
sudo supervisorctl restart avatar_tool
```

**Fix B — No TTS at all:**
```bash
# Install gTTS as absolute last-resort fallback
pip install gtts
# Verify internet access
curl -s https://api.groq.com --head | head -1
```

**Fix C — ffmpeg concat:**
```bash
# Test ffmpeg is working
ffmpeg -version | head -1
# Should show: ffmpeg version 4.x

# If missing:
sudo apt-get install -y ffmpeg
```

---

### Failure 3 — Video generation hangs or produces no video

**Symptom:** Progress bar stalls at "Animating avatar" (55%) for more than 10 minutes. Job eventually returns "Video generation had issues but audio was created."

**Cause A:** MuseTalk or LivePortrait subprocess crashed silently.  
**Cause B:** GPU out of memory during video generation.  
**Cause C:** No base driving video for LivePortrait (falls to SadTalker which also fails).

**Diagnose:**
```bash
# Check GPU memory during generation
watch -n 2 nvidia-smi

# Check logs for subprocess errors
tail -100 ~/avatar_tool/logs/app.err.log | grep -i "error\|failed\|killed"

# Verify model weights are present
ls -lh ~/avatar_tool/models/weights/musetalk/pytorch_model.bin
ls -lh ~/avatar_tool/models/weights/liveportrait/*.safetensors | wc -l
# Should show 5 files

# Verify driving video exists
ls ~/avatar_tool/web/assets/base_driving_neutral.mp4
```

**Fix A — Regenerate driving video if missing:**
```bash
FIRST_AVATAR=$(ls ~/avatar_tool/avatars/*.png | head -1)
ffmpeg -loop 1 -i "$FIRST_AVATAR" \
    -c:v libx264 -tune stillimage -pix_fmt yuv420p \
    -t 10 -r 25 ~/avatar_tool/web/assets/base_driving_neutral.mp4 -y
```

**Fix B — GPU OOM:**
```bash
# Free GPU memory and restart app
sudo supervisorctl stop avatar_tool
sleep 5
nvidia-smi  # confirm GPU memory freed
sudo supervisorctl start avatar_tool
# Use shorter scripts (under 2 minutes) until confirmed stable
```

**Fix C — Re-download missing weights:**
```bash
source venv/bin/activate
python scripts/download_weights.py
```

---

### Failure 4 — Gradio UI not accessible at http://IP

**Symptom:** Browser shows "This site can't be reached" or "502 Bad Gateway".

**Cause A:** Nginx not running.  
**Cause B:** Avatar tool app not running (Gradio on 7860 is down).  
**Cause C:** Security group blocking port 80.  
**Cause D:** App still loading (models take 15-20 seconds to init).

**Diagnose:**
```bash
# Check Nginx
sudo systemctl status nginx
curl -s http://localhost/  # should return Gradio HTML

# Check app
sudo supervisorctl status avatar_tool
curl -s http://localhost:7860/  # should return Gradio HTML

# Check port 80 listening
sudo ss -tlnp | grep :80
sudo ss -tlnp | grep :7860
```

**Fix A — Nginx down:**
```bash
sudo nginx -t  # check for config errors
sudo systemctl restart nginx
```

**Fix B — App not running:**
```bash
sudo supervisorctl start avatar_tool
sleep 20  # wait for models to load
tail -20 ~/avatar_tool/logs/app.out.log
# Should end with "Engines ready."
```

**Fix C — Security group:**
Go to AWS Console → EC2 → Security Groups → your group → Inbound rules → verify port 80 is open to `0.0.0.0/0`.

**Fix D — App still loading:**
```bash
tail -f ~/avatar_tool/logs/app.out.log
# Wait for: "Engines ready." before testing browser
```

---

### Failure 5 — YouTube ingestion fails or transcript is empty

**Symptom:** YouTube mode returns "Could not extract content from YouTube URL" or generates a video with the wrong/empty script.

**Cause A:** yt-dlp outdated (YouTube changes their API regularly).  
**Cause B:** Video has no captions and audio download fails.  
**Cause C:** Whisper large-v3 OOM during transcription.  
**Cause D:** LLM script generation returned empty (Ollama + Groq both down).

**Diagnose:**
```bash
source venv/bin/activate

# Test yt-dlp directly
yt-dlp --get-title "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Test audio download
yt-dlp --extract-audio --audio-format wav -o /tmp/test_audio.%(ext)s \
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
ls -lh /tmp/test_audio.wav
```

**Fix A — Update yt-dlp:**
```bash
pip install -U yt-dlp
yt-dlp --version  # should be recent 2024.x
```

**Fix B — Whisper OOM:**
```bash
# Switch to smaller whisper model in content_ingester.py line 192:
# model = whisper.load_model("large-v3")  →  model = whisper.load_model("medium")
# The medium model uses ~5GB VRAM instead of 10GB
nano ~/avatar_tool/scripts/content_ingester.py
# Change "large-v3" to "medium" on line 192
sudo supervisorctl restart avatar_tool
```

**Fix C — Ollama down + no Groq key:**
```bash
# Quick check
ollama list
# If empty: ollama pull gemma3:4b

# Set Groq as backup (free at groq.com)
echo "GROQ_API_KEY=your_key" >> ~/avatar_tool/.env
sudo supervisorctl restart avatar_tool
```

---

## Quick Reference Card

```bash
# ── Daily Operations ───────────────────────────────────────
bash start.sh                          # Start app
bash stop.sh                           # Stop app
sudo supervisorctl restart avatar_tool # Restart after code change
tail -f logs/app.out.log               # Live app log
tail -f logs/app.err.log               # Live error log

# ── Health Checks ──────────────────────────────────────────
nvidia-smi                             # GPU status
ollama list                            # LLM models
curl http://localhost:7860/            # App responding?
source venv/bin/activate && python test_all.py  # Full test

# ── Disk Management ────────────────────────────────────────
df -h                                  # Disk space
du -sh outputs/                        # Video storage used
find outputs/ -name "*.mp4" -mtime +30 -delete  # Cleanup old videos

# ── Restart from Scratch ───────────────────────────────────
sudo supervisorctl stop avatar_tool
sudo systemctl restart ollama
sleep 5
sudo supervisorctl start avatar_tool
```
