#!/bin/bash
# ============================================================
#  AVATAR STUDIO — ONE-TIME SETUP
#  Run ONCE on fresh EC2 g4dn.2xlarge (Ubuntu 22.04)
#  Deep Learning AMI recommended (CUDA pre-installed)
#  Usage: bash setup.sh
# ============================================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "\n${CYAN}━━━ $1 ━━━${NC}"; }

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$TOOL_DIR/models"

# Verify we're on the right machine
if ! command -v nvidia-smi &>/dev/null; then
    warn "No GPU detected. Some features will be slow or unavailable."
fi

# ─── STEP 1: System packages ─────────────────────────────────────────────────
step "1/10 System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg git curl wget unzip build-essential cmake \
    python3-pip python3-venv python3-dev \
    libsndfile1 libportaudio2 portaudio19-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libopencv-dev \
    nginx supervisor \
    fonts-dejavu fonts-dejavu-core fonts-dejavu-extra \
    2>/dev/null
log "System packages installed"

# ─── STEP 2: Python virtualenv ───────────────────────────────────────────────
step "2/10 Python virtualenv"
python3 -m venv "$TOOL_DIR/venv"
source "$TOOL_DIR/venv/bin/activate"
pip install --upgrade pip wheel setuptools -q
log "Virtualenv ready at $TOOL_DIR/venv"

# ─── STEP 3: PyTorch (CUDA 11.8) then requirements ───────────────────────────
step "3/10 Python dependencies"
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -r "$TOOL_DIR/requirements.txt"
log "Python dependencies installed"

# ─── STEP 4: Clone model repositories ────────────────────────────────────────
step "4/10 Clone model repositories"
cd "$TOOL_DIR"

clone_if_missing() {
    local name="$1" url="$2" dest="$3"
    if [ ! -d "$dest" ]; then
        git clone "$url" "$dest" -q && log "$name cloned" || warn "$name clone failed"
        if [ -f "$dest/requirements.txt" ]; then
            pip install -q -r "$dest/requirements.txt" 2>/dev/null || true
        fi
    else
        warn "$name already exists, skipping"
    fi
}

# EchoMimicV2 — PRIMARY animation engine (upper body + hands, CVPR 2025, Apache 2.0)
clone_if_missing "EchoMimicV2" \
    "https://github.com/antgroup/echomimic_v2.git" \
    "$MODELS_DIR/EchoMimicV2"

# LatentSync — Lip sync (ByteDance, Apache 2.0, language-agnostic)
clone_if_missing "LatentSync" \
    "https://github.com/bytedance/LatentSync.git" \
    "$MODELS_DIR/LatentSync"

# MusePose — Dance and full body pose transfer (Apache 2.0)
clone_if_missing "MusePose" \
    "https://github.com/TMElyralab/MusePose.git" \
    "$MODELS_DIR/MusePose"

# CyberHost — Full body talking diffusion (fallback)
clone_if_missing "CyberHost" \
    "https://github.com/deepbrainai-research/cyberhost.git" \
    "$MODELS_DIR/CyberHost" || true  # May not exist yet — non-fatal

# SadTalker — Head animation fallback
clone_if_missing "SadTalker" \
    "https://github.com/OpenTalker/SadTalker.git" \
    "$MODELS_DIR/SadTalker"

# Real-ESRGAN — 4x upscale to 1080p
clone_if_missing "Real-ESRGAN" \
    "https://github.com/xinntao/Real-ESRGAN.git" \
    "$MODELS_DIR/Real-ESRGAN"
if [ -d "$MODELS_DIR/Real-ESRGAN" ]; then
    pip install -q basicsr realesrgan 2>/dev/null || true
fi

log "Model repositories cloned"

# ─── STEP 5: Download model weights ──────────────────────────────────────────
step "5/10 Download model weights"
mkdir -p "$MODELS_DIR/weights"

download_weight() {
    local name="$1" url="$2" dest="$3"
    if [ ! -f "$dest" ]; then
        log "Downloading $name..."
        wget -q --show-progress -O "$dest" "$url" || warn "$name download failed"
    else
        warn "$name already exists"
    fi
}

# GFPGAN v1.4 — Face enhancement
download_weight "GFPGANv1.4" \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
    "$MODELS_DIR/weights/GFPGANv1.4.pth"

# Real-ESRGAN x2 (for GFPGAN bg upsampler)
download_weight "RealESRGAN_x2plus" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
    "$MODELS_DIR/weights/RealESRGAN_x2plus.pth"

# Real-ESRGAN x4 (for video upscale)
download_weight "RealESRGAN_x4plus" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    "$MODELS_DIR/weights/RealESRGAN_x4plus.pth"

# LatentSync checkpoint
if [ ! -f "$MODELS_DIR/LatentSync/checkpoints/latentsync_unet.pt" ]; then
    log "Downloading LatentSync checkpoint..."
    mkdir -p "$MODELS_DIR/LatentSync/checkpoints"
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='ByteDance/LatentSync', filename='latentsync_unet.pt')
shutil.copy(path, '$MODELS_DIR/LatentSync/checkpoints/latentsync_unet.pt')
print('LatentSync checkpoint downloaded')
" || warn "LatentSync checkpoint download failed — will retry on first use"
fi

# EchoMimicV2 weights
if [ -d "$MODELS_DIR/EchoMimicV2" ]; then
    log "Downloading EchoMimicV2 weights..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='BadToBest/EchoMimicV2', local_dir='$MODELS_DIR/EchoMimicV2/weights')
print('EchoMimicV2 weights downloaded')
" || warn "EchoMimicV2 weights download failed"
fi

log "Model weights downloaded"

# ─── STEP 6: TTS engines ─────────────────────────────────────────────────────
step "6/10 TTS engines"

# IndicF5 — Indic TTS (VERIFY LICENSE before production)
# License check: https://huggingface.co/ai4bharat/IndicF5
log "Pre-downloading IndicF5 (ai4bharat/IndicF5)..."
python3 -c "
from transformers import pipeline
try:
    p = pipeline('text-to-speech', model='ai4bharat/IndicF5')
    print('IndicF5 ready')
except Exception as e:
    print(f'IndicF5 pre-download failed (will load on first use): {e}')
" || warn "IndicF5 pre-download failed"

# Chatterbox — English TTS with voice cloning
pip install -q chatterbox-tts 2>/dev/null || warn "Chatterbox install failed"

# Coqui XTTS v2 — multilingual fallback
pip install -q TTS 2>/dev/null || warn "Coqui TTS install failed"

# Pedalboard — audio effects
pip install -q pedalboard 2>/dev/null || warn "Pedalboard install failed"

# Voice cloning directory
mkdir -p "$MODELS_DIR/voices"
log "Voice clone dir ready: $MODELS_DIR/voices/"
log "  Drop te_female.wav, en_male.wav etc. here for voice cloning"

log "TTS engines ready"

# ─── STEP 7: Generate default avatars ────────────────────────────────────────
step "7/10 Generate default South Indian avatars"
python3 "$TOOL_DIR/scripts/generate_avatars.py" || warn "Avatar generation failed — run manually later"
log "Avatars generated"

# ─── STEP 8: Download background video loops ─────────────────────────────────
step "8/10 Background video library"
python3 "$TOOL_DIR/scripts/download_backgrounds.py" || warn "Background download failed"
log "Backgrounds ready"

# ─── STEP 9: Ollama ──────────────────────────────────────────────────────────
step "9/10 Ollama (local LLM)"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama installed"
else
    warn "Ollama already installed"
fi

sudo systemctl enable ollama 2>/dev/null || true
sudo systemctl start ollama 2>/dev/null || (ollama serve &>/dev/null &)
sleep 5

ollama pull gemma3:4b && log "Gemma3 4B ready" || warn "Gemma3 pull failed — retry: ollama pull gemma3:4b"

# ─── STEP 10: Nginx + Supervisor ─────────────────────────────────────────────
step "10/10 Nginx + Supervisor"

# Nginx
sudo tee /etc/nginx/sites-available/avatar_studio > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;
    client_max_body_size 500M;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /outputs/ {
        alias TOOL_DIR_PLACEHOLDER/outputs/;
        add_header Content-Disposition "attachment";
    }
}
NGINX

sudo sed -i "s|TOOL_DIR_PLACEHOLDER|$TOOL_DIR|g" /etc/nginx/sites-available/avatar_studio
sudo ln -sf /etc/nginx/sites-available/avatar_studio /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx && log "Nginx configured"

# Supervisor
mkdir -p "$TOOL_DIR/logs"
sudo tee /etc/supervisor/conf.d/avatar_studio.conf > /dev/null <<SUPERVISOR
[program:avatar_studio]
command=$TOOL_DIR/venv/bin/python $TOOL_DIR/app.py
directory=$TOOL_DIR
user=ubuntu
autostart=true
autorestart=true
stopwaitsecs=30
stderr_logfile=$TOOL_DIR/logs/app.err.log
stdout_logfile=$TOOL_DIR/logs/app.out.log
environment=PATH="$TOOL_DIR/venv/bin:%(ENV_PATH)s",HOME="/home/ubuntu"
SUPERVISOR

sudo supervisorctl reread && sudo supervisorctl update && log "Supervisor configured"

# ─── Done ────────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR-EC2-IP")
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  AVATAR STUDIO SETUP COMPLETE!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Browser:   ${CYAN}http://$PUBLIC_IP${NC}"
echo -e "  Logs:      ${CYAN}tail -f $TOOL_DIR/logs/app.out.log${NC}"
echo -e "  Status:    ${CYAN}sudo supervisorctl status avatar_studio${NC}"
echo -e "  Restart:   ${CYAN}sudo supervisorctl restart avatar_studio${NC}"
echo ""
echo -e "${YELLOW}  IMPORTANT:${NC}"
echo -e "  1. Verify IndicF5 license: huggingface.co/ai4bharat/IndicF5"
echo -e "  2. Drop voice WAV files in: $MODELS_DIR/voices/"
echo -e "  3. Port 7860 is NOT exposed (Nginx proxies on 80) ✓"
echo ""
