#!/bin/bash
# ============================================================
#  AVATAR TOOL — ONE-TIME SETUP
#  Run this ONCE on a fresh EC2 g4dn.2xlarge (Ubuntu 22.04)
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

# ─── STEP 1: System packages ────────────────────────────────
step "1/9 System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg git curl wget unzip build-essential \
    python3-pip python3-venv python3-dev \
    libsndfile1 libportaudio2 portaudio19-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    nginx supervisor 2>/dev/null
log "System packages installed"

# ─── STEP 2: Python virtualenv ──────────────────────────────
step "2/9 Python virtualenv"
python3 -m venv "$TOOL_DIR/venv"
source "$TOOL_DIR/venv/bin/activate"
pip install --upgrade pip wheel setuptools -q
log "Virtualenv created at $TOOL_DIR/venv"

# ─── STEP 3: Core Python dependencies ───────────────────────
step "3/9 Python dependencies"
pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    gradio==4.44.0 \
    numpy opencv-python-headless \
    scipy soundfile librosa \
    yt-dlp \
    openai-whisper \
    requests httpx \
    Pillow \
    facenet-pytorch \
    gfpgan \
    basicsr \
    realesrgan \
    imageio imageio-ffmpeg \
    tqdm \
    pydub \
    transformers accelerate \
    langchain-community \
    ollama
log "Python dependencies installed"

# ─── STEP 4: Clone core model repos ─────────────────────────
step "4/9 Clone model repositories"
cd "$TOOL_DIR"

# MuseTalk
if [ ! -d "models/MuseTalk" ]; then
    git clone https://github.com/TMElyralab/MuseTalk.git models/MuseTalk -q
    pip install -q -r models/MuseTalk/requirements.txt 2>/dev/null || true
    log "MuseTalk cloned"
else
    warn "MuseTalk already exists, skipping"
fi

# LivePortrait
if [ ! -d "models/LivePortrait" ]; then
    git clone https://github.com/KwaiVGI/LivePortrait.git models/LivePortrait -q
    pip install -q -r models/LivePortrait/requirements.txt 2>/dev/null || true
    log "LivePortrait cloned"
else
    warn "LivePortrait already exists, skipping"
fi

# SadTalker (fallback)
if [ ! -d "models/SadTalker" ]; then
    git clone https://github.com/OpenTalker/SadTalker.git models/SadTalker -q
    pip install -q -r models/SadTalker/requirements.txt 2>/dev/null || true
    log "SadTalker cloned"
else
    warn "SadTalker already exists, skipping"
fi

# Svara TTS (Indic languages incl. Telugu)
pip install -q svara-tts 2>/dev/null || \
    pip install -q git+https://github.com/Kenpath/svara-tts-inference.git 2>/dev/null || \
    warn "Svara TTS pip install failed — will use HuggingFace inference"

# Chatterbox TTS (English)
pip install -q chatterbox-tts 2>/dev/null || \
    warn "Chatterbox pip install failed — will use Coqui fallback"

# Coqui TTS (fallback English TTS)
pip install -q TTS 2>/dev/null || warn "Coqui TTS install failed"

log "All model repos cloned"

# ─── STEP 5: Download model weights ─────────────────────────
step "5/9 Download model weights"
python3 "$TOOL_DIR/scripts/download_weights.py"
log "Model weights downloaded"

# ─── STEP 6: Install and start Ollama ───────────────────────
step "6/9 Ollama (local LLM for emotion tagging)"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama installed"
else
    warn "Ollama already installed"
fi

# Start Ollama service
sudo systemctl enable ollama 2>/dev/null || true
sudo systemctl start ollama 2>/dev/null || (ollama serve &>/dev/null &)
sleep 3

# Pull Gemma3 4B for emotion tagging
ollama pull gemma3:4b && log "Gemma3 4B pulled" || warn "Gemma3 pull failed — will retry on first run"

# ─── STEP 7: Generate default South Indian avatars ──────────
step "7/9 Generate default South Indian avatars"
python3 "$TOOL_DIR/scripts/generate_avatars.py"
log "Default avatars generated"

# ─── STEP 8: Nginx config ───────────────────────────────────
step "8/9 Configure Nginx"
sudo tee /etc/nginx/sites-available/avatar_tool > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;
    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }

    location /outputs/ {
        alias /home/ubuntu/avatar_tool/outputs/;
        add_header Content-Disposition "attachment";
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/avatar_tool /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
log "Nginx configured"

# ─── STEP 9: Supervisor for auto-start ──────────────────────
step "9/9 Supervisor (auto-restart on crash)"
sudo tee /etc/supervisor/conf.d/avatar_tool.conf > /dev/null <<SUPERVISOR
[program:avatar_tool]
command=$TOOL_DIR/venv/bin/python $TOOL_DIR/app.py
directory=$TOOL_DIR
user=ubuntu
autostart=true
autorestart=true
stderr_logfile=$TOOL_DIR/logs/app.err.log
stdout_logfile=$TOOL_DIR/logs/app.out.log
environment=PATH="$TOOL_DIR/venv/bin:%(ENV_PATH)s"
SUPERVISOR
sudo supervisorctl reread && sudo supervisorctl update
log "Supervisor configured"

# ─── Done ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  SETUP COMPLETE!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Open browser: ${CYAN}http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR-EC2-IP')${NC}"
echo -e "  Logs: ${CYAN}tail -f $TOOL_DIR/logs/app.out.log${NC}"
echo -e "  Start: ${CYAN}bash start.sh${NC}"
echo -e "  Stop:  ${CYAN}bash stop.sh${NC}"
echo ""
