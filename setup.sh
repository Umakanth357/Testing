#!/usr/bin/env bash
# Avatar Studio v3.0 — EC2 Setup Script
# Target: Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04), Tesla T4, CUDA 12.x
#
# PIPELINE: SadTalker (head/blink/expression) → MuseTalk (lip sync) → GFPGAN (face quality)
#
# INSTALL ORDER — CRITICAL:
#   numpy MUST be pinned at 1.26.4 before any ML package
#   gfpgan >=1.3.8 (older requires numpy<1.23 — kills everything)
#   SadTalker before MuseTalk (different requirements, easier to manage)
#   Torch/torchvision skipped if AMI already has CUDA working version
#
# Run: chmod +x setup.sh && ./setup.sh 2>&1 | tee setup.log
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

check_numpy() {
    local ver
    ver=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "missing")
    if [[ "$ver" != "1.26"* ]]; then
        warn "numpy drifted to $ver — re-pinning to 1.26.4..."
        pip install "numpy==1.26.4" --force-reinstall --quiet
    else
        info "numpy $ver OK"
    fi
}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Avatar Studio v3.0 — EC2 Setup         ║"
echo "║   SadTalker + MuseTalk + GFPGAN           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 0. Pre-flight ─────────────────────────────────────────────────────────────
[ ! -f .env ] && error ".env not found. Create it:\n  echo 'HF_TOKEN=hf_...' > .env"
source .env
[ -z "${HF_TOKEN:-}" ] && error "HF_TOKEN not set in .env"

info "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
info "Python : $(python3 --version)"
info "Disk   : $(df -h . | tail -1 | awk '{print $4}') free"

# ── 1. System packages ────────────────────────────────────────────────────────
info "[1/12] System packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs curl wget unzip \
    libsndfile1 libsndfile1-dev \
    libgl1-mesa-glx libglib2.0-0 \
    python3-pip python3-venv \
    libboost-all-dev cmake \
    2>/dev/null
git lfs install --skip-repo 2>/dev/null || true

# ── 2. Virtual environment ────────────────────────────────────────────────────
info "[2/12] Python virtual environment..."
[ ! -d venv ] && python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools --quiet

# ── 3. Pin numpy FIRST ────────────────────────────────────────────────────────
info "[3/12] Pinning numpy==1.26.4 (MUST come first)..."
pip install "numpy==1.26.4" --quiet
check_numpy

# ── 4. PyTorch ────────────────────────────────────────────────────────────────
info "[4/12] Checking PyTorch + CUDA..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python -c 'import torch; print(torch.__version__)')
    info "PyTorch $TORCH_VER with CUDA — skipping reinstall"
else
    warn "PyTorch CUDA not found — installing PyTorch 2.1 + CUDA 12.1..."
    pip install \
        "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2" \
        --index-url https://download.pytorch.org/whl/cu121 --quiet
fi
check_numpy

# ── 5. Core ML packages ───────────────────────────────────────────────────────
info "[5/12] Core ML packages..."
pip install \
    "transformers>=4.40.0" \
    "diffusers>=0.31.0" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.0" \
    "peft>=0.10.0" \
    "huggingface-hub>=0.22.0" \
    "omegaconf>=2.3.0" \
    --quiet
check_numpy

# ── 6. Image restoration stack ───────────────────────────────────────────────
info "[6/12] Image restoration (basicsr → facexlib → gfpgan >= 1.3.8)..."
pip install "Pillow>=10.0.0" "opencv-python>=4.9.0" --quiet
pip install "basicsr>=1.4.2" --quiet
pip install "facexlib>=0.3.0" --quiet
pip install "gfpgan>=1.3.8" --quiet
pip install "realesrgan>=0.3.0" --quiet
check_numpy

# ── 7. TTS stack ─────────────────────────────────────────────────────────────
info "[7/12] TTS stack..."
pip install \
    "TTS>=0.22.0" \
    "gtts>=2.5.0" \
    "soundfile>=0.12.0" \
    "sounddevice>=0.4.6" \
    "librosa>=0.10.0" \
    "pyloudnorm" \
    "edge-tts>=6.1.0" \
    --quiet
check_numpy

# ── 8. Audio/video utilities ──────────────────────────────────────────────────
info "[8/12] Audio/video utilities..."
pip install \
    "pedalboard>=0.9.0" \
    "pydub>=0.25.0" \
    "scipy>=1.11.0" \
    "ffmpeg-python>=0.2.0" \
    "imageio>=2.33.0" \
    "imageio-ffmpeg>=0.4.9" \
    "PyYAML>=6.0" \
    --quiet
check_numpy

# ── 9. Script engine + UI ─────────────────────────────────────────────────────
info "[9/12] Script engine, utilities, Gradio UI..."
pip install \
    "openai-whisper>=20231117" \
    "faster-whisper>=1.0.0" \
    "yt-dlp>=2024.1.0" \
    "youtube-transcript-api>=0.6.0" \
    "requests>=2.31.0" \
    "httpx>=0.27.0" \
    "python-dotenv>=1.0.0" \
    "tqdm>=4.66.0" \
    "psutil>=5.9.0" \
    "rich>=13.7.0" \
    "gradio==4.44.1" \
    "starlette<1.0.0" \
    "instaloader" \
    --quiet
check_numpy

# ── Patch gradio_client bool schema bug ───────────────────────────────────────
info "Patching gradio_client bool schema bug..."
python3 - <<'PYEOF'
import sys
from pathlib import Path

utils_path = None
for p in sys.path:
    candidate = Path(p) / "gradio_client" / "utils.py"
    if candidate.exists():
        utils_path = candidate
        break

if not utils_path:
    print("  SKIP: gradio_client/utils.py not found")
    sys.exit(0)

txt = utils_path.read_text()
patched = False

old1 = 'def get_type(schema: dict):\n    if "const" in schema:'
new1 = 'def get_type(schema: dict):\n    if not isinstance(schema, dict): return "any"\n    if "const" in schema:'
if old1 in txt:
    txt = txt.replace(old1, new1); patched = True

old2 = "f\"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}\""
new2 = "f\"str, {_json_schema_to_python_type(schema['additionalProperties'], defs) if isinstance(schema['additionalProperties'], dict) else 'any'}\""
if old2 in txt:
    txt = txt.replace(old2, new2); patched = True

utils_path.write_text(txt)
print(f"  {'PATCHED' if patched else 'already patched'}: {utils_path}")
PYEOF

# ── 10. SadTalker (head motion + eye blink + expressions) ────────────────────
# SadTalker by OpenTalker (CVPR 2023) — MIT license
# audio-driven: head pose + expression + eye blink from speech audio
# ~700MB weights, ~8 min generation time for 10 min video
info "[10/12] Installing SadTalker..."
mkdir -p models

if [ ! -d "models/SadTalker" ]; then
    info "Cloning SadTalker..."
    git clone --depth=1 https://github.com/OpenTalker/SadTalker.git models/SadTalker
fi

# SadTalker requirements
if [ -f "models/SadTalker/requirements.txt" ]; then
    info "Installing SadTalker requirements..."
    pip install -r models/SadTalker/requirements.txt --quiet 2>/dev/null || \
        warn "Some SadTalker deps failed — non-blocking"
fi
check_numpy

# Download SadTalker weights
info "Downloading SadTalker weights (~700MB)..."
python3 - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
ckpt_dir = Path("models/SadTalker/checkpoints")
gfpgan_dir = Path("models/SadTalker/gfpgan/weights")
ckpt_dir.mkdir(parents=True, exist_ok=True)
gfpgan_dir.mkdir(parents=True, exist_ok=True)

try:
    snapshot_download(
        "vinthony/SadTalker",
        local_dir=str(ckpt_dir),
        token=token,
        ignore_patterns=["*.git*"],
    )
    print("  OK: SadTalker weights")
except Exception as e:
    print(f"  WARN: SadTalker weights — {e}")

try:
    snapshot_download(
        "TencentARC/GFPGAN",
        local_dir=str(gfpgan_dir),
        token=token,
        ignore_patterns=["*.git*", "experiments/*"],
    )
    print("  OK: GFPGAN weights for SadTalker")
except Exception as e:
    print(f"  WARN: GFPGAN weights — {e}")
PYEOF

# ── 11. MuseTalk (lip sync) ───────────────────────────────────────────────────
# MuseTalk by ByteDance — Apache 2.0 license
# Best open-source lip sync 2025, ~30fps on T4, ~3.5GB weights
info "[11/12] Installing MuseTalk lip sync..."

if [ ! -d "models/MuseTalk" ]; then
    info "Cloning MuseTalk..."
    git clone --depth=1 https://github.com/TMElyralab/MuseTalk.git models/MuseTalk
fi

if [ -f "models/MuseTalk/requirements.txt" ]; then
    info "Installing MuseTalk requirements..."
    pip install -r models/MuseTalk/requirements.txt --quiet 2>/dev/null || \
        warn "Some MuseTalk deps failed — non-blocking"
fi
check_numpy

info "Downloading MuseTalk weights (~3.5GB)..."
python3 - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
weights_dir = Path("models/MuseTalk/models")
weights_dir.mkdir(parents=True, exist_ok=True)

repos = [
    ("TMElyralab/MuseTalk",        str(weights_dir),             []),
    ("stabilityai/sd-vae-ft-mse",  str(weights_dir / "sd-vae-ft-mse"), []),
]
for repo_id, local_dir, ignore in repos:
    try:
        snapshot_download(repo_id, local_dir=local_dir, token=token,
                          ignore_patterns=ignore or None)
        print(f"  OK: {repo_id}")
    except Exception as e:
        print(f"  WARN: {repo_id} — {e}")
PYEOF

# ── 12. Ollama + models + Nginx ───────────────────────────────────────────────
info "[12/12] Ollama, LLM, Nginx..."

# Ollama
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
pgrep -x ollama >/dev/null || (ollama serve >/dev/null 2>&1 &)
sleep 6
ollama pull llama3.1:8b
ollama pull gemma3:4b || true

# Nginx
sudo apt-get install -y nginx --quiet
sudo tee /etc/nginx/sites-available/avatar-studio > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;
    client_max_body_size 100M;

    location / {
        proxy_pass         http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/avatar-studio /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

# HuggingFace auth
source .env
export HF_TOKEN=$(grep -i HF_TOKEN .env | sed 's/.*HF_TOKEN[=: ]*//' | tr -d '\r\n ')
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

# ── Assets directory ──────────────────────────────────────────────────────────
mkdir -p assets/fonts models/backgrounds models/avatars outputs logs

# Download Noto Sans Telugu fonts (for subtitles + lower thirds)
info "Downloading Noto Sans Telugu fonts..."
FONT_DIR="assets/fonts"
FONT_BASE="https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu"
wget -q -O "$FONT_DIR/NotoSansTelugu-Regular.ttf" \
    "${FONT_BASE}/NotoSansTelugu-Regular.ttf" 2>/dev/null || \
    warn "Font download failed — fallback fonts will be used"
wget -q -O "$FONT_DIR/NotoSansTelugu-Bold.ttf" \
    "${FONT_BASE}/NotoSansTelugu-Bold.ttf" 2>/dev/null || true

# Generate fallback backgrounds and avatars
info "Generating background images..."
python scripts/download_backgrounds.py

info "Generating avatar images (SDXL, ~5 min)..."
python scripts/generate_avatars.py

# Avatar directory structure
info "Creating avatar directory structure..."
python3 - <<'PYEOF'
from pathlib import Path
import shutil

avatars_dir = Path("models/avatars")
mappings = [
    ("navya_telugu_f", "half_body_professional.png",  "navya_professional.png"),
    ("navya_telugu_f", "half_body_traditional.png",   "navya_traditional.png"),
    ("navya_telugu_f", "half_body_casual.png",         "navya_casual.png"),
    ("navya_telugu_f", "standing_professional.png",   "navya_professional.png"),
    ("priya_telugu_f", "half_body_professional.png",  "navya_professional.png"),
    ("priya_telugu_f", "half_body_traditional.png",   "navya_traditional.png"),
    ("arjun_telugu_m", "half_body_professional.png",  "arjun_suit.png"),
    ("arjun_telugu_m", "half_body_traditional.png",   "arjun_kurta.png"),
    ("arjun_telugu_m", "half_body_casual.png",         "arjun_casual.png"),
    ("arjun_telugu_m", "standing_professional.png",   "arjun_suit.png"),
]
for persona_dir, dest_name, src_name in mappings:
    dest_d = avatars_dir / persona_dir
    dest_d.mkdir(parents=True, exist_ok=True)
    src = avatars_dir / src_name
    dest = dest_d / dest_name
    if src.exists() and not dest.exists():
        shutil.copy2(src, dest)
        print(f"  COPY {src_name} -> {persona_dir}/{dest_name}")
    elif dest.exists():
        print(f"  SKIP {dest}")
    else:
        print(f"  WARN source missing: {src_name}")
print("Avatar directory structure ready.")
PYEOF

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup Complete — Avatar Studio v3.0                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  %-20s %s\n" "GPU:" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
printf "║  %-20s %s\n" "numpy:" "$(python -c 'import numpy; print(numpy.__version__)')"
printf "║  %-20s %s\n" "PyTorch + CUDA:" "$(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
printf "║  %-20s %s\n" "SadTalker:" "$([ -f models/SadTalker/inference.py ] && echo 'installed' || echo 'MISSING')"
printf "║  %-20s %s\n" "MuseTalk:" "$([ -d models/MuseTalk/scripts ] && echo 'installed' || echo 'MISSING')"
printf "║  %-20s %s\n" "GFPGAN:" "$(python -c 'import gfpgan; print("OK")' 2>/dev/null || echo 'MISSING')"
printf "║  %-20s %s\n" "edge-tts:" "$(python -c 'import edge_tts; print("OK")' 2>/dev/null || echo 'MISSING')"
printf "║  %-20s %s\n" "Ollama LLM:" "$(ollama list 2>/dev/null | grep -c 'llama3.1' | xargs echo 'llama3.1:8b models found:')"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo "    source venv/bin/activate"
echo "    python app.py"
echo "    Open: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_EC2_IP')"
echo ""
echo "  Optional — Indic Parler-TTS (best Telugu voice):"
echo "    Request access at huggingface.co/ai4bharat/indic-parler-tts"
echo "    Then run: pip install git+https://github.com/huggingface/parler-tts"
echo ""
echo "  Optional — Pexels real backgrounds:"
echo "    Get free API key at pexels.com/api"
echo "    Add to .env: PEXELS_API_KEY=your_key"
