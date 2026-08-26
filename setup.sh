#!/usr/bin/env bash
# Avatar Studio — EC2 Setup Script
# Target: Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04), Tesla T4, CUDA 12.x/13.x
#
# WHY THIS ORDER MATTERS:
#   - numpy must be pinned at 1.26.4 BEFORE any other ML package installs
#   - gfpgan must be >=1.3.8 (older versions require numpy<1.23 — kills everything)
#   - chatterbox-tts dropped: English-only, no Telugu, causes numpy==1.26.0 hard conflict
#   - torch/torchvision/torchaudio skipped: AMI already has them with CUDA
#   - Each group is installed separately so conflicts surface early and clearly
#
# Run: chmod +x setup.sh && ./setup.sh 2>&1 | tee setup.log
set -euo pipefail

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── 0. Pre-flight checks ──────────────────────────────────────────────────────
echo ""
echo "=== Avatar Studio Setup ==="

if [ ! -f .env ]; then
    error ".env file not found. Run: cp /dev/null .env && nano .env  then add HF_TOKEN=hf_..."
fi
source .env
[ -z "${HF_TOKEN:-}" ] && error "HF_TOKEN not set in .env"

info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
info "Python: $(python3 --version)"

# ── 1. System packages ────────────────────────────────────────────────────────
info "[1/10] System packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs curl wget unzip \
    libsndfile1 libsndfile1-dev \
    libgl1-mesa-glx libglib2.0-0 \
    python3-pip python3-venv 2>/dev/null
git lfs install --skip-repo 2>/dev/null || true

# ── 2. Virtual environment ────────────────────────────────────────────────────
info "[2/10] Python virtual environment..."
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip wheel setuptools --quiet

# ── 3. Pin numpy FIRST — CRITICAL ────────────────────────────────────────────
info "[3/10] Pinning numpy==1.26.4 (must come before everything else)..."
# This MUST be the first pip install. Any package that runs before this
# and pulls numpy will get the wrong version and break the entire stack.
pip install "numpy==1.26.4" --quiet
check_numpy

# ── 4. PyTorch — skip if AMI already has working CUDA version ────────────────
info "[4/10] Checking PyTorch + CUDA..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python -c 'import torch; print(torch.__version__)')
    info "PyTorch $TORCH_VER with CUDA already available — skipping reinstall."
else
    warn "PyTorch CUDA not found — installing PyTorch 2.1 + CUDA 12.1..."
    pip install \
        "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2" \
        --index-url https://download.pytorch.org/whl/cu121 --quiet
fi
check_numpy

# ── 5. Core ML packages ───────────────────────────────────────────────────────
info "[5/10] Core ML packages..."
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
info "[6/10] Image restoration (basicsr -> facexlib -> gfpgan >= 1.3.8 -> realesrgan)..."
# gfpgan MUST be >=1.3.8
# gfpgan <1.3.7 requires numpy<1.23 which breaks the entire stack.
# This is the most common hidden cause of the numpy ResolutionImpossible error.
pip install "Pillow>=10.0.0" "opencv-python>=4.9.0" --quiet
pip install "basicsr>=1.4.2" --quiet
pip install "facexlib>=0.3.0" --quiet
pip install "gfpgan>=1.3.8" --quiet
pip install "realesrgan>=0.3.0" --quiet
check_numpy

# ── 7. TTS stack ─────────────────────────────────────────────────────────────
info "[7/10] TTS stack..."
# chatterbox-tts is intentionally NOT installed:
#   1. It pins numpy==1.26.0 (causes ResolutionImpossible)
#   2. It has NO native Telugu language support (English-first)
#   3. ai4bharat/indic-parler-tts is strictly better for Telugu:
#      1806h training, 6 emotion params, 69 voices, loaded via transformers
pip install \
    "TTS>=0.22.0" \
    "gtts>=2.5.0" \
    "soundfile>=0.12.0" \
    "sounddevice>=0.4.6" \
    "librosa>=0.10.0" \
    "pyloudnorm" \
    --quiet
check_numpy

# ── 8. Audio + video utilities ────────────────────────────────────────────────
info "[8/10] Audio/video utilities..."
pip install \
    "pedalboard>=0.9.0" \
    "pydub>=0.25.0" \
    "scipy>=1.11.0" \
    "ffmpeg-python>=0.2.0" \
    "imageio>=2.33.0" \
    "imageio-ffmpeg>=0.4.9" \
    --quiet
check_numpy

# ── 9. Script engine + UI ─────────────────────────────────────────────────────
info "[9/10] Script engine, utilities, Gradio UI..."
pip install \
    "openai-whisper>=20231117" \
    "faster-whisper>=1.0.0" \
    "yt-dlp>=2024.1.0" \
    "youtube-transcript-api>=0.6.0" \
    "requests>=2.31.0" \
    "httpx>=0.27.0" \
    "sentence-transformers>=2.7.0" \
    "python-dotenv>=1.0.0" \
    "tqdm>=4.66.0" \
    "psutil>=5.9.0" \
    "rich>=13.7.0" \
    "gradio==4.44.1" \
    "starlette<1.0.0" \
    "edge-tts>=6.1.0" \
    --quiet
check_numpy

# ── Patch gradio_client utils.py (bool schema bug in 4.44.x) ─────────────────
info "Patching gradio_client for bool additionalProperties schema bug..."
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
    txt = txt.replace(old1, new1)
    patched = True

old2 = "f\"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}\""
new2 = "f\"str, {_json_schema_to_python_type(schema['additionalProperties'], defs) if isinstance(schema['additionalProperties'], dict) else 'any'}\""
if old2 in txt:
    txt = txt.replace(old2, new2)
    patched = True

utils_path.write_text(txt)
print(f"  {'PATCHED' if patched else 'already patched or not needed'}: {utils_path}")
PYEOF

# ── 10. Animation models + weights ───────────────────────────────────────────
info "[10/10] Cloning animation models..."
mkdir -p models

if [ ! -d "models/EchoMimicV2" ]; then
    git clone --depth=1 https://github.com/antgroup/echomimic_v2.git models/EchoMimicV2
fi

if [ ! -d "models/LatentSync" ]; then
    git clone --depth=1 https://github.com/bytedance/LatentSync.git models/LatentSync
fi

info "Downloading model weights from HuggingFace (~10-15 min)..."
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
os.makedirs("models/weights", exist_ok=True)

for repo_id, local_dir, ignore in [
    ("BadToBest/EchoMimicV2",    "models/weights/EchoMimicV2", ["*.bin"]),
    ("ByteDance/LatentSync-1.5", "models/weights/LatentSync",  []),
]:
    try:
        snapshot_download(repo_id, local_dir=local_dir, token=token,
                          ignore_patterns=ignore or None)
        print(f"  OK: {repo_id}")
    except Exception as e:
        print(f"  SKIP: {repo_id} — {e}")
PYEOF

# ── Ollama + Gemma3 ───────────────────────────────────────────────────────────
info "Installing Ollama + Gemma3:4b..."
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
pgrep -x ollama >/dev/null || (ollama serve >/dev/null 2>&1 &)
sleep 6
ollama pull gemma3:4b

# ── Nginx ─────────────────────────────────────────────────────────────────────
info "Configuring Nginx (port 80 -> 127.0.0.1:7860)..."
sudo apt-get install -y nginx --quiet
sudo tee /etc/nginx/sites-available/avatar-studio > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;
    client_max_body_size 50M;

    location / {
        proxy_pass         http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/avatar-studio /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

# ── HF Auth ──────────────────────────────────────────────────────────────────
info "Authenticating with HuggingFace..."
source .env
export HF_TOKEN=$(grep -i HF_TOKEN .env | sed 's/.*HF_TOKEN[=: ]*//' | tr -d '\r\n ')
hf auth login --token "$HF_TOKEN" 2>/dev/null || huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

# ── Generate backgrounds + avatars ───────────────────────────────────────────
info "Generating background images..."
python scripts/download_backgrounds.py

info "Generating avatar images (SDXL, ~5 min)..."
python scripts/generate_avatars.py

# ── Avatar directory structure (app expects persona_id/pose_attire.png) ───────
info "Creating avatar directory structure..."
python3 - <<'PYEOF'
from pathlib import Path
import shutil

avatars_dir = Path("models/avatars")

# Map: (persona_dir, pose_attire.png) -> source file
mappings = [
    # navya_telugu_f
    ("navya_telugu_f", "half_body_professional.png",  "navya_professional.png"),
    ("navya_telugu_f", "half_body_traditional.png",   "navya_traditional.png"),
    ("navya_telugu_f", "half_body_casual.png",         "navya_casual.png"),
    ("navya_telugu_f", "standing_professional.png",   "navya_professional.png"),
    # priya_telugu_f (same avatars as navya)
    ("priya_telugu_f", "half_body_professional.png",  "navya_professional.png"),
    ("priya_telugu_f", "half_body_traditional.png",   "navya_traditional.png"),
    ("priya_telugu_f", "half_body_casual.png",         "navya_casual.png"),
    # arjun_telugu_m
    ("arjun_telugu_m", "half_body_professional.png",  "arjun_suit.png"),
    ("arjun_telugu_m", "half_body_traditional.png",   "arjun_kurta.png"),
    ("arjun_telugu_m", "half_body_casual.png",         "arjun_casual.png"),
    ("arjun_telugu_m", "standing_professional.png",   "arjun_suit.png"),
]

for persona_dir, dest_name, src_name in mappings:
    dest_dir = avatars_dir / persona_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = avatars_dir / src_name
    dest = dest_dir / dest_name
    if src.exists() and not dest.exists():
        shutil.copy2(src, dest)
        print(f"  COPY {src_name} -> {persona_dir}/{dest_name}")
    elif dest.exists():
        print(f"  SKIP {persona_dir}/{dest_name} (exists)")
    else:
        print(f"  WARN source not found: {src_name} (run generate_avatars.py first)")

print("Avatar directory structure ready.")
PYEOF

# ── Final verification ────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo "  GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "  numpy : $(python -c 'import numpy; print(numpy.__version__)')"
echo "  torch : $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
echo "  TTS   : $(python -c 'import TTS; print(TTS.__version__)' 2>/dev/null || echo 'check manually')"
echo ""
echo "Next:"
echo "  source venv/bin/activate"
echo "  python scripts/generate_avatars.py"
echo "  python app.py"
echo "  Open: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_EC2_IP')"
