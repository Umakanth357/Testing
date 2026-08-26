#!/usr/bin/env bash
# Avatar Studio — EC2 Setup Script
# Target: Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04), Tesla T4, CUDA 13.x
# Run as: ./setup.sh 2>&1 | tee setup.log
set -e

echo "=== Avatar Studio Setup ==="
echo "Platform: $(uname -m)  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'unknown')"

# ── 0. Load .env ──────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example → .env and set HF_TOKEN."
    exit 1
fi
source .env
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set in .env"
    exit 1
fi

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/9] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs curl wget unzip \
    python3-pip python3-venv \
    libsndfile1 libsndfile1-dev \
    libgl1-mesa-glx libglib2.0-0

git lfs install

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo "[2/9] Setting up Python venv..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and wheel
pip install --upgrade pip wheel setuptools

# ── 3. Pin numpy FIRST — critical to avoid conflicts ─────────────────────────
# chatterbox-tts, basicsr, realesrgan all require numpy<2.
# Must install before anything else pulls in a newer numpy.
echo "[3/9] Pinning numpy==1.26.4 (required by basicsr/realesrgan/chatterbox-tts)..."
pip install "numpy==1.26.4"

# ── 4. PyTorch — skip if AMI already has it ──────────────────────────────────
echo "[4/9] Checking PyTorch..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "    PyTorch with CUDA already available — skipping reinstall."
    echo "    Version: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "    Installing PyTorch 2.1 + CUDA 12.1..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121
fi

# ── 5. Core ML packages ───────────────────────────────────────────────────────
echo "[5/9] Installing core ML packages..."
pip install \
    "transformers>=4.40.0" \
    "diffusers>=0.27.0" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.0" \
    "peft>=0.10.0" \
    "huggingface-hub>=0.22.0" \
    "sentence-transformers>=2.7.0"

# ── 6. Image/video packages ───────────────────────────────────────────────────
echo "[6/9] Installing image/video packages..."
pip install \
    "Pillow>=10.0.0" \
    "opencv-python>=4.9.0" \
    "facexlib>=0.3.0" \
    "ffmpeg-python>=0.2.0" \
    "imageio>=2.33.0" \
    "imageio-ffmpeg>=0.4.9"

# Install gfpgan/basicsr/realesrgan with --no-deps to prevent numpy upgrade
echo "    Installing gfpgan (--no-deps to protect numpy pin)..."
pip install gfpgan --no-deps
pip install basicsr --no-deps
pip install realesrgan --no-deps
# Install their missing deps manually (excluding numpy)
pip install "scipy>=1.11.0" "tb-nightly" 2>/dev/null || true

# ── 7. TTS packages ───────────────────────────────────────────────────────────
echo "[7/9] Installing TTS packages..."
# chatterbox-tts: install with --no-deps then add its deps manually
pip install chatterbox-tts --no-deps
pip install \
    "TTS>=0.22.0" \
    "gtts>=2.5.0" \
    "soundfile>=0.12.0" \
    "sounddevice>=0.4.6" \
    "librosa>=0.10.0" \
    "pedalboard>=0.9.0" \
    "pydub>=0.25.0"

# Verify numpy hasn't been upgraded
NUMPY_VER=$(python -c "import numpy; print(numpy.__version__)")
echo "    numpy version after TTS install: $NUMPY_VER"
if [[ "$NUMPY_VER" != "1.26"* ]]; then
    echo "    WARNING: numpy was upgraded to $NUMPY_VER — re-pinning..."
    pip install "numpy==1.26.4" --force-reinstall
fi

# ── 8. Script engine and UI packages ─────────────────────────────────────────
echo "[8/9] Installing script/UI packages..."
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
    "gradio==4.44.0"

# ── 9. Clone animation models ─────────────────────────────────────────────────
echo "[9/9] Cloning EchoMimicV2 and LatentSync..."

mkdir -p models

if [ ! -d "models/EchoMimicV2" ]; then
    git clone https://github.com/antgroup/echomimic_v2.git models/EchoMimicV2
    echo "    EchoMimicV2 cloned."
else
    echo "    EchoMimicV2 already exists — skipping."
fi

if [ ! -d "models/LatentSync" ]; then
    git clone https://github.com/bytedance/LatentSync.git models/LatentSync
    echo "    LatentSync cloned."
else
    echo "    LatentSync already exists — skipping."
fi

# ── Download model weights via HF hub ────────────────────────────────────────
echo "Downloading model weights from HuggingFace..."
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download, hf_hub_download

token = os.environ.get("HF_TOKEN")
os.makedirs("models/weights", exist_ok=True)

# EchoMimicV2 weights
try:
    snapshot_download("BadToBest/EchoMimicV2",
                      local_dir="models/weights/EchoMimicV2",
                      token=token, ignore_patterns=["*.bin"])
    print("EchoMimicV2 weights downloaded.")
except Exception as e:
    print(f"WARNING: EchoMimicV2 weights failed: {e}")

# LatentSync weights
try:
    snapshot_download("ByteDance/LatentSync-1.5",
                      local_dir="models/weights/LatentSync",
                      token=token)
    print("LatentSync weights downloaded.")
except Exception as e:
    print(f"WARNING: LatentSync weights failed: {e}")

print("Model download complete.")
PYEOF

# ── Ollama + Gemma3 ───────────────────────────────────────────────────────────
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Starting Ollama service..."
ollama serve &>/dev/null &
sleep 5
ollama pull gemma3:4b
echo "Gemma3:4b model ready."

# ── Nginx reverse proxy ───────────────────────────────────────────────────────
echo "Configuring Nginx..."
sudo apt-get install -y nginx
sudo tee /etc/nginx/sites-available/avatar-studio > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/avatar-studio /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
echo "Nginx configured: port 80 → 127.0.0.1:7860"

# ── Final checks ──────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "torch: $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"
echo ""
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python scripts/generate_avatars.py   # generate Navya + Arjun faces"
echo "  3. python app.py                         # start the app"
echo "  4. Open http://YOUR_EC2_PUBLIC_IP in browser"
