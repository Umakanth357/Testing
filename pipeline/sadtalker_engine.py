"""
sadtalker_engine.py — Audio-driven head motion, eye blink, and expression

SadTalker (CVPR 2023, OpenTalker) drives realistic head pose + eye blink +
micro-expressions from audio. This runs BEFORE MuseTalk in the pipeline:

  source_image + audio → SadTalker → animated_video (head + blink + expression)
  animated_video + audio → MuseTalk → lipsync_video (precise lip sync)
  lipsync_video → GFPGAN → final_video (face quality)

Why this order:
  SadTalker generates the base motion. MuseTalk then refines only the lip region
  on top of that motion. Together they produce:
    ✅ Natural head nods and tilts
    ✅ Regular eye blinking (irregular timing, like humans)
    ✅ Micro-expression changes driven by audio energy
    ✅ Precise lip sync (MuseTalk layer)

Install: models/SadTalker/ (setup.sh handles this)
Weights: ~700MB (much lighter than MuseTalk)
Speed:   ~0.8× real-time on T4 (8 min for 10-min video)
License: MIT
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("sadtalker_engine")

ROOT         = Path(__file__).parent.parent.resolve()
SADTALKER_DIR = ROOT / "models" / "SadTalker"
CHECKPOINTS   = SADTALKER_DIR / "checkpoints"
GFPGAN_DIR    = SADTALKER_DIR / "gfpgan" / "weights"


# ── Availability check ────────────────────────────────────────────────────────

def is_available() -> bool:
    """Check if SadTalker is installed with weights."""
    if not SADTALKER_DIR.exists():
        return False
    if not (SADTALKER_DIR / "inference.py").exists():
        return False
    if not CHECKPOINTS.exists() or not list(CHECKPOINTS.glob("*.safetensors")):
        return False
    return True


def install_sadtalker() -> bool:
    """Clone SadTalker and download model weights. Called by setup.sh."""
    SADTALKER_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not SADTALKER_DIR.exists():
        log.info("Cloning SadTalker...")
        r = subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/OpenTalker/SadTalker.git",
             str(SADTALKER_DIR)],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            log.error(f"SadTalker clone failed: {r.stderr.decode()[:300]}")
            return False

    # Install requirements
    req = SADTALKER_DIR / "requirements.txt"
    if req.exists():
        subprocess.run(["pip", "install", "-r", str(req), "--quiet"], timeout=300)

    # Download weights via HuggingFace
    log.info("Downloading SadTalker weights (~700MB)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "vinthony/SadTalker",
            local_dir=str(SADTALKER_DIR / "checkpoints"),
            ignore_patterns=["*.git*"],
        )
        # Also need GFPGAN weights for face enhancement in SadTalker
        os.makedirs(str(GFPGAN_DIR), exist_ok=True)
        snapshot_download(
            "TencentARC/GFPGAN",
            local_dir=str(GFPGAN_DIR),
            ignore_patterns=["*.git*", "experiments/*"],
        )
        log.info("SadTalker weights downloaded")
        return True
    except Exception as e:
        log.error(f"SadTalker weight download failed: {e}")
        return False


# ── Core inference ────────────────────────────────────────────────────────────

def run_sadtalker(
    source_image: Path,
    audio_path: Path,
    out_path: Path,
    expression_scale: float = 1.2,   # >1.0 = more expressive, <1.0 = subtle
    pose_style: int = 0,              # 0-45, randomise slightly per video for variety
    enhancer: str = "gfpgan",         # 'gfpgan' | 'RestoreFormer' | None
    preprocess: str = "crop",         # 'crop' | 'extcrop' | 'resize' | 'full'
    size: int = 256,                  # 256 or 512 (512 = higher quality, more VRAM)
) -> bool:
    """
    Run SadTalker audio-driven animation.
    source_image: portrait PNG (front-facing, well-lit)
    audio_path:   WAV file (44.1kHz mono)
    out_path:     output MP4 path
    Returns True if animation was generated successfully.
    """
    if not is_available():
        log.warning("SadTalker not installed")
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        result_dir = Path(tmpdir) / "results"
        result_dir.mkdir()

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SADTALKER_DIR)

        cmd = [
            "python", "inference.py",
            "--driven_audio", str(audio_path),
            "--source_image", str(source_image),
            "--result_dir", str(result_dir),
            "--expression_scale", str(expression_scale),
            "--pose_style", str(pose_style),
            "--preprocess", preprocess,
            "--size", str(size),
            "--still",           # reduces excessive head motion (more professional)
            "--face3dvis",       # better face reconstruction
        ]

        # Add enhancer if available
        if enhancer and GFPGAN_DIR.exists():
            cmd += ["--enhancer", enhancer]

        log.info(f"Running SadTalker | image={source_image.name} audio={audio_path.name}")
        r = subprocess.run(
            cmd,
            cwd=str(SADTALKER_DIR),
            env=env,
            capture_output=True,
            timeout=1800,   # 30 min max
        )

        if r.returncode != 0:
            log.error(f"SadTalker failed:\n{r.stderr.decode()[-600:]}")
            return False

        # Find output — SadTalker saves as <audio_name>_<expression>_still.mp4
        outputs = list(result_dir.rglob("*.mp4"))
        if not outputs:
            log.error("SadTalker ran but no output video found")
            return False

        outputs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        raw_out = outputs[0]

        # Re-encode for pipeline consistency
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_out),
             "-c:v", "libx264", "-preset", "fast", "-crf", "16",
             "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart",
             str(out_path)],
            capture_output=True, timeout=300,
        )

        if r2.returncode != 0:
            import shutil; shutil.copy2(str(raw_out), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"SadTalker OK → {out_path} ({out_path.stat().st_size//1024}KB)")
    return ok


# ── Public API ────────────────────────────────────────────────────────────────

def animate_with_sadtalker(
    source_image: Path,
    audio_path: Path,
    out_path: Path,
    emotion: str = "professional",   # controls expression_scale
) -> bool:
    """
    Wrapper: runs SadTalker if available, returns False if not installed.
    emotion: 'professional' | 'energetic' | 'warm' | 'serious'
    """
    # Map emotion to expression_scale
    EMOTION_SCALE = {
        "professional": 1.0,
        "energetic":    1.5,
        "warm":         1.2,
        "serious":      0.8,
        "excited":      1.6,
        "calm":         0.7,
    }
    scale = EMOTION_SCALE.get(emotion, 1.0)

    if not is_available():
        log.warning("SadTalker unavailable — head motion skipped")
        return False

    return run_sadtalker(
        source_image=source_image,
        audio_path=audio_path,
        out_path=out_path,
        expression_scale=scale,
        enhancer="gfpgan" if GFPGAN_DIR.exists() else None,
    )
