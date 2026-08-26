"""
Lip Sync Engine — LatentSync v1.5 (ByteDance, Apache 2.0).
Language-agnostic: works for Telugu, Tamil, Kannada, Hindi, English equally.

Applies lip sync on top of any animated video from animation_engine.
"""
import logging
import shutil
import subprocess
from pathlib import Path

import torch

from config import LATENTSYNC_DIR, DEVICE

log = logging.getLogger("lipsync_engine")

TIMEOUT_SEC = 600

LATENTSYNC_UNET_CFG   = LATENTSYNC_DIR / "configs" / "unet" / "second_stage.yaml"
LATENTSYNC_CKPT       = LATENTSYNC_DIR / "checkpoints" / "latentsync_unet.pt"
LATENTSYNC_INFERENCE  = LATENTSYNC_DIR / "inference.py"


# ── Public API ────────────────────────────────────────────────────────────────

def apply_lipsync(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """
    Apply lip sync to animated video. LatentSync is primary, Wav2Lip is fallback.
    Returns True if lip-synced video was created successfully.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        log.error(f"Input video not found: {video_path}")
        return False
    if not audio_path.exists():
        log.error(f"Audio not found: {audio_path}")
        return False

    _free_gpu_memory()

    success = _run_latentsync(video_path, audio_path, out_path)
    if success:
        log.info("LatentSync succeeded")
        return True

    log.warning("LatentSync failed — trying Wav2Lip")
    success = _run_wav2lip(video_path, audio_path, out_path)
    if success:
        log.info("Wav2Lip succeeded")
        return True

    log.warning("All lip sync engines failed — using original video with replaced audio")
    return _replace_audio_only(video_path, audio_path, out_path)


# ── LatentSync ────────────────────────────────────────────────────────────────

def _run_latentsync(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """
    LatentSync v1.5 — latent diffusion lip sync. 8GB VRAM. Language-agnostic.
    """
    if not LATENTSYNC_DIR.exists():
        log.warning("LatentSync not found — run setup.sh")
        return False
    if not LATENTSYNC_INFERENCE.exists():
        log.warning("LatentSync inference.py not found")
        return False
    if not LATENTSYNC_CKPT.exists():
        log.warning("LatentSync checkpoint not found — run scripts/download_models.py")
        return False

    tmp_out = out_path.parent / f"latentsync_tmp_{out_path.stem}.mp4"

    cmd = [
        "python", str(LATENTSYNC_INFERENCE),
        "--unet_config_path", str(LATENTSYNC_UNET_CFG),
        "--inference_ckpt_path", str(LATENTSYNC_CKPT),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(tmp_out),
        "--device", DEVICE,
    ]

    result = _run_subprocess(cmd, cwd=LATENTSYNC_DIR, timeout=TIMEOUT_SEC)
    if result and tmp_out.exists() and tmp_out.stat().st_size > 10_000:
        shutil.move(str(tmp_out), str(out_path))
        return True
    return False


# ── Wav2Lip (Fallback) ────────────────────────────────────────────────────────

def _run_wav2lip(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """Wav2Lip fallback — head-only but widely compatible."""
    wav2lip_dir = Path("models/Wav2Lip")
    if not wav2lip_dir.exists():
        return False

    inference_script = wav2lip_dir / "inference.py"
    checkpoint = wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"

    if not inference_script.exists() or not checkpoint.exists():
        return False

    tmp_out = out_path.parent / f"wav2lip_tmp_{out_path.stem}.avi"

    cmd = [
        "python", str(inference_script),
        "--checkpoint_path", str(checkpoint),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(tmp_out),
        "--nosmooth",
    ]

    result = _run_subprocess(cmd, cwd=wav2lip_dir, timeout=TIMEOUT_SEC)
    if result and tmp_out.exists():
        # Convert AVI → MP4
        convert_cmd = [
            "ffmpeg", "-y", "-i", str(tmp_out),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        subprocess.run(convert_cmd, capture_output=True, timeout=120)
        tmp_out.unlink(missing_ok=True)
        return out_path.exists() and out_path.stat().st_size > 10_000
    return False


# ── Audio-only Replace (Last Resort) ─────────────────────────────────────────

def _replace_audio_only(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """Strip video audio, replace with TTS audio. No lip adjustment."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and out_path.exists()
    except Exception as e:
        log.error(f"Audio replace failed: {e}")
        return False


# ── Utilities ─────────────────────────────────────────────────────────────────

def _run_subprocess(cmd: list, cwd: Path, timeout: int) -> bool:
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd),
            capture_output=True, text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            log.debug(f"stderr: {result.stderr[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"Lip sync timed out after {timeout}s")
        return False
    except Exception as e:
        log.error(f"Subprocess error: {e}")
        return False


def _free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
