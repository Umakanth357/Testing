"""
Animation Engine — Half-body animation from audio. EchoMimicV2 is primary.

Priority chain:
  EchoMimicV2 (upper body + hands) → CyberHost (full body) → SadTalker → Static fallback

All outputs are MP4 videos ready for lip sync stage.
"""
import logging
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch

from config import ECHOMIMIC_DIR, CYBERHOST_DIR, LATENTSYNC_DIR, DEVICE, OUTPUT_FPS

log = logging.getLogger("animation_engine")

TIMEOUT_SEC = 600   # 10 min max per animation job


# ── Public API ────────────────────────────────────────────────────────────────

def animate(
    avatar_image: Path,
    audio_path: Path,
    out_path: Path,
    pose: str = "half_body",         # standing | half_body | sitting_desk
    width: int = 768,
    height: int = 768,
) -> bool:
    """
    Animate avatar image driven by audio. Half-body minimum.
    Returns True if video was created successfully.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not avatar_image.exists():
        log.error(f"Avatar image not found: {avatar_image}")
        return False
    if not audio_path.exists():
        log.error(f"Audio not found: {audio_path}")
        return False

    _free_gpu_memory()

    # Try each engine in order
    success = False

    success = _run_echomimic(avatar_image, audio_path, out_path, pose, width, height)
    if success:
        log.info("EchoMimicV2 succeeded")
        return True

    log.warning("EchoMimicV2 failed — trying CyberHost")
    success = _run_cyberhost(avatar_image, audio_path, out_path)
    if success:
        log.info("CyberHost succeeded")
        return True

    log.warning("CyberHost failed — trying SadTalker")
    success = _run_sadtalker(avatar_image, audio_path, out_path)
    if success:
        log.info("SadTalker succeeded")
        return True

    log.warning("SadTalker failed — using static fallback")
    success = _run_static_fallback(avatar_image, audio_path, out_path)
    return success


# ── EchoMimicV2 (PRIMARY — upper body + hands) ───────────────────────────────

def _run_echomimic(avatar_image: Path, audio_path: Path, out_path: Path,
                   pose: str, width: int, height: int) -> bool:
    """
    EchoMimicV2: audio-driven upper body animation (CVPR 2025, Apache 2.0).
    GitHub: antgroup/echomimic_v2
    """
    if not ECHOMIMIC_DIR.exists():
        log.warning("EchoMimicV2 not found — run setup.sh first")
        return False

    inference_script = ECHOMIMIC_DIR / "inference.py"
    if not inference_script.exists():
        inference_script = ECHOMIMIC_DIR / "scripts" / "inference.py"
    if not inference_script.exists():
        log.warning("EchoMimicV2 inference.py not found")
        return False

    # EchoMimicV2 config
    pose_config = {
        "standing":     {"pose_weight": 0.8, "face_weight": 1.0, "lip_weight": 1.0},
        "half_body":    {"pose_weight": 0.7, "face_weight": 1.0, "lip_weight": 1.0},
        "sitting_desk": {"pose_weight": 0.6, "face_weight": 1.0, "lip_weight": 0.9},
    }
    cfg = pose_config.get(pose, pose_config["half_body"])

    tmp_out = out_path.parent / f"echomimic_tmp_{out_path.stem}.mp4"

    cmd = [
        "python", str(inference_script),
        "--ref_image_path",    str(avatar_image),
        "--audio_path",        str(audio_path),
        "--output_path",       str(tmp_out),
        "--width",             str(width),
        "--height",            str(height),
        "--fps",               str(OUTPUT_FPS),
        "--pose_weight",       str(cfg["pose_weight"]),
        "--face_weight",       str(cfg["face_weight"]),
        "--lip_weight",        str(cfg["lip_weight"]),
        "--device",            DEVICE,
    ]

    result = _run_subprocess(cmd, cwd=ECHOMIMIC_DIR, timeout=TIMEOUT_SEC)
    if result and tmp_out.exists() and tmp_out.stat().st_size > 10_000:
        shutil.move(str(tmp_out), str(out_path))
        return True
    return False


# ── CyberHost (FALLBACK 1 — full body) ───────────────────────────────────────

def _run_cyberhost(avatar_image: Path, audio_path: Path, out_path: Path) -> bool:
    """
    CyberHost: one-stage audio-driven full body diffusion model.
    GitHub: deepbrainai-research/cyberhost
    """
    if not CYBERHOST_DIR.exists():
        log.warning("CyberHost not found")
        return False

    inference_script = CYBERHOST_DIR / "inference.py"
    if not inference_script.exists():
        log.warning("CyberHost inference.py not found")
        return False

    tmp_out = out_path.parent / f"cyberhost_tmp_{out_path.stem}.mp4"

    cmd = [
        "python", str(inference_script),
        "--source_image", str(avatar_image),
        "--audio_path",   str(audio_path),
        "--output",       str(tmp_out),
        "--device",       DEVICE,
    ]

    result = _run_subprocess(cmd, cwd=CYBERHOST_DIR, timeout=TIMEOUT_SEC)
    if result and tmp_out.exists() and tmp_out.stat().st_size > 10_000:
        shutil.move(str(tmp_out), str(out_path))
        return True
    return False


# ── SadTalker (FALLBACK 2 — head + shoulders) ────────────────────────────────

def _run_sadtalker(avatar_image: Path, audio_path: Path, out_path: Path) -> bool:
    """
    SadTalker: audio-driven facial animation. Head + upper shoulders.
    GitHub: OpenTalker/SadTalker
    """
    sadtalker_dir = Path("models/SadTalker")
    if not sadtalker_dir.exists():
        log.warning("SadTalker not found")
        return False

    inference_script = sadtalker_dir / "inference.py"
    if not inference_script.exists():
        return False

    tmp_dir = Path(tempfile.mkdtemp(prefix="sadtalker_"))
    cmd = [
        "python", str(inference_script),
        "--driven_audio",  str(audio_path),
        "--source_image",  str(avatar_image),
        "--result_dir",    str(tmp_dir),
        "--still",
        "--preprocess",    "full",
        "--enhancer",      "gfpgan",
    ]

    result = _run_subprocess(cmd, cwd=sadtalker_dir, timeout=TIMEOUT_SEC)
    if result:
        # Find generated video in tmp_dir
        videos = list(tmp_dir.rglob("*.mp4"))
        if videos:
            shutil.copy(str(videos[0]), str(out_path))
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            return True
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    return False


# ── Static Fallback (LAST RESORT) ────────────────────────────────────────────

def _run_static_fallback(avatar_image: Path, audio_path: Path, out_path: Path) -> bool:
    """
    Last resort: loop static avatar image to audio duration.
    No animation — but pipeline can at least complete.
    """
    log.warning("Using static avatar fallback — no body animation")
    try:
        import subprocess
        # Get audio duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip() or "5.0")

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(avatar_image),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-t", str(duration),
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and out_path.exists()
    except Exception as e:
        log.error(f"Static fallback failed: {e}")
        return False


# ── Walking Entrance (Optional pre-clip) ──────────────────────────────────────

def generate_walk_entrance(avatar_image: Path, out_path: Path, duration_sec: float = 3.0) -> bool:
    """
    Generate a short walking entrance clip using MimicMotion.
    Falls back to pan+zoom effect if MimicMotion unavailable.
    """
    musecpose_dir = Path("models/MusePose")
    if musecpose_dir.exists():
        # Use MusePose with a reference walk pose sequence
        pose_ref = musecpose_dir / "assets" / "walk_reference.mp4"
        if pose_ref.exists():
            inference = musecpose_dir / "inference.py"
            cmd = [
                "python", str(inference),
                "--ref_image", str(avatar_image),
                "--pose_video", str(pose_ref),
                "--output", str(out_path),
                "--device", DEVICE,
            ]
            if _run_subprocess(cmd, cwd=musecpose_dir, timeout=120):
                return True

    # Fallback: zoom-in pan effect to simulate entrance
    try:
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(avatar_image),
            "-vf", f"scale=1920:1080,zoompan=z='min(zoom+0.001,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration_sec * OUTPUT_FPS)}:s=768x768,fade=t=in:st=0:d=0.5",
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and out_path.exists()
    except Exception as e:
        log.error(f"Walk entrance fallback failed: {e}")
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
            log.debug(f"Process stderr: {result.stderr[-500:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error(f"Process timed out after {timeout}s")
        return False
    except Exception as e:
        log.error(f"Subprocess error: {e}")
        return False


def _free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
