"""
musetalk_engine.py — Full lip sync pipeline with SadTalker + MuseTalk + GFPGAN

Pipeline order (most to least important for human authenticity):
  1. SadTalker   — head motion + eye blink + micro-expressions (audio-driven)
  2. MuseTalk    — precise lip sync on SadTalker animated output
  3. GFPGAN v1.4 — per-frame face quality enhancement (artifact removal)

Why this order:
  SadTalker makes the face move naturally (nods, blinks, slight expressions).
  MuseTalk then refines ONLY the lip region on that already-moving face.
  GFPGAN removes compression artifacts and improves skin quality per-frame.

Fallbacks:
  If SadTalker not installed → skip to MuseTalk directly
  If MuseTalk not installed → static loop with audio
  If GFPGAN fails → use MuseTalk output as-is

Install: setup.sh handles both SadTalker and MuseTalk
"""
import logging
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Optional

log = logging.getLogger("musetalk_engine")

ROOT         = Path(__file__).parent.parent.resolve()
MUSETALK_DIR = ROOT / "models" / "MuseTalk"
MUSETALK_WEIGHTS = MUSETALK_DIR / "models"


# ── Availability checks ───────────────────────────────────────────────────────

def is_musetalk_available() -> bool:
    if not MUSETALK_DIR.exists():
        return False
    if not (MUSETALK_DIR / "scripts" / "inference.py").exists():
        return False
    if not MUSETALK_WEIGHTS.exists() or not list(MUSETALK_WEIGHTS.glob("**/*.pth")):
        return False
    return True


def is_sadtalker_available() -> bool:
    from pipeline.sadtalker_engine import is_available
    return is_available()


def is_gfpgan_available() -> bool:
    try:
        import gfpgan
        return True
    except ImportError:
        return False


def get_pipeline_status() -> dict:
    return {
        "sadtalker": is_sadtalker_available(),
        "musetalk":  is_musetalk_available(),
        "gfpgan":    is_gfpgan_available(),
    }


# ── MuseTalk lip sync ─────────────────────────────────────────────────────────

def run_musetalk(
    avatar_video: Path,    # can be static image OR animated video (SadTalker output)
    audio_path: Path,
    out_path: Path,
) -> bool:
    """
    Run MuseTalk lip sync on avatar_video.
    If avatar_video is a PNG image, MuseTalk handles it.
    If it's a video from SadTalker, MuseTalk refines only the lip region.
    """
    if not is_musetalk_available():
        log.warning("MuseTalk not installed")
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Build inference config YAML
        config = {
            "video_path": str(avatar_video),
            "audio_path": str(audio_path),
            "result_dir": str(tmp / "result"),
            "use_float16": True,
            "version": "v15",
            "gpu_id": 0,
        }
        config_path = tmp / "inference_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(MUSETALK_DIR)

        cmd = [
            "python", "-m", "scripts.inference",
            "--inference_config", str(config_path),
        ]

        log.info(f"Running MuseTalk | input={avatar_video.name}")
        r = subprocess.run(
            cmd,
            cwd=str(MUSETALK_DIR),
            env=env,
            capture_output=True,
            timeout=3600,
        )

        if r.returncode != 0:
            log.error(f"MuseTalk failed:\n{r.stderr.decode()[-600:]}")
            return False

        # Find output
        result_dir = tmp / "result"
        outputs    = list(result_dir.rglob("*.mp4")) if result_dir.exists() else []
        if not outputs:
            log.error("MuseTalk produced no output")
            return False

        outputs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Re-encode at high quality
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-i", str(outputs[0]),
             "-c:v", "libx264", "-preset", "fast", "-crf", "15",
             "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart",
             str(out_path)],
            capture_output=True, timeout=300,
        )
        if r2.returncode != 0:
            import shutil; shutil.copy2(str(outputs[0]), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"MuseTalk OK → {out_path.name}")
    return ok


# ── GFPGAN face enhancement ───────────────────────────────────────────────────

def enhance_faces(
    video_path: Path,
    out_path: Path,
    upscale: int = 1,    # 1 = no upscale, 2 = 2x (slower)
) -> bool:
    """
    Apply GFPGAN v1.4 face enhancement frame-by-frame.
    Removes compression artifacts, improves skin quality, sharpens eyes.
    """
    if not is_gfpgan_available():
        log.warning("GFPGAN not installed — skipping face enhancement")
        import shutil; shutil.copy2(str(video_path), str(out_path))
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        frames_dir    = tmp / "frames"
        enhanced_dir  = tmp / "enhanced"
        frames_dir.mkdir()
        enhanced_dir.mkdir()

        # Extract frames
        log.info(f"Extracting frames from {video_path.name}...")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             str(frames_dir / "frame_%06d.png")],
            capture_output=True, timeout=600
        )
        if r.returncode != 0:
            log.error("Frame extraction failed")
            return False

        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            log.error("No frames extracted")
            return False

        log.info(f"Enhancing {len(frames)} frames with GFPGAN...")
        try:
            import cv2
            import numpy as np
            from gfpgan import GFPGANer

            enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                upscale=upscale,
                arch="clean",
                channel_multiplier=2,
            )

            for i, frame_path in enumerate(frames):
                if i % 50 == 0:
                    log.info(f"  Enhancing frame {i}/{len(frames)}")
                img = cv2.imread(str(frame_path))
                if img is None:
                    import shutil; shutil.copy2(str(frame_path), str(enhanced_dir / frame_path.name))
                    continue
                _, _, restored = enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
                out_frame = enhanced_dir / frame_path.name
                cv2.imwrite(str(out_frame), restored)

        except Exception as e:
            log.error(f"GFPGAN enhancement failed: {e}")
            import shutil
            for f in frames:
                shutil.copy2(str(f), str(enhanced_dir / f.name))

        # Get original FPS
        fps_r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        fps_str = fps_r.stdout.strip().split("\n")[0]
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        except Exception:
            fps = 25.0

        # Re-assemble enhanced frames + original audio
        r2 = subprocess.run(
            ["ffmpeg", "-y",
             "-framerate", str(fps),
             "-i", str(enhanced_dir / "frame_%06d.png"),
             "-i", str(video_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "15",
             "-c:a", "copy",
             "-map", "0:v", "-map", "1:a",
             "-shortest",
             str(out_path)],
            capture_output=True, timeout=600
        )
        if r2.returncode != 0:
            log.error(f"Re-assembly failed: {r2.stderr.decode()[-300:]}")
            import shutil; shutil.copy2(str(video_path), str(out_path))
            return True

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"GFPGAN enhancement OK → {out_path.name}")
    return ok


# ── Static video fallback ─────────────────────────────────────────────────────

def make_static_video(avatar_image: Path, audio_path: Path, out_path: Path) -> bool:
    """Fallback: loop static avatar image for audio duration."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.warning("No lip sync available — using static avatar loop")

    r = subprocess.run(
        ["ffmpeg", "-y",
         "-loop", "1", "-i", str(avatar_image),
         "-i", str(audio_path),
         "-c:v", "libx264", "-tune", "stillimage", "-preset", "fast",
         "-c:a", "aac", "-b:a", "192k",
         "-shortest",
         str(out_path)],
        capture_output=True, timeout=600,
    )
    ok = r.returncode == 0 and out_path.exists()
    if ok:
        log.info(f"Static video → {out_path.name}")
    return ok


# ── Master pipeline ───────────────────────────────────────────────────────────

def generate_lipsync(
    source_image: Path,
    audio_path: Path,
    out_path: Path,
    use_gfpgan: bool = True,
    emotion: str = "professional",
) -> bool:
    """
    Full lip sync pipeline:
      source_image + audio
        → SadTalker (head motion + eye blink + expressions) [if available]
        → MuseTalk (precise lip sync) [if available]
        → GFPGAN (face quality) [if use_gfpgan and available]
        → final video

    Fallback chain ensures video is always produced.
    """
    out_path    = Path(out_path)
    source_image = Path(source_image)
    audio_path  = Path(audio_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    status = get_pipeline_status()
    log.info(f"Lip sync pipeline | SadTalker={status['sadtalker']} MuseTalk={status['musetalk']} GFPGAN={status['gfpgan']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Step 1: SadTalker (head motion + expressions) ──────────────────
        sadtalker_out = tmp / "sadtalker.mp4"
        sadtalker_ok  = False

        if status["sadtalker"]:
            from pipeline.sadtalker_engine import animate_with_sadtalker
            sadtalker_ok = animate_with_sadtalker(
                source_image=source_image,
                audio_path=audio_path,
                out_path=sadtalker_out,
                emotion=emotion,
            )

        # Input for MuseTalk: SadTalker output OR original image
        musetalk_input = sadtalker_out if sadtalker_ok else source_image

        # ── Step 2: MuseTalk (lip sync) ────────────────────────────────────
        musetalk_out = tmp / "musetalk.mp4"
        musetalk_ok  = False

        if status["musetalk"]:
            musetalk_ok = run_musetalk(
                avatar_video=musetalk_input,
                audio_path=audio_path,
                out_path=musetalk_out,
            )

        # Determine what we have after lip sync steps
        if musetalk_ok:
            lipsync_result = musetalk_out
        elif sadtalker_ok:
            lipsync_result = sadtalker_out
        else:
            # Fallback: static video
            return make_static_video(source_image, audio_path, out_path)

        # ── Step 3: GFPGAN face enhancement ────────────────────────────────
        if use_gfpgan and status["gfpgan"]:
            gfpgan_out = tmp / "gfpgan.mp4"
            enhance_ok = enhance_faces(lipsync_result, gfpgan_out)
            if enhance_ok:
                lipsync_result = gfpgan_out

        # Copy final result to out_path
        import shutil
        shutil.copy2(str(lipsync_result), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"Lip sync pipeline complete → {out_path.name}")
    return ok
