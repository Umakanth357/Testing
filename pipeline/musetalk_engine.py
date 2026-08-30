"""
musetalk_engine.py — Complete lip sync pipeline

NEW pipeline (v4.0 — motion library based):

  source_image + audio
    → LivePortrait (motion from real Telugu presenter clips)  ← THE GAME CHANGER
    → MuseTalk (precise lip sync on animated face)
    → GFPGAN (face quality enhancement)
    → final video

vs old pipeline (v3.0):
  source_image + audio → SadTalker (estimated motion) → MuseTalk → GFPGAN

Why the new pipeline is categorically better:
  SadTalker ESTIMATES motion from audio energy.
  LivePortrait COPIES motion from a real Telugu presenter.
  The difference is visible — one looks like an animation, the other like a person.

Fallback chain:
  LivePortrait → MuseTalk → GFPGAN     (full stack — best quality)
  LivePortrait → MuseTalk              (if GFPGAN unavailable)
  LivePortrait only                    (if MuseTalk unavailable)
  SadTalker → MuseTalk → GFPGAN       (if motion library empty)
  Static loop                          (if nothing works)
"""
import logging
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Optional

log = logging.getLogger("musetalk_engine")

ROOT             = Path(__file__).parent.parent.resolve()
MUSETALK_DIR     = ROOT / "models" / "MuseTalk"
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


def is_liveportrait_available() -> bool:
    from pipeline.liveportrait_engine import is_available
    return is_available()


def is_motion_library_ready() -> bool:
    from pipeline.motion_library import get_library_status
    status = get_library_status()
    return status.get("partial", False)   # partial = at least some clips exist


def is_gfpgan_available() -> bool:
    try:
        import gfpgan; return True
    except ImportError:
        return False


def get_pipeline_status() -> dict:
    return {
        "liveportrait":   is_liveportrait_available(),
        "motion_library": is_motion_library_ready(),
        "musetalk":       is_musetalk_available(),
        "gfpgan":         is_gfpgan_available(),
    }


# ── MuseTalk lip sync ─────────────────────────────────────────────────────────

def run_musetalk(
    avatar_video: Path,
    audio_path: Path,
    out_path: Path,
) -> bool:
    """Run MuseTalk lip sync. Input can be image or video."""
    if not is_musetalk_available():
        log.warning("MuseTalk not installed")
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        config = {
            "video_path":  str(avatar_video),
            "audio_path":  str(audio_path),
            "result_dir":  str(tmp / "result"),
            "use_float16": True,
            "version":     "v15",
            "gpu_id":      0,
        }
        config_path = tmp / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(MUSETALK_DIR)

        cmd = [
            "python", "-m", "scripts.inference",
            "--inference_config", str(config_path),
        ]

        log.info(f"Running MuseTalk | {avatar_video.name}")
        r = subprocess.run(
            cmd, cwd=str(MUSETALK_DIR), env=env,
            capture_output=True, timeout=3600,
        )

        if r.returncode != 0:
            log.error(f"MuseTalk failed:\n{r.stderr.decode()[-600:]}")
            return False

        result_dir = tmp / "result"
        outputs    = list(result_dir.rglob("*.mp4")) if result_dir.exists() else []
        if not outputs:
            log.error("MuseTalk: no output file")
            return False

        outputs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-i", str(outputs[0]),
             "-c:v", "libx264", "-preset", "fast", "-crf", "15",
             "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", str(out_path)],
            capture_output=True, timeout=300,
        )
        if r2.returncode != 0:
            import shutil; shutil.copy2(str(outputs[0]), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"MuseTalk OK → {out_path.name}")
    return ok


# ── GFPGAN enhancement ────────────────────────────────────────────────────────

def enhance_faces(video_path: Path, out_path: Path, upscale: int = 1) -> bool:
    """GFPGAN v1.4 per-frame face quality enhancement."""
    if not is_gfpgan_available():
        import shutil; shutil.copy2(str(video_path), str(out_path))
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp          = Path(tmpdir)
        frames_dir   = tmp / "frames"
        enhanced_dir = tmp / "enhanced"
        frames_dir.mkdir(); enhanced_dir.mkdir()

        # Extract frames
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             str(frames_dir / "frame_%06d.png")],
            capture_output=True, timeout=600
        )
        if r.returncode != 0:
            import shutil; shutil.copy2(str(video_path), str(out_path)); return True

        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            import shutil; shutil.copy2(str(video_path), str(out_path)); return True

        log.info(f"GFPGAN: enhancing {len(frames)} frames...")
        try:
            import cv2
            from gfpgan import GFPGANer

            enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
                upscale=upscale, arch="clean", channel_multiplier=2,
            )
            for i, fp in enumerate(frames):
                if i % 50 == 0:
                    log.info(f"  frame {i}/{len(frames)}")
                img = cv2.imread(str(fp))
                if img is None:
                    import shutil; shutil.copy2(str(fp), str(enhanced_dir / fp.name))
                    continue
                _, _, restored = enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                cv2.imwrite(str(enhanced_dir / fp.name), restored)
        except Exception as e:
            log.warning(f"GFPGAN error: {e} — copying frames as-is")
            import shutil
            for f in frames:
                shutil.copy2(str(f), str(enhanced_dir / f.name))

        # Get FPS
        fps_r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        try:
            n, d = fps_r.stdout.strip().split("\n")[0].split("/")
            fps = float(n) / float(d)
        except Exception:
            fps = 25.0

        r2 = subprocess.run(
            ["ffmpeg", "-y",
             "-framerate", str(fps),
             "-i", str(enhanced_dir / "frame_%06d.png"),
             "-i", str(video_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "15",
             "-c:a", "copy", "-map", "0:v", "-map", "1:a", "-shortest",
             str(out_path)],
            capture_output=True, timeout=600
        )
        if r2.returncode != 0:
            import shutil; shutil.copy2(str(video_path), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"GFPGAN OK → {out_path.name}")
    return ok


# ── Static fallback ───────────────────────────────────────────────────────────

def make_static_video(avatar_image: Path, audio_path: Path, out_path: Path) -> bool:
    """Last resort: loop static image for audio duration."""
    log.warning("Using static avatar loop — install LivePortrait + build motion library")
    r = subprocess.run(
        ["ffmpeg", "-y",
         "-loop", "1", "-i", str(avatar_image),
         "-i", str(audio_path),
         "-c:v", "libx264", "-tune", "stillimage", "-preset", "fast",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)],
        capture_output=True, timeout=600,
    )
    return r.returncode == 0 and out_path.exists()


# ── Master pipeline ───────────────────────────────────────────────────────────

def generate_lipsync(
    source_image: Path,
    audio_path: Path,
    out_path: Path,
    use_gfpgan: bool = True,
    emotion: str = "professional",
) -> bool:
    """
    Full lip sync pipeline — v4.0:

      source_image + audio
        → LivePortrait (real human motion from library)  [PRIORITY 1]
        → MuseTalk (precise lip sync)
        → GFPGAN (face quality)

      Fallbacks:
        → SadTalker (if no motion library)
        → MuseTalk on image directly
        → Static loop
    """
    out_path     = Path(out_path)
    source_image = Path(source_image)
    audio_path   = Path(audio_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    status = get_pipeline_status()
    log.info(
        f"Pipeline v4.0 | "
        f"LivePortrait={status['liveportrait']} "
        f"MotionLib={status['motion_library']} "
        f"MuseTalk={status['musetalk']} "
        f"GFPGAN={status['gfpgan']}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Step 1: Motion animation ───────────────────────────────────────
        animated_video = tmp / "animated.mp4"
        animated_ok    = False

        if status["liveportrait"] and status["motion_library"]:
            # PRIMARY: LivePortrait with real human motion from library
            from pipeline.liveportrait_engine import animate_avatar
            animated_ok = animate_avatar(
                source_image=source_image,
                audio_path=audio_path,
                out_path=animated_video,
                emotion=emotion,
            )
            if animated_ok:
                log.info("Motion: LivePortrait (real human motion) ✅")

        if not animated_ok:
            # FALLBACK: SadTalker (estimated motion)
            try:
                from pipeline.sadtalker_engine import animate_with_sadtalker, is_available
                if is_available():
                    animated_ok = animate_with_sadtalker(
                        source_image=source_image,
                        audio_path=audio_path,
                        out_path=animated_video,
                        emotion=emotion,
                    )
                    if animated_ok:
                        log.info("Motion: SadTalker (estimated) ✅")
            except Exception as e:
                log.warning(f"SadTalker failed: {e}")

        # Input for MuseTalk
        musetalk_input = animated_video if animated_ok else source_image

        # ── Step 2: MuseTalk lip sync ──────────────────────────────────────
        lipsync_video = tmp / "lipsync.mp4"
        lipsync_ok    = False

        if status["musetalk"]:
            lipsync_ok = run_musetalk(musetalk_input, audio_path, lipsync_video)
            if lipsync_ok:
                log.info("Lip sync: MuseTalk ✅")

        if not lipsync_ok:
            if animated_ok:
                # Have animated video but no lip sync — add audio and use it
                r = subprocess.run(
                    ["ffmpeg", "-y",
                     "-i", str(animated_video), "-i", str(audio_path),
                     "-c:v", "copy", "-c:a", "aac", "-shortest", str(lipsync_video)],
                    capture_output=True, timeout=300
                )
                lipsync_ok = r.returncode == 0
                if lipsync_ok:
                    log.info("Lip sync: skipped (no MuseTalk) — using animated video ⚠️")
            else:
                # Nothing works — static fallback
                return make_static_video(source_image, audio_path, out_path)

        lipsync_result = lipsync_video if lipsync_ok else animated_video

        # ── Step 3: GFPGAN face quality ────────────────────────────────────
        if use_gfpgan and status["gfpgan"]:
            gfpgan_out = tmp / "gfpgan.mp4"
            if enhance_faces(lipsync_result, gfpgan_out):
                lipsync_result = gfpgan_out
                log.info("Face quality: GFPGAN ✅")

        # Copy to output
        import shutil
        shutil.copy2(str(lipsync_result), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"Pipeline complete → {out_path.name}")
    return ok
