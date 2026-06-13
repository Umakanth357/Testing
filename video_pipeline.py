"""
video_pipeline.py
Full avatar video generation:
  1. LivePortrait  — drives expressions + head movement from base video
  2. MuseTalk      — frame-perfect lip sync (diffusion-based)
  3. GFPGAN        — face restoration + 4x upscale
  4. ffmpeg        — compose avatar on background + captions + logo
Fallback: SadTalker if MuseTalk/LivePortrait unavailable
"""
import os
import sys
import logging
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
WEIGHTS    = ROOT / "models" / "weights"
MUSETALK   = ROOT / "models" / "MuseTalk"
LIVEPORT   = ROOT / "models" / "LivePortrait"
SADTALKER  = ROOT / "models" / "SadTalker"
AVATARS    = ROOT / "avatars"
OUTPUTS    = ROOT / "outputs"
ASSETS     = ROOT / "web" / "assets"

OUTPUTS.mkdir(exist_ok=True)


class VideoPipeline:
    """
    Generates a talking-head video from:
      - avatar_image: path to PNG/JPG of the avatar face
      - audio_path:   path to WAV audio file (from TTS engine)
      - output_path:  desired output MP4 path
      - background:   "office" | "studio" | "gradient" | path to custom image
    """

    def generate(
        self,
        avatar_image: str,
        audio_path: str,
        output_path: str,
        background: str = "office",
        add_captions: bool = True,
        logo_path: Optional[str] = None,
        job_id: str = "video"
    ) -> Optional[str]:

        logger.info(f"[{job_id}] Starting video generation pipeline")
        tmpdir = tempfile.mkdtemp(prefix=f"vidgen_{job_id}_")

        try:
            # Stage 1: LivePortrait (expressions + head motion)
            logger.info(f"[{job_id}] Stage 1: LivePortrait animation")
            animated = self._run_liveportrait(avatar_image, audio_path, tmpdir, job_id)
            if not animated:
                logger.warning(f"[{job_id}] LivePortrait failed — trying SadTalker fallback")
                animated = self._run_sadtalker(avatar_image, audio_path, tmpdir, job_id)
            if not animated:
                logger.warning(f"[{job_id}] SadTalker failed — using static avatar fallback")
                animated = self._static_avatar_fallback(avatar_image, audio_path, tmpdir, job_id)

            # Stage 2: MuseTalk (lip sync on top of LivePortrait output)
            logger.info(f"[{job_id}] Stage 2: MuseTalk lip sync")
            lip_synced = self._run_musetalk(animated, audio_path, tmpdir, job_id)
            if not lip_synced:
                logger.warning(f"[{job_id}] MuseTalk failed — using LivePortrait output directly")
                lip_synced = animated

            # Stage 3: GFPGAN face restoration + upscale
            logger.info(f"[{job_id}] Stage 3: GFPGAN face restore + upscale")
            enhanced = self._run_gfpgan(lip_synced, tmpdir, job_id)
            if not enhanced:
                logger.warning(f"[{job_id}] GFPGAN failed — using raw lip sync output")
                enhanced = lip_synced

            # Stage 4: Compose — background + avatar + captions + logo
            logger.info(f"[{job_id}] Stage 4: Compose final video")
            final = self._compose_video(
                enhanced, audio_path, background, output_path,
                add_captions, logo_path, tmpdir, job_id
            )

            if final and Path(final).exists():
                size_mb = Path(final).stat().st_size / 1_000_000
                logger.info(f"[{job_id}] ✓ Video generated: {final} ({size_mb:.1f} MB)")
                return final
            else:
                logger.error(f"[{job_id}] Final composition failed")
                return None

        except Exception as e:
            logger.error(f"[{job_id}] Pipeline error: {e}", exc_info=True)
            return None
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # ─── Stage 1: LivePortrait ───────────────────────────────────
    def _run_liveportrait(self, avatar_img: str, audio: str, tmpdir: str, job_id: str) -> Optional[str]:
        out = os.path.join(tmpdir, "liveportrait_out.mp4")
        try:
            if not LIVEPORT.exists():
                return None
            # Get audio duration for video length
            duration = self._get_audio_duration(audio)
            # LivePortrait CLI
            cmd = [
                sys.executable,
                str(LIVEPORT / "inference.py"),
                "--source_image", avatar_img,
                "--driving_video", self._get_base_driving_video(audio),
                "--output_dir", tmpdir,
                "--output_name", "liveportrait_out",
                "--weights_dir", str(WEIGHTS / "liveportrait"),
                "--no_flag_pasteback",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(LIVEPORT))
            if result.returncode == 0 and Path(out).exists():
                return out
            logger.debug(f"LivePortrait stderr: {result.stderr[-500:]}")
            return None
        except Exception as e:
            logger.debug(f"LivePortrait error: {e}")
            return None

    def _get_base_driving_video(self, audio: str) -> str:
        """
        Returns a short neutral driving video for LivePortrait.
        This provides natural head + shoulder movement.
        Falls back to generating a still-image loop.
        """
        base_vids = list(ASSETS.glob("base_driving_*.mp4")) if ASSETS.exists() else []
        if base_vids:
            return str(base_vids[0])
        # Generate a 10-second neutral face loop as driving video
        # In production: use a 10-15 sec recording of any person looking at camera
        return str(AVATARS / "base_driving.mp4") if (AVATARS / "base_driving.mp4").exists() else ""

    # ─── Stage 1b: SadTalker (fallback) ─────────────────────────
    def _run_sadtalker(self, avatar_img: str, audio: str, tmpdir: str, job_id: str) -> Optional[str]:
        out = os.path.join(tmpdir, "sadtalker_out.mp4")
        try:
            if not SADTALKER.exists():
                return None
            cmd = [
                sys.executable,
                str(SADTALKER / "inference.py"),
                "--driven_audio", audio,
                "--source_image", avatar_img,
                "--result_dir", tmpdir,
                "--enhancer", "gfpgan",
                "--preprocess", "crop",
                "--checkpoint_dir", str(WEIGHTS / "sadtalker"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(SADTALKER))
            # SadTalker puts output in a subdir
            candidates = list(Path(tmpdir).glob("**/*.mp4"))
            if candidates:
                shutil.copy(str(candidates[0]), out)
                return out
            return None
        except Exception as e:
            logger.debug(f"SadTalker error: {e}")
            return None

    # ─── Stage 1c: Static avatar fallback ───────────────────────
    def _static_avatar_fallback(self, avatar_img: str, audio: str, tmpdir: str, job_id: str) -> str:
        """Last resort: loop static avatar image for audio duration."""
        out = os.path.join(tmpdir, "static_avatar.mp4")
        duration = self._get_audio_duration(audio)
        subprocess.run([
            "ffmpeg",
            "-loop", "1", "-i", avatar_img,
            "-i", audio,
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest", "-t", str(duration),
            out, "-y", "-loglevel", "error"
        ], check=True)
        return out

    # ─── Stage 2: MuseTalk ───────────────────────────────────────
    def _run_musetalk(self, video_in: str, audio: str, tmpdir: str, job_id: str) -> Optional[str]:
        out = os.path.join(tmpdir, "musetalk_out.mp4")
        try:
            if not MUSETALK.exists():
                return None
            cmd = [
                sys.executable,
                str(MUSETALK / "scripts" / "inference.py"),
                "--video_path", video_in,
                "--audio_path", audio,
                "--result_dir", tmpdir,
                "--fps", "30",
                "--batch_size", "8",
                "--unet_model_path", str(WEIGHTS / "musetalk" / "pytorch_model.bin"),
                "--unet_config", str(WEIGHTS / "musetalk" / "musetalk.json"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(MUSETALK))
            candidates = list(Path(tmpdir).glob("**/*.mp4"))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                shutil.copy(str(latest), out)
                return out
            return None
        except Exception as e:
            logger.debug(f"MuseTalk error: {e}")
            return None

    # ─── Stage 3: GFPGAN ────────────────────────────────────────
    def _run_gfpgan(self, video_in: str, tmpdir: str, job_id: str) -> Optional[str]:
        """Extract frames → GFPGAN enhance → reassemble."""
        out = os.path.join(tmpdir, "gfpgan_out.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        enhanced_dir = os.path.join(tmpdir, "enhanced_frames")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(enhanced_dir, exist_ok=True)

        try:
            gfpgan_weight = WEIGHTS / "GFPGANv1.4.pth"
            if not gfpgan_weight.exists():
                return None

            # Extract frames
            subprocess.run([
                "ffmpeg", "-i", video_in, "-q:v", "2",
                os.path.join(frames_dir, "frame_%05d.png"),
                "-y", "-loglevel", "error"
            ], check=True)

            frame_count = len(list(Path(frames_dir).glob("*.png")))
            if frame_count == 0:
                return None

            # Run GFPGAN on all frames
            subprocess.run([
                sys.executable, "-c", f"""
import sys; sys.path.insert(0, '{str(LIVEPORT.parent)}')
from gfpgan import GFPGANer
import cv2, os, glob, numpy as np

restorer = GFPGANer(
    model_path='{str(gfpgan_weight)}',
    upscale=2,
    arch='clean',
    channel_multiplier=2
)

frames = sorted(glob.glob('{frames_dir}/frame_*.png'))
for i, fpath in enumerate(frames):
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    _, _, restored = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    out_path = '{enhanced_dir}/frame_' + f'{{i+1:05d}}.png'
    cv2.imwrite(out_path, restored)
    if (i+1) % 50 == 0:
        print(f'GFPGAN: {{i+1}}/{{len(frames)}} frames processed')
print('GFPGAN done')
"""
            ], capture_output=True, text=True, timeout=600)

            enhanced_count = len(list(Path(enhanced_dir).glob("*.png")))
            if enhanced_count == 0:
                return None

            # Get original fps
            fps = self._get_video_fps(video_in)

            # Reassemble video (no audio — will be added in compose step)
            subprocess.run([
                "ffmpeg",
                "-framerate", str(fps),
                "-i", os.path.join(enhanced_dir, "frame_%05d.png"),
                "-c:v", "libx264", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-crf", "18",
                out, "-y", "-loglevel", "error"
            ], check=True)

            return out if Path(out).exists() else None

        except Exception as e:
            logger.debug(f"GFPGAN error: {e}")
            return None

    # ─── Stage 4: Compose ────────────────────────────────────────
    def _compose_video(
        self,
        avatar_video: str,
        audio: str,
        background: str,
        output_path: str,
        add_captions: bool,
        logo_path: Optional[str],
        tmpdir: str,
        job_id: str
    ) -> Optional[str]:
        """Compose: background + avatar overlay + audio + captions + logo."""
        try:
            bg = self._resolve_background(background, tmpdir)
            # Get dimensions
            av_w, av_h = self._get_video_dims(avatar_video)
            target_w, target_h = 1280, 720

            # Avatar position: centered, lower 70% of frame
            av_display_h = int(target_h * 0.75)
            av_display_w = int(av_w * (av_display_h / av_h))
            av_x = (target_w - av_display_w) // 2
            av_y = target_h - av_display_h

            composed = os.path.join(tmpdir, "composed.mp4")

            # Build ffmpeg filter complex
            filter_parts = []
            inputs = ["-i", bg, "-i", avatar_video, "-i", audio]

            # Scale background to 1280x720
            filter_parts.append(f"[0:v]scale={target_w}:{target_h},setsar=1[bg]")
            # Scale avatar
            filter_parts.append(f"[1:v]scale={av_display_w}:{av_display_h}[av]")
            # Overlay avatar on background
            filter_parts.append(f"[bg][av]overlay={av_x}:{av_y}[composed]")

            # Add logo if provided
            final_label = "composed"
            if logo_path and Path(logo_path).exists():
                inputs += ["-i", logo_path]
                filter_parts.append(f"[3:v]scale=120:40[logo]")
                filter_parts.append(f"[{final_label}][logo]overlay=W-w-20:20[with_logo]")
                final_label = "with_logo"

            filter_complex = ";".join(filter_parts)

            cmd = [
                "ffmpeg",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", f"[{final_label}]",
                "-map", "2:a",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                composed, "-y", "-loglevel", "error"
            ]
            subprocess.run(cmd, check=True, timeout=300)

            # Add auto-captions with Whisper
            if add_captions and Path(composed).exists():
                captioned = self._add_captions(composed, audio, output_path, tmpdir)
                if captioned:
                    return captioned

            # Copy composed to output if captions skipped
            shutil.copy(composed, output_path)
            return output_path

        except Exception as e:
            logger.error(f"Compose error: {e}", exc_info=True)
            # Emergency: just mux avatar video + audio
            try:
                subprocess.run([
                    "ffmpeg", "-i", avatar_video, "-i", audio,
                    "-c:v", "copy", "-c:a", "aac", "-shortest",
                    output_path, "-y", "-loglevel", "error"
                ], check=True, timeout=120)
                return output_path
            except Exception:
                return None

    def _add_captions(self, video: str, audio: str, output: str, tmpdir: str) -> Optional[str]:
        """Auto-generate captions using Whisper and burn them in."""
        try:
            import whisper
            logger.info("Generating captions with Whisper...")
            model = whisper.load_model("base")
            result = model.transcribe(audio)

            # Build SRT
            srt_path = os.path.join(tmpdir, "captions.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(result["segments"], 1):
                    def ts(t):
                        h, m, s = int(t//3600), int((t%3600)//60), t%60
                        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
                    f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text'].strip()}\n\n")

            # Burn captions — bottom center, clean style
            subprocess.run([
                "ffmpeg", "-i", video,
                "-vf", f"subtitles={srt_path}:force_style='Fontsize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Alignment=2,MarginV=30'",
                "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                output, "-y", "-loglevel", "error"
            ], check=True, timeout=300)

            return output if Path(output).exists() else None
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            shutil.copy(video, output)
            return output

    def _resolve_background(self, background: str, tmpdir: str) -> str:
        """Return path to background image/video."""
        builtin_bgs = {
            "office":   ROOT / "web" / "assets" / "bg_office.jpg",
            "studio":   ROOT / "web" / "assets" / "bg_studio.jpg",
            "gradient": ROOT / "web" / "assets" / "bg_gradient.jpg",
        }

        if background in builtin_bgs and builtin_bgs[background].exists():
            return str(builtin_bgs[background])

        if Path(background).exists():
            return background

        # Generate a simple gradient background as fallback
        gradient_path = os.path.join(tmpdir, "bg_fallback.jpg")
        subprocess.run([
            "ffmpeg", "-f", "lavfi",
            "-i", "color=c=0x1a1a2e:s=1280x720",
            "-frames:v", "1", gradient_path, "-y", "-loglevel", "error"
        ], check=True)
        return gradient_path

    # ─── Utility helpers ─────────────────────────────────────────
    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            r = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ], capture_output=True, text=True)
            return float(r.stdout.strip())
        except Exception:
            return 60.0

    def _get_video_fps(self, video_path: str) -> float:
        try:
            r = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ], capture_output=True, text=True)
            num, den = r.stdout.strip().split("/")
            return float(num) / float(den)
        except Exception:
            return 30.0

    def _get_video_dims(self, video_path: str):
        try:
            r = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0", video_path
            ], capture_output=True, text=True)
            w, h = r.stdout.strip().split(",")
            return int(w), int(h)
        except Exception:
            return 512, 512


if __name__ == "__main__":
    print("VideoPipeline self-test — requires models to be downloaded first.")
    print("Run setup.sh first, then test via the web UI.")
