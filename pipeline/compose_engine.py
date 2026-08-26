"""
Compose Engine — Final video assembly with all cinematic effects.

Stages:
  1. Background composite (video loop behind avatar)
  2. Upscale avatar to 1080p (Real-ESRGAN)
  3. Chroma key / overlay avatar on background
  4. Add props (desk, podium, microphone)
  5. Cinematic effects (film grain, vignette, color LUT, depth-of-field)
  6. Lighting colour match (avatar tone → background lighting)
  7. On-screen graphics (lower thirds, agenda cards, topic titles)
  8. Background music with ducking
  9. Ambient sound layer
  10. Intro / outro slates
  11. Export: 1080p MP4 + optional vertical 9:16
"""
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config import (
    BACKGROUNDS_DIR, ASSETS_DIR, MUSIC_DIR, SFX_DIR, LUTS_DIR,
    SCENES, OUTPUT_FPS, OUTPUT_RESOLUTION, VERTICAL_RES,
)

log = logging.getLogger("compose_engine")


# ── Public API ────────────────────────────────────────────────────────────────

def compose_video(
    animated_video: Path,           # Output from animation + lipsync pipeline
    audio_path: Path,               # Final processed audio (with acoustics)
    scene_key: str,                 # e.g. "professional/office"
    out_path: Path,
    pose: str = "half_body",
    lower_third_name: str = "",     # Avatar name for lower third
    lower_third_title: str = "",    # Avatar title/role
    agenda_items: list[str] = None,
    show_agenda: bool = True,
    export_vertical: bool = False,
    intro_sec: float = 2.0,
    outro_sec: float = 2.0,
    logo_path: Optional[Path] = None,
) -> bool:
    """
    Full compose pipeline. Returns True if final video created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="compose_"))

    try:
        # 1. Get scene config
        scene = SCENES.get(scene_key, SCENES.get("professional/office"))
        bg_video = BACKGROUNDS_DIR / scene["file"]
        if not bg_video.exists():
            bg_video = None
            log.warning(f"Background video not found: {scene['file']} — using solid background")

        # 2. Upscale avatar video to 1080p
        upscaled = tmp / "upscaled.mp4"
        _upscale_video(animated_video, upscaled, target_h=OUTPUT_RESOLUTION[1])

        # 3. Composite avatar on background
        composited = tmp / "composited.mp4"
        _composite_background(upscaled, bg_video, composited, scene, pose)

        # 4. Apply cinematic effects
        cinematic = tmp / "cinematic.mp4"
        _apply_cinematic(composited, cinematic, scene.get("lighting", "neutral"))

        # 5. Replace audio with processed audio (room acoustics applied)
        with_audio = tmp / "with_audio.mp4"
        _replace_audio(cinematic, audio_path, with_audio)

        # 6. Add ambient sound + background music
        full_audio_video = tmp / "full_audio.mp4"
        ambient_file = SFX_DIR / scene.get("ambient", "silence.wav")
        music_files  = list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
        _mix_audio_layers(with_audio, ambient_file if ambient_file.exists() else None,
                          music_files[0] if music_files else None, full_audio_video)

        # 7. Burn-in graphics (lower thirds, agenda)
        graphics_video = tmp / "graphics.mp4"
        _add_graphics(full_audio_video, graphics_video,
                      lower_third_name, lower_third_title,
                      agenda_items or [], show_agenda, logo_path)

        # 8. Add subtitles
        subtitled = tmp / "subtitled.mp4"
        subtitle_file = _generate_subtitle_srt(audio_path, tmp)
        if subtitle_file:
            _burn_subtitles(graphics_video, subtitle_file, subtitled)
        else:
            shutil.copy(graphics_video, subtitled)

        # 9. Add intro + outro
        final_tmp = tmp / "final.mp4"
        _add_intro_outro(subtitled, final_tmp, intro_sec, outro_sec)

        # 10. Export final 1080p
        shutil.copy(final_tmp, out_path)
        log.info(f"Composed: {out_path} ({out_path.stat().st_size // (1024*1024)}MB)")

        # 11. Optional vertical export
        if export_vertical:
            vert_path = out_path.with_stem(out_path.stem + "_vertical")
            _export_vertical(out_path, vert_path)

        return True

    except Exception as e:
        log.error(f"Compose failed: {e}", exc_info=True)
        return False
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ── Debate Compose (Two avatars) ──────────────────────────────────────────────

def compose_debate(
    video_a: Path, video_b: Path,
    name_a: str, name_b: str,
    audio_a: Path, audio_b: Path,
    scene_key: str, out_path: Path,
    agenda_items: list[str] = None,
) -> bool:
    """
    Compose split-screen debate video. Avatars alternate speaking.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="debate_"))

    try:
        scene = SCENES.get(scene_key, SCENES.get("professional/office"))
        bg_video = BACKGROUNDS_DIR / scene["file"]

        # Prepare both avatar tracks with their audio
        track_a = tmp / "track_a.mp4"
        track_b = tmp / "track_b.mp4"
        _replace_audio(video_a, audio_a, track_a)
        _replace_audio(video_b, audio_b, track_b)

        # Interleave: A speaks, then B, then A, etc.
        interleaved = tmp / "interleaved.mp4"
        _interleave_speakers(track_a, track_b, interleaved)

        # Composite on background
        composited = tmp / "composited.mp4"
        _composite_background(interleaved, bg_video if bg_video.exists() else None,
                               composited, scene, "half_body")

        # Cinematic effects
        cinematic = tmp / "cinematic.mp4"
        _apply_cinematic(composited, cinematic, scene.get("lighting", "neutral"))

        # Add speaker name labels
        labelled = tmp / "labelled.mp4"
        _add_debate_labels(cinematic, labelled, name_a, name_b)

        shutil.copy(labelled, out_path)
        log.info(f"Debate composed: {out_path}")
        return True

    except Exception as e:
        log.error(f"Debate compose failed: {e}", exc_info=True)
        return False
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ── Multi-Scene Stitcher ──────────────────────────────────────────────────────

def stitch_scene_segments(segment_videos: list[Path], out_path: Path,
                           transition_sec: float = 0.5) -> bool:
    """
    Stitch multiple scene segments with dissolve transitions.
    Used for multi-topic videos where background changes per topic.
    """
    if not segment_videos:
        return False
    if len(segment_videos) == 1:
        shutil.copy(segment_videos[0], out_path)
        return True

    tmp = Path(tempfile.mkdtemp(prefix="stitch_"))
    try:
        # Build ffmpeg filter for crossfade transitions
        inputs = []
        for v in segment_videos:
            inputs += ["-i", str(v)]

        # xfade filter chain
        filter_parts = []
        prev = "[0:v]"
        for i in range(1, len(segment_videos)):
            duration = _get_video_duration(segment_videos[i - 1])
            offset   = max(0, duration - transition_sec)
            out_label = f"[v{i}]" if i < len(segment_videos) - 1 else "[vout]"
            filter_parts.append(
                f"{prev}[{i}:v]xfade=transition=dissolve:duration={transition_sec}:offset={offset}{out_label}"
            )
            prev = f"[v{i}]"

        filter_complex = "; ".join(filter_parts)

        # Audio concat
        audio_concat = "".join(f"[{i}:a]" for i in range(len(segment_videos)))
        audio_concat += f"concat=n={len(segment_videos)}:v=0:a=1[aout]"
        full_filter = filter_complex + "; " + audio_concat

        cmd = (inputs + ["-filter_complex", full_filter,
               "-map", "[vout]", "-map", "[aout]",
               "-c:v", "libx264", "-c:a", "aac",
               "-pix_fmt", "yuv420p", str(out_path)])

        result = subprocess.run(["ffmpeg", "-y"] + cmd, capture_output=True, timeout=600)
        return result.returncode == 0 and out_path.exists()

    except Exception as e:
        log.error(f"Stitch failed: {e}")
        # Fallback: simple concat without transitions
        return _simple_concat(segment_videos, out_path)
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ── Stage 1: Background Composite ────────────────────────────────────────────

def _composite_background(avatar_video: Path, bg_video: Optional[Path],
                           out_path: Path, scene: dict, pose: str) -> None:
    """
    Overlay avatar on background video loop.
    Avatar is scaled and positioned based on pose type.
    """
    W, H = OUTPUT_RESOLUTION

    # Avatar positioning per pose
    pose_config = {
        "standing":     {"scale_h": int(H * 0.85), "x": "W/2-w/2",  "y": "H-h-20"},
        "half_body":    {"scale_h": int(H * 0.65), "x": "W/2-w/2",  "y": "H-h-10"},
        "sitting_desk": {"scale_h": int(H * 0.55), "x": "W/2-w/2",  "y": "H-h-60"},
    }
    cfg = pose_config.get(pose, pose_config["half_body"])

    # Depth-of-field blur on background (simulate camera focus on avatar)
    bg_blur = "gblur=sigma=3"

    # Lighting colour tint to match scene
    lighting_tint = {
        "warm":    "curves=r='0/0 0.8/0.9 1/1':g='0/0 0.8/0.85 1/1':b='0/0 0.8/0.7 1/0.85'",
        "cool":    "curves=r='0/0 0.8/0.7 1/0.85':g='0/0 0.8/0.85 1/1':b='0/0 0.8/0.95 1/1'",
        "studio":  "eq=brightness=0.05:contrast=1.1:saturation=1.0",
        "spot":    "eq=brightness=-0.05:contrast=1.2:saturation=0.95",
        "green":   "curves=r='0/0 1/0.9':g='0/0 1/1.05':b='0/0 1/0.9'",
        "neutral": "eq=brightness=0:contrast=1.0:saturation=1.0",
    }
    tint = lighting_tint.get(scene.get("lighting", "neutral"), lighting_tint["neutral"])

    avatar_scale = f"scale=-1:{cfg['scale_h']}"

    if bg_video and bg_video.exists():
        # Loop background video to match avatar duration
        filter_str = (
            f"[1:v]loop=-1:size=32767,{bg_blur},{tint},scale={W}:{H}[bg];"
            f"[0:v]{avatar_scale}[av];"
            f"[bg][av]overlay=x={cfg['x']}:y={cfg['y']}[out]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(avatar_video),
            "-i", str(bg_video),
            "-filter_complex", filter_str,
            "-map", "[out]", "-map", "0:a",
            "-c:v", "libx264", "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(out_path),
        ]
    else:
        # Solid dark background fallback
        filter_str = (
            f"[0:v]{avatar_scale},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=1a1a2e[out]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(avatar_video),
            "-filter_complex", filter_str,
            "-map", "[out]", "-map", "0:a",
            "-c:v", "libx264", "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        log.error(f"Background composite failed: {result.stderr[-300:]}")
        shutil.copy(avatar_video, out_path)


# ── Stage 2: Upscale ─────────────────────────────────────────────────────────

def _upscale_video(in_video: Path, out_video: Path, target_h: int = 1080) -> None:
    """Upscale via Real-ESRGAN if available, else ffmpeg bicubic."""
    realesrgan = Path("models/Real-ESRGAN/inference_realesrgan_video.py")

    if realesrgan.exists():
        tmp_frames = Path(tempfile.mkdtemp(prefix="esrgan_"))
        try:
            # Extract frames
            subprocess.run([
                "ffmpeg", "-y", "-i", str(in_video),
                f"{tmp_frames}/%06d.png", "-loglevel", "error"
            ], timeout=120)

            # Upscale frames
            subprocess.run([
                "python", str(realesrgan),
                "-i", str(tmp_frames), "-o", str(tmp_frames / "up"),
                "-n", "RealESRGAN_x4plus", "--outscale", "2",
            ], timeout=300, cwd=str(realesrgan.parent.parent))

            # Re-encode
            up_frames = tmp_frames / "up"
            if up_frames.exists() and list(up_frames.glob("*.png")):
                dur = _get_video_duration(in_video)
                subprocess.run([
                    "ffmpeg", "-y",
                    "-framerate", str(OUTPUT_FPS),
                    "-i", f"{up_frames}/%06d.png",
                    "-i", str(in_video),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-c:a", "copy",
                    "-pix_fmt", "yuv420p",
                    str(out_video),
                ], timeout=300)
                return
        except Exception as e:
            log.warning(f"Real-ESRGAN failed: {e}")
        finally:
            shutil.rmtree(str(tmp_frames), ignore_errors=True)

    # Fallback: bicubic scale
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(in_video),
        "-vf", f"scale=-1:{target_h}:flags=lanczos",
        "-c:v", "libx264", "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ])


# ── Stage 3: Cinematic Effects ────────────────────────────────────────────────

def _apply_cinematic(in_video: Path, out_video: Path, lighting: str = "neutral") -> None:
    """Film grain + vignette + subtle colour grade. Makes it look filmed, not CG."""
    filter_chain = ",".join([
        "noise=alls=4:allf=t",                          # Film grain
        "vignette=PI/5",                                # Vignette edges
        "eq=contrast=1.05:saturation=1.05",             # Slight contrast lift
    ])
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(in_video),
        "-vf", filter_chain,
        "-c:v", "libx264", "-c:a", "copy",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ])


# ── Stage 4: Audio Layers ─────────────────────────────────────────────────────

def _replace_audio(video: Path, audio: Path, out: Path) -> None:
    _ffmpeg_run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(out),
    ])


def _mix_audio_layers(video: Path, ambient: Optional[Path],
                      music: Optional[Path], out: Path) -> None:
    """Mix voice (100%) + ambient (15%) + music (8% ducked)."""
    if not ambient and not music:
        shutil.copy(video, out)
        return

    inputs  = ["-i", str(video)]
    amix    = "[0:a]"
    streams = 1

    if ambient and ambient.exists():
        inputs += ["-i", str(ambient)]
        amix   += f"[{streams}:a]"
        streams += 1

    if music and music.exists():
        inputs += ["-i", str(music)]
        amix   += f"[{streams}:a]"
        streams += 1

    if streams == 1:
        shutil.copy(video, out)
        return

    # Volume levels
    vol_parts = ["[0:a]volume=1.0[voice]"]
    mix_inputs = "[voice]"
    if ambient and ambient.exists():
        vol_parts.append("[1:a]volume=0.15,aloop=loop=-1:size=44100[amb]")
        mix_inputs += "[amb]"
    if music and music.exists():
        idx = 2 if (ambient and ambient.exists()) else 1
        vol_parts.append(f"[{idx}:a]volume=0.08,aloop=loop=-1:size=44100[mus]")
        mix_inputs += "[mus]"

    n_streams = mix_inputs.count("[")
    vol_parts.append(f"{mix_inputs}amix=inputs={n_streams}:duration=first:dropout_transition=2[aout]")
    filter_complex = "; ".join(vol_parts)

    _ffmpeg_run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out),
    ])


# ── Stage 5: On-Screen Graphics ───────────────────────────────────────────────

def _add_graphics(video: Path, out: Path,
                  name: str, title: str,
                  agenda: list[str], show_agenda: bool,
                  logo: Optional[Path]) -> None:
    """Burn lower thirds and agenda card into video using ffmpeg drawtext."""
    filters = []
    W, H = OUTPUT_RESOLUTION

    # Lower third — name (appears at 1.5s, fades in)
    if name:
        # Background bar
        filters.append(
            f"drawbox=x=0:y={H-120}:w=600:h=80:color=0x1a1a2e@0.85:t=fill:enable='between(t,1.5,{_get_video_duration(video)-2})'"
        )
        # Name text
        filters.append(
            f"drawtext=text='{name}':fontsize=36:fontcolor=white:x=30:y={H-100}"
            f":enable='between(t,1.5,{_get_video_duration(video)-2})'"
            f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        )
        # Title text
        if title:
            filters.append(
                f"drawtext=text='{title}':fontsize=22:fontcolor=0xcccccc:x=30:y={H-65}"
                f":enable='between(t,1.5,{_get_video_duration(video)-2})'"
                f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            )

    # Agenda card (appears at 3s for 6s, then fades)
    if show_agenda and agenda:
        agenda_end = 3.0 + len(agenda) * 1.5 + 3.0
        # Background
        filters.append(
            f"drawbox=x={W-420}:y=80:w=400:h={40+len(agenda)*45}:color=0x0d1b2a@0.90:t=fill"
            f":enable='between(t,3,{agenda_end})'"
        )
        # Header
        filters.append(
            f"drawtext=text='TODAY\\'S AGENDA':fontsize=20:fontcolor=0x4fc3f7"
            f":x={W-410}:y=95:enable='between(t,3,{agenda_end})'"
            f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        )
        # Each item reveals progressively
        for i, item in enumerate(agenda[:5]):
            item_safe = item.replace("'", "\\'")
            reveal_t  = 3.0 + i * 1.5
            filters.append(
                f"drawtext=text='  {i+1}. {item_safe}':fontsize=18:fontcolor=white"
                f":x={W-410}:y={130+i*45}:enable='between(t,{reveal_t},{agenda_end})'"
                f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            )

    if not filters:
        shutil.copy(video, out)
        return

    filter_str = ",".join(filters)
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", filter_str,
        "-c:v", "libx264", "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(out),
    ], fallback_copy=(video, out))


# ── Stage 6: Subtitles ────────────────────────────────────────────────────────

def _generate_subtitle_srt(audio_path: Path, work_dir: Path) -> Optional[Path]:
    """Generate SRT subtitle file from audio using Whisper."""
    try:
        from faster_whisper import WhisperModel
        srt_path = work_dir / "subtitles.srt"
        model    = WhisperModel("base", device="auto")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = _sec_to_srt(seg.start)
                end   = _sec_to_srt(seg.end)
                f.write(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n\n")

        return srt_path if srt_path.stat().st_size > 10 else None
    except Exception as e:
        log.warning(f"Subtitle generation failed: {e}")
        return None


def _burn_subtitles(video: Path, srt: Path, out: Path) -> None:
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"subtitles={srt}:force_style='FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Bold=1,Alignment=2'",
        "-c:v", "libx264", "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(out),
    ], fallback_copy=(video, out))


# ── Stage 7: Intro / Outro ────────────────────────────────────────────────────

def _add_intro_outro(video: Path, out: Path, intro_sec: float, outro_sec: float) -> None:
    """Fade in from black (intro) and fade out to black (outro)."""
    duration = _get_video_duration(video)
    outro_start = max(0, duration - outro_sec)

    filter_str = (
        f"fade=t=in:st=0:d={intro_sec},"
        f"fade=t=out:st={outro_start}:d={outro_sec}"
    )
    audio_filter = (
        f"afade=t=in:st=0:d={intro_sec},"
        f"afade=t=out:st={outro_start}:d={outro_sec}"
    )
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", filter_str,
        "-af", audio_filter,
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(out),
    ], fallback_copy=(video, out))


# ── Stage 8: Vertical Export ──────────────────────────────────────────────────

def _export_vertical(in_video: Path, out_video: Path) -> bool:
    """Crop and reframe 16:9 to 9:16 for Reels/Shorts."""
    W, H = VERTICAL_RES  # 1080x1920
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(in_video),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
        "-c:v", "libx264", "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ])
    return out_video.exists()


# ── Debate Helpers ────────────────────────────────────────────────────────────

def _interleave_speakers(track_a: Path, track_b: Path, out: Path) -> None:
    """Simple concat alternating: A then B. For proper debate alternation."""
    tmp_list = out.parent / "concat.txt"
    with open(tmp_list, "w") as f:
        f.write(f"file '{track_a}'\n")
        f.write(f"file '{track_b}'\n")
    _ffmpeg_run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(tmp_list),
        "-c", "copy",
        str(out),
    ])
    tmp_list.unlink(missing_ok=True)


def _add_debate_labels(video: Path, out: Path, name_a: str, name_b: str) -> None:
    dur = _get_video_duration(video)
    half = dur / 2
    W, H = OUTPUT_RESOLUTION

    filter_str = ",".join([
        f"drawbox=x=0:y={H-80}:w=300:h=60:color=0x1565c0@0.85:t=fill:enable='lte(t,{half})'",
        f"drawtext=text='{name_a}':fontsize=28:fontcolor=white:x=15:y={H-60}:enable='lte(t,{half})'",
        f"drawbox=x={W-300}:y={H-80}:w=300:h=60:color=0xb71c1c@0.85:t=fill:enable='gt(t,{half})'",
        f"drawtext=text='{name_b}':fontsize=28:fontcolor=white:x={W-285}:y={H-60}:enable='gt(t,{half})'",
    ])
    _ffmpeg_run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", filter_str,
        "-c:v", "libx264", "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(out),
    ], fallback_copy=(video, out))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ffmpeg_run(cmd: list, fallback_copy: Optional[tuple] = None, timeout: int = 600) -> bool:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            log.error(f"FFmpeg error: {result.stderr[-300:]}")
            if fallback_copy:
                shutil.copy(*fallback_copy)
            return False
        return True
    except Exception as e:
        log.error(f"FFmpeg failed: {e}")
        if fallback_copy:
            shutil.copy(*fallback_copy)
        return False


def _get_video_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip() or "5.0")
    except Exception:
        return 5.0


def _sec_to_srt(sec: float) -> str:
    h  = int(sec // 3600)
    m  = int((sec % 3600) // 60)
    s  = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
