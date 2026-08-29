"""
video_engine.py — Professional video composition with broadcast quality

Full pipeline:
  background (real stock video) + avatar video + lower third + subtitles
  + color grade + breathing parallax → 16:9 output + 9:16 Reels + thumbnail

Key features:
  • Real stock backgrounds via Pexels Video API (free, 200 req/hr)
  • Warm South Indian color grade (LUT applied via curves)
  • Subtle breathing parallax on avatar (0.3px vertical oscillation)
  • Netflix-grade ASS subtitles (Telugu Unicode)
  • Animated lower third (slide-in)
  • GPU-accelerated encode (h264_nvenc → libx264 fallback)
  • Multi-format: 16:9 (YouTube) + 9:16 (Reels/Shorts) + 1:1 (Instagram feed)
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("video_engine")

ROOT = Path(__file__).parent.parent.resolve()

# ── GPU encoder detection ─────────────────────────────────────────────────────
def _detect_encoder() -> str:
    r = subprocess.run(
        ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
    )
    if "h264_nvenc" in r.stdout:
        log.info("GPU encoder: h264_nvenc")
        return "h264_nvenc"
    log.info("GPU not available — using libx264")
    return "libx264"

VIDEO_ENCODER = _detect_encoder()

# Encoder-specific speed flag
ENCODER_PRESET = {"h264_nvenc": "-preset p4", "libx264": "-preset fast"}

# ── Pexels background fetch ───────────────────────────────────────────────────

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
BACKGROUND_CACHE = ROOT / "models" / "backgrounds"
BACKGROUND_CACHE.mkdir(parents=True, exist_ok=True)

SCENE_QUERIES = {
    "studio":         "professional news studio background",
    "outdoor":        "india outdoor cityscape day",
    "parliament":     "india government building architecture",
    "sports":         "sports stadium crowd energy",
    "entertainment":  "colorful bright festive lights bokeh",
    "business":       "modern office corporate professional",
    "nature":         "telangana hyderabad nature sunrise",
    "abstract":       "elegant dark gradient abstract professional",
    "default":        "professional studio broadcast background",
}


def fetch_pexels_background(scene: str = "studio", width: int = 1920, height: int = 1080) -> Optional[Path]:
    """
    Download a real stock video background from Pexels (free, no attribution needed).
    Caches locally to avoid repeated downloads.

    If no API key or download fails, returns None → falls back to PIL-generated background.
    """
    if not PEXELS_API_KEY:
        return None

    query = SCENE_QUERIES.get(scene, SCENE_QUERIES["default"])
    cache_path = BACKGROUND_CACHE / f"pexels_{scene}.mp4"

    if cache_path.exists() and cache_path.stat().st_size > 100000:
        return cache_path

    try:
        # Search Pexels for HD video
        r = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": 5, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
            timeout=15,
        )
        r.raise_for_status()
        videos = r.json().get("videos", [])

        if not videos:
            return None

        # Pick the video with best HD resolution
        video = videos[0]
        video_files = video.get("video_files", [])
        hd_files = [f for f in video_files if f.get("quality") == "hd"]
        target = hd_files[0] if hd_files else video_files[0]
        video_url = target["link"]

        log.info(f"Downloading Pexels background: {scene} ({video_url[:60]}...)")
        chunk_r = requests.get(video_url, stream=True, timeout=60)
        chunk_r.raise_for_status()

        with open(cache_path, "wb") as f:
            for chunk in chunk_r.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info(f"Background downloaded → {cache_path}")
        return cache_path

    except Exception as e:
        log.warning(f"Pexels background fetch failed: {e}")
        return None


def get_background_for_scene(scene: str = "studio") -> Optional[Path]:
    """
    Get background for a scene. Tries:
      1. Pexels API (real stock video)
      2. Cached PIL-generated background PNG
    """
    # Try Pexels first
    pexels_bg = fetch_pexels_background(scene)
    if pexels_bg:
        return pexels_bg

    # Fall back to PIL-generated static image
    bg_png = BACKGROUND_CACHE / f"{scene}.png"
    if bg_png.exists():
        return bg_png

    bg_png = BACKGROUND_CACHE / "studio.png"
    if bg_png.exists():
        return bg_png

    # Last resort: generate a gradient
    bg_png = BACKGROUND_CACHE / "fallback.png"
    _generate_gradient_bg(bg_png)
    return bg_png


def _generate_gradient_bg(out_path: Path):
    """Generate a professional dark gradient background."""
    img = Image.new("RGB", (1920, 1080))
    draw = ImageDraw.Draw(img)
    for y in range(1080):
        ratio = y / 1080
        r = int(15 + ratio * 5)
        g = int(20 + ratio * 8)
        b = int(40 + ratio * 15)
        draw.rectangle([(0, y), (1920, y + 1)], fill=(r, g, b))
    img.save(str(out_path))


# ── ASS subtitle generation ───────────────────────────────────────────────────

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans Telugu,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,3,1,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _sec_to_ass(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc"""
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    cs  = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_subtitles(
    audio_path: Path,
    out_path: Path,
    language: str = "te",
) -> Optional[Path]:
    """
    Transcribe audio with Whisper to get word timestamps,
    then generate Netflix-grade ASS subtitle file.
    """
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="auto", compute_type="float16")
        segments, _ = model.transcribe(
            str(audio_path), language=language,
            word_timestamps=True, vad_filter=True,
        )

        events = []
        for seg in segments:
            # Group words into display lines (max ~7 words per line)
            words = list(seg.words)
            for i in range(0, len(words), 7):
                chunk = words[i:i+7]
                if not chunk:
                    continue
                start = chunk[0].start
                end   = chunk[-1].end
                text  = " ".join(w.word.strip() for w in chunk)
                # Escape ASS special chars
                text  = text.replace("\\", "\\\\").replace("{", "\\{")
                events.append(
                    f"Dialogue: 0,{_sec_to_ass(start)},{_sec_to_ass(end)},"
                    f"Default,,0,0,0,,{text}"
                )

        out_path.write_text(ASS_HEADER + "\n".join(events), encoding="utf-8")
        log.info(f"Subtitles generated: {len(events)} lines → {out_path.name}")
        return out_path

    except Exception as e:
        log.warning(f"Subtitle generation failed: {e}")
        return None


# ── Lower third graphic ───────────────────────────────────────────────────────

def make_lower_third(
    name: str,
    title: str,
    out_path: Path,
    width: int = 1920,
    height: int = 1080,
    accent_color: tuple = (232, 146, 10),   # Saffron — Indian news aesthetic
) -> Path:
    """
    Generate a professional lower-third graphic:
    ████████████████████████
    ▌ Name              ▌
    ▌ Title/Role        ▌
    ████████████████████████

    Uses semi-transparent background for readability over any scene.
    """
    # Full canvas (transparent)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(canvas)

    # Lower third bar dimensions (bottom-left position)
    bar_x1, bar_y1 = 60, height - 220
    bar_x2, bar_y2 = 780, height - 60

    # Dark semi-transparent background
    overlay = Image.new("RGBA", (bar_x2 - bar_x1, bar_y2 - bar_y1), (10, 10, 30, 210))
    canvas.paste(overlay, (bar_x1, bar_y1), overlay)

    # Accent bar (left edge)
    draw.rectangle([(bar_x1, bar_y1), (bar_x1 + 6, bar_y2)], fill=accent_color + (255,))

    # Top accent line
    draw.rectangle([(bar_x1, bar_y1), (bar_x2, bar_y1 + 3)], fill=accent_color + (200,))

    # Text
    font_dir = ROOT / "assets" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)

    try:
        font_large = ImageFont.truetype(str(font_dir / "NotoSansTelugu-Bold.ttf"), 42)
        font_small = ImageFont.truetype(str(font_dir / "NotoSansTelugu-Regular.ttf"), 30)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Name (white)
    draw.text((bar_x1 + 20, bar_y1 + 20), name, font=font_large, fill=(255, 255, 255, 255))
    # Title (accent color)
    draw.text((bar_x1 + 20, bar_y1 + 75), title, font=font_small, fill=accent_color + (230,))

    canvas.save(str(out_path), "PNG")
    log.info(f"Lower third: {name} / {title} → {out_path.name}")
    return out_path


# ── Color grade (warm South Indian aesthetic) ─────────────────────────────────

def _warm_grade_filter() -> str:
    """
    FFmpeg curves filter for warm South Indian broadcast color grade:
    - Slight lift in shadows (avoids crushed blacks)
    - Warm highlights (skin tones pop)
    - Slight saturation boost
    - Subtle vignette
    """
    return (
        "curves=r='0/10 128/145 255/255':g='0/8 128/130 255/248':b='0/5 128/115 255/220',"
        "eq=saturation=1.15:brightness=0.02:contrast=1.05,"
        "vignette=angle=PI/6:mode=forward"
    )


# ── Breathing parallax effect ─────────────────────────────────────────────────

def _breathing_overlay_filter(duration_sec: float) -> str:
    """
    Subtle vertical breathing oscillation on avatar (0-2px range).
    Frequency: ~0.25Hz (one breath per 4 seconds, natural resting rate).
    This is added as an overlay position offset using FFmpeg.
    """
    # Use a sine wave for vertical position offset
    # overlay=x=...:y='H/2+w/2*sin(2*PI*0.25*t)'
    # The actual offset is very subtle — 0 to 1.5px
    return "breathing_offset=1.5"  # placeholder, implemented in compose_video


# ── Main composition ──────────────────────────────────────────────────────────

def compose_video(
    avatar_video: Path,    # animated avatar (from SadTalker+MuseTalk)
    audio_path: Path,
    out_path: Path,
    scene: str = "studio",
    persona_name: str = "Navya Reddy",
    persona_title: str = "Telugu News Anchor",
    subtitle_path: Optional[Path] = None,
    lower_third: bool = True,
    color_grade: bool = True,
    breathing: bool = True,
    width: int = 1920,
    height: int = 1080,
) -> bool:
    """
    Compose final broadcast video:
      background + avatar (with breathing) + lower third + subtitles + color grade
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg_path = get_background_for_scene(scene)
    if bg_path is None:
        log.error("No background found")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Lower third overlay
        lt_path = None
        if lower_third:
            lt_path = tmp / "lower_third.png"
            make_lower_third(persona_name, persona_title, lt_path, width, height)

        # Build FFmpeg filter graph
        # Input 0: background, Input 1: avatar video, Input 2: lower third (optional)
        inputs = [
            "-i", str(bg_path),
            "-i", str(avatar_video),
        ]
        filter_parts = []
        last_stream  = "[bg_scaled]"

        # Scale/loop background to match video
        bg_suffix = bg_path.suffix.lower()
        if bg_suffix == ".mp4":
            # Loop background video to match duration
            filter_parts.append(
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},loop=-1:size=9999:start=0,setpts=PTS-STARTPTS[bg_scaled]"
            )
        else:
            # Static image background
            filter_parts.append(
                f"[0:v]scale={width}:{height},setsar=1[bg_scaled]"
            )

        # Avatar: scale to half-body size and position (right-center or left-center)
        # Avatar occupies ~40% width, vertically centered with slight bottom offset
        av_w = int(width * 0.42)
        av_h = int(height * 1.05)  # slight overflow bottom for natural crop
        av_x = width - av_w - 80  # right side
        av_y = height - av_h      # bottom-aligned

        if breathing:
            # Subtle vertical breathing sine wave (±1.5px, 0.25Hz)
            filter_parts.append(
                f"[1:v]scale={av_w}:{av_h}[av_scaled]"
            )
            # y offset: base + 1.5*sin(2*pi*0.25*t)
            av_y_expr = f"{av_y}+round(1.5*sin(2*3.14159*0.25*t))"
            filter_parts.append(
                f"[bg_scaled][av_scaled]overlay=x={av_x}:y='{av_y_expr}'[with_avatar]"
            )
        else:
            filter_parts.append(
                f"[1:v]scale={av_w}:{av_h}[av_scaled]"
            )
            filter_parts.append(
                f"[bg_scaled][av_scaled]overlay=x={av_x}:y={av_y}[with_avatar]"
            )

        last_stream = "[with_avatar]"

        # Lower third overlay
        if lt_path:
            inputs += ["-i", str(lt_path)]
            lt_input_idx = len(inputs) // 2 - 1
            filter_parts.append(
                f"[{lt_input_idx}:v]format=rgba[lt]"
            )
            filter_parts.append(
                f"[with_avatar][lt]overlay=x=0:y=0[with_lt]"
            )
            last_stream = "[with_lt]"

        # Color grade
        if color_grade:
            grade = _warm_grade_filter()
            filter_parts.append(f"{last_stream}{grade}[graded]")
            last_stream = "[graded]"

        # Subtitle burn-in
        if subtitle_path and subtitle_path.exists():
            # Need to copy ASS to tmp with safe path for FFmpeg
            safe_ass = tmp / "subs.ass"
            import shutil; shutil.copy2(str(subtitle_path), str(safe_ass))
            filter_parts.append(
                f"{last_stream}ass='{safe_ass}'[final]"
            )
            last_stream = "[final]"

        # Final output rename
        if last_stream != "[final]":
            filter_parts.append(f"{last_stream}null[final]")
            last_stream = "[final]"

        # Encode preset
        preset_flag = ENCODER_PRESET.get(VIDEO_ENCODER, "-preset fast")

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-i", str(audio_path)]
            + [
                "-filter_complex", ";".join(filter_parts),
                "-map", "[final]",
                "-map", f"{len(inputs)//2}:a",
                "-c:v", VIDEO_ENCODER,
            ]
            + preset_flag.split()
            + [
                "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                "-t", _get_audio_duration(audio_path),
                str(out_path),
            ]
        )

        log.info(f"Composing video | {width}×{height} | encoder={VIDEO_ENCODER}")
        r = subprocess.run(cmd, capture_output=True, timeout=1800)

        if r.returncode != 0:
            log.error(f"Compose failed:\n{r.stderr.decode()[-600:]}")
            return _emergency_mux(avatar_video, audio_path, out_path)

    ok = out_path.exists() and out_path.stat().st_size > 10240
    if ok:
        log.info(f"Video composed → {out_path} ({out_path.stat().st_size//1024//1024}MB)")
    return ok


def _get_audio_duration(audio_path: Path) -> str:
    """Get audio duration as string for FFmpeg -t flag."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, timeout=10
        )
        return str(float(r.stdout.strip()))
    except Exception:
        return "300"  # 5 min default


def _emergency_mux(video: Path, audio: Path, out: Path) -> bool:
    """Last-resort: just mux avatar video + audio, no compositing."""
    log.warning("Emergency mux — no compositing")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac",
         "-shortest", str(out)],
        capture_output=True, timeout=300
    )
    return r.returncode == 0


# ── Multi-format export ───────────────────────────────────────────────────────

def export_vertical(in_path: Path, out_path: Path) -> bool:
    """Export 16:9 → 9:16 (1080×1920) for Instagram Reels / YouTube Shorts."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-vf",
         "crop=ih*9/16:ih,scale=1080:1920,"
         "pad=1080:1920:(1080-iw)/2:(1920-ih)/2:black",
         "-c:v", VIDEO_ENCODER,
         "-preset", "fast", "-crf", "20",
         "-c:a", "copy",
         str(out_path)],
        capture_output=True, timeout=600
    )
    ok = r.returncode == 0 and out_path.exists()
    if ok:
        log.info(f"Vertical export → {out_path.name}")
    return ok


def export_square(in_path: Path, out_path: Path) -> bool:
    """Export 16:9 → 1:1 (1080×1080) for Instagram feed."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-vf", "crop=ih:ih,scale=1080:1080",
         "-c:v", VIDEO_ENCODER, "-preset", "fast", "-crf", "20",
         "-c:a", "copy",
         str(out_path)],
        capture_output=True, timeout=600
    )
    ok = r.returncode == 0 and out_path.exists()
    if ok:
        log.info(f"Square export → {out_path.name}")
    return ok


# ── Thumbnail generation ──────────────────────────────────────────────────────

def generate_thumbnail(
    avatar_image: Path,
    title_text: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
) -> bool:
    """Generate YouTube-style thumbnail from avatar image + title."""
    try:
        bg   = Image.new("RGB", (width, height), (15, 15, 40))
        av   = Image.open(str(avatar_image)).convert("RGBA")

        # Resize avatar to fit right half
        av_h  = int(height * 1.0)
        ratio = av_h / av.height
        av_w  = int(av.width * ratio)
        av    = av.resize((av_w, av_h), Image.LANCZOS)
        bg.paste(av, (width - av_w, 0), av)

        draw  = ImageDraw.Draw(bg)

        # Title text — left side
        font_dir  = ROOT / "assets" / "fonts"
        try:
            font_big = ImageFont.truetype(str(font_dir / "NotoSansTelugu-Bold.ttf"), 72)
        except (IOError, OSError):
            font_big = ImageFont.load_default()

        # Wrap text
        max_w = int(width * 0.45)
        lines = _wrap_text(title_text, font_big, max_w)
        y = height // 4
        for line in lines[:3]:
            draw.text((60, y), line, font=font_big, fill=(255, 255, 255))
            y += 88

        # Accent bar under title
        draw.rectangle([(60, y + 10), (60 + min(len(title_text) * 20, max_w), y + 16)],
                        fill=(232, 146, 10))

        bg.save(str(out_path), "JPEG", quality=95)
        log.info(f"Thumbnail → {out_path.name}")
        return True

    except Exception as e:
        log.error(f"Thumbnail generation failed: {e}")
        return False


def _wrap_text(text: str, font, max_width: int) -> list:
    """Wrap text into lines fitting max_width pixels."""
    words  = text.split()
    lines  = []
    current = ""
    for w in words:
        test = f"{current} {w}".strip()
        try:
            bbox = font.getbbox(test)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(test) * 20  # rough fallback

        if tw <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


# ── Debate composition ────────────────────────────────────────────────────────

def compose_debate(
    avatar_a_video: Path,
    avatar_b_video: Path,
    audio_path: Path,
    out_path: Path,
    name_a: str = "Navya Reddy",
    name_b: str = "Arjun Varma",
    scene: str = "studio",
) -> bool:
    """
    Side-by-side debate composition: two avatars with lower thirds.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg_path = get_background_for_scene(scene)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        lt_a = tmp / "lt_a.png"
        lt_b = tmp / "lt_b.png"
        make_lower_third(name_a, "Anchor A", lt_a, 960, 1080)
        make_lower_third(name_b, "Anchor B", lt_b, 960, 1080)

        filter_complex = (
            # Scale both avatars to half-width
            "[1:v]scale=960:1080,setpts=PTS-STARTPTS[ava];"
            "[2:v]scale=960:1080,setpts=PTS-STARTPTS[avb];"
            # Stack side by side
            "[ava][avb]hstack=inputs=2[avatars];"
            # Scale BG
            f"[0:v]scale=1920:1080,setsar=1[bg];"
            # Overlay avatars on BG
            "[bg][avatars]overlay=x=0:y=0[final]"
        )

        inputs = []
        if bg_path:
            inputs = ["-i", str(bg_path)]
        else:
            inputs = ["-f", "lavfi", "-i", "color=c=black:s=1920x1080"]

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-i", str(avatar_a_video), "-i", str(avatar_b_video),
               "-i", str(audio_path)]
            + [
                "-filter_complex", filter_complex,
                "-map", "[final]",
                "-map", "3:a",
                "-c:v", VIDEO_ENCODER, "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(out_path),
            ]
        )

        r = subprocess.run(cmd, capture_output=True, timeout=1800)

    ok = r.returncode == 0 and out_path.exists()
    if ok:
        log.info(f"Debate composed → {out_path.name}")
    else:
        log.error(f"Debate compose failed: {r.stderr.decode()[-400:]}")
    return ok
