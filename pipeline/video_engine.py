"""
video_engine.py — Final video composition pipeline

Takes:
  - lipsync.mp4 (MuseTalk output — talking head)
  - background image/video
  - audio (normalised WAV)
  - lower third data (name, title)
  - subtitles (ASS format, Telugu Unicode)
  - optional agenda overlay

Outputs:
  - 16:9  (1920×1080) — YouTube/LinkedIn
  - 9:16  (1080×1920) — Instagram Reels / YouTube Shorts (if requested)
  - 1:1   (1080×1080) — Instagram feed (if requested)

All compositing via FFmpeg (GPU-accelerated where available).
Lower thirds and overlays generated with Pillow.
Subtitles burned in ASS format (preserves Telugu Unicode).
"""
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("video_engine")

ROOT = Path(__file__).parent.parent.resolve()
BACKGROUNDS_DIR = ROOT / "models" / "backgrounds"
ASSETS_DIR      = ROOT / "assets"
FONTS_DIR       = ASSETS_DIR / "fonts"

# Target resolutions
RES_16_9  = (1920, 1080)
RES_9_16  = (1080, 1920)
RES_1_1   = (1080, 1080)

# Try GPU encoder first, fall back to software
def _get_encoder():
    """Check if h264_nvenc is available."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        if r.returncode == 0:
            return "h264_nvenc", ["-preset", "p4", "-tune", "hq"]
    except Exception:
        pass
    return "libx264", ["-preset", "fast", "-crf", "18"]


_ENCODER, _ENCODER_OPTS = _get_encoder()
log.info(f"Video encoder: {_ENCODER}")


# ── Subtitle generation ───────────────────────────────────────────────────────

def generate_ass_subtitles(
    audio_path: Path,
    script_text: str,
    out_ass: Path,
    language: str = "te",
    fps: int = 25,
) -> bool:
    """
    Generate ASS subtitle file from audio + script using Whisper alignment.
    ASS format preserves Telugu Unicode perfectly.
    Falls back to simple even-time distribution if Whisper fails.
    """
    try:
        # Try Whisper with word-level timestamps for accurate sync
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="auto", compute_type="float16")
        segments, _ = model.transcribe(
            str(audio_path), language=language,
            word_timestamps=True, beam_size=5
        )

        events = []
        for seg in segments:
            start = _sec_to_ass(seg.start)
            end   = _sec_to_ass(seg.end)
            text  = seg.text.strip().replace("\n", " ")
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        _write_ass(out_ass, events)
        log.info(f"Whisper subtitles: {len(events)} segments → {out_ass}")
        return True

    except Exception as e:
        log.warning(f"Whisper subtitle alignment failed: {e} — using simple distribution")

    # Fallback: split script evenly across audio duration
    return _make_simple_subtitles(audio_path, script_text, out_ass)


def _sec_to_ass(t: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _write_ass(out_path: Path, events: list[str]) -> None:
    """Write ASS file with Telugu-compatible styles."""
    header = """[Script Info]
Title: Avatar Studio Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans Telugu,38,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _make_simple_subtitles(audio_path: Path, script_text: str, out_ass: Path) -> bool:
    """Even-time subtitle distribution when Whisper alignment is unavailable."""
    try:
        # Get audio duration
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True,
        )
        duration = float(r.stdout.strip()) if r.returncode == 0 else 180.0

        # Split script into sentences
        sentences = [s.strip() for s in re.split(r'[।.!?]+', script_text) if s.strip()]
        if not sentences:
            return False

        time_per = duration / len(sentences)
        events = []
        for i, sent in enumerate(sentences):
            start = _sec_to_ass(i * time_per)
            end   = _sec_to_ass(min((i + 1) * time_per, duration))
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{sent}")

        _write_ass(out_ass, events)
        log.info(f"Simple subtitles: {len(events)} lines → {out_ass}")
        return True
    except Exception as e:
        log.error(f"Simple subtitles failed: {e}")
        return False


# ── Lower third overlay ───────────────────────────────────────────────────────

def make_lower_third(
    name: str,
    title: str,
    width: int = 1920,
    out_png: Optional[Path] = None,
) -> Optional[Path]:
    """
    Generate a professional lower third PNG with transparent background.
    Accent bar | Name (bold white) | Title (small gold)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        H = 90
        img = Image.new("RGBA", (width, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Background panel (dark, semi-transparent)
        panel_w = min(width - 80, len(name) * 32 + len(title) * 20 + 240)
        draw.rectangle([(40, 0), (40 + panel_w, H)], fill=(10, 10, 20, 210))

        # Accent bar (saffron gold)
        draw.rectangle([(40, 0), (48, H)], fill=(232, 146, 10, 255))

        # Try loading a font; fall back to default
        def _font(size: int):
            for fp in [
                FONTS_DIR / "NotoSans-Bold.ttf",
                FONTS_DIR / "Roboto-Bold.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            ]:
                try:
                    return ImageFont.truetype(str(fp), size)
                except Exception:
                    pass
            return ImageFont.load_default()

        name_font  = _font(36)
        title_font = _font(22)

        draw.text((60, 12),  name,  font=name_font,  fill=(255, 255, 255, 255))
        draw.text((60, 56),  title, font=title_font, fill=(232, 146, 10, 240))

        if out_png is None:
            import tempfile
            out_png = Path(tempfile.mktemp(suffix="_lower_third.png"))

        img.save(str(out_png), "PNG")
        return out_png

    except Exception as e:
        log.warning(f"Lower third generation failed: {e}")
        return None


# ── Background handling ───────────────────────────────────────────────────────

def _get_background(scene_key: str, width: int, height: int) -> Optional[Path]:
    """
    Find background image for scene_key. Returns path or None.
    Searches BACKGROUNDS_DIR for matching PNG.
    """
    # Convert scene key to filename: "professional/office" → "professional_office.png"
    filename = scene_key.replace("/", "_") + ".png"
    candidates = [
        BACKGROUNDS_DIR / filename,
        BACKGROUNDS_DIR / (filename.split("_", 1)[-1]),   # "office.png"
        BACKGROUNDS_DIR / "professional_office.png",        # default
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last resort: pick any PNG from backgrounds dir
    pngs = list(BACKGROUNDS_DIR.glob("*.png"))
    return pngs[0] if pngs else None


# ── Core compose function ─────────────────────────────────────────────────────

def compose_video(
    lipsync_video: Path,
    audio_path: Path,
    scene_key: str,
    out_path: Path,
    lower_third_name: str = "",
    lower_third_title: str = "",
    script_text: str = "",
    language: str = "te",
    show_subtitles: bool = True,
    export_vertical: bool = False,
    export_square: bool = False,
    fps: int = 25,
) -> bool:
    """
    Full video composition:
      background + lipsync_avatar + lower_third + subtitles → final MP4
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = RES_16_9

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Step 1: Background ────────────────────────────────────────────────
        bg_path = _get_background(scene_key, W, H)
        bg_input = []
        if bg_path:
            # Scale background to full resolution
            bg_scaled = tmp / "bg.png"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(bg_path),
                 "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                 str(bg_scaled)],
                capture_output=True, timeout=30,
            )
            if bg_scaled.exists():
                bg_input = ["-i", str(bg_scaled)]
            else:
                bg_input = ["-i", str(bg_path)]

        # ── Step 2: Lower third ───────────────────────────────────────────────
        lt_path = None
        if lower_third_name:
            lt_path = make_lower_third(lower_third_name, lower_third_title, W, tmp / "lt.png")

        # ── Step 3: Subtitles ─────────────────────────────────────────────────
        ass_path = None
        if show_subtitles and script_text:
            ass_path = tmp / "subs.ass"
            gen_ok = generate_ass_subtitles(audio_path, script_text, ass_path, language, fps)
            if not gen_ok:
                ass_path = None

        # ── Step 4: Build FFmpeg filtergraph ─────────────────────────────────
        # Input slots
        # [0] background
        # [1] lipsync video
        # [2] lower third PNG (optional)

        inputs = []
        filter_parts = []
        input_idx = 0

        if bg_input:
            inputs += bg_input
            bg_label = f"[{input_idx}:v]"
            input_idx += 1
        else:
            # No background: use black frame
            inputs += ["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={fps}"]
            bg_label = f"[{input_idx}:v]"
            input_idx += 1

        # Lipsync video (scale to ~40% width, position bottom-right for news style)
        inputs += ["-i", str(lipsync_video)]
        av_idx = input_idx
        input_idx += 1

        # Avatar size: 45% of width, maintain aspect
        av_w = int(W * 0.45)
        av_h = int(H * 0.90)
        av_x = W - av_w - 20        # right side, 20px margin
        av_y = H - av_h - 10        # bottom, 10px margin

        # Build filtergraph
        # Scale background to output resolution
        filter_parts.append(
            f"{bg_label}scale={W}:{H},setsar=1[bg]"
        )
        # Scale avatar video
        filter_parts.append(
            f"[{av_idx}:v]scale={av_w}:{av_h}:force_original_aspect_ratio=decrease,"
            f"pad={av_w}:{av_h}:(ow-iw)/2:(oh-ih)/2:color=black@0,setsar=1[av]"
        )
        # Overlay avatar on background
        filter_parts.append(
            f"[bg][av]overlay={av_x}:{av_y}[comp]"
        )

        last_label = "[comp]"

        # Lower third overlay
        if lt_path and lt_path.exists():
            inputs += ["-i", str(lt_path)]
            lt_idx = input_idx
            input_idx += 1
            lt_y = H - 90 - 60    # 60px from bottom
            filter_parts.append(
                f"{last_label}[{lt_idx}:v]overlay=0:{lt_y}[withlower]"
            )
            last_label = "[withlower]"

        # Subtitle burn-in
        if ass_path and ass_path.exists():
            # ASS subtitle via subtitles filter
            filter_parts.append(
                f"{last_label}subtitles={str(ass_path)}:fontsdir={str(FONTS_DIR)}[withsubs]"
            )
            last_label = "[withsubs]"

        # Final output label
        filter_parts.append(f"{last_label}copy[out]")

        # Audio: use our normalised WAV
        inputs += ["-i", str(audio_path)]
        audio_idx = input_idx

        # Build full FFmpeg command
        filtergraph = ";".join(filter_parts)
        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-filter_complex", filtergraph,
               "-map", "[out]",
               "-map", f"{audio_idx}:a",
               "-c:v", _ENCODER] + _ENCODER_OPTS
            + ["-c:a", "aac", "-b:a", "192k",
               "-r", str(fps),
               "-movflags", "+faststart",
               "-t", _get_audio_duration(audio_path),   # trim to audio length
               str(out_path)]
        )

        log.info(f"FFmpeg compose | encoder={_ENCODER}")
        r = subprocess.run(cmd, capture_output=True, timeout=900)

        if r.returncode != 0 or not out_path.exists():
            log.error(f"FFmpeg compose failed:\n{r.stderr.decode()[-800:]}")
            # Emergency fallback: just mux lipsync + audio
            return _emergency_mux(lipsync_video, audio_path, out_path)

        log.info(f"Compose OK → {out_path} ({out_path.stat().st_size//1024}KB)")

        # ── Export additional formats ─────────────────────────────────────────
        if export_vertical:
            vert_path = out_path.with_stem(out_path.stem + "_reels")
            _export_vertical(out_path, vert_path)

        if export_square:
            sq_path = out_path.with_stem(out_path.stem + "_square")
            _export_square(out_path, sq_path)

    return out_path.exists() and out_path.stat().st_size > 10240


# ── Format converters ─────────────────────────────────────────────────────────

def _export_vertical(src: Path, dst: Path) -> bool:
    """Convert 16:9 → 9:16 (1080×1920) for Reels/Shorts."""
    W, H = RES_9_16
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vf", (
             f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
             f"setsar=1"
         ),
         "-c:v", _ENCODER] + _ENCODER_OPTS + ["-c:a", "copy", str(dst)],
        capture_output=True, timeout=300,
    )
    ok = r.returncode == 0 and dst.exists()
    if ok:
        log.info(f"Vertical export OK → {dst}")
    return ok


def _export_square(src: Path, dst: Path) -> bool:
    """Convert 16:9 → 1:1 (1080×1080) for Instagram feed."""
    W = H = 1080
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-vf", (
             f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
             f"setsar=1"
         ),
         "-c:v", _ENCODER] + _ENCODER_OPTS + ["-c:a", "copy", str(dst)],
        capture_output=True, timeout=300,
    )
    ok = r.returncode == 0 and dst.exists()
    if ok:
        log.info(f"Square export OK → {dst}")
    return ok


# ── Debate layout ─────────────────────────────────────────────────────────────

def compose_debate(
    lipsync_a: Path,
    lipsync_b: Path,
    audio_a: Path,
    audio_b: Path,
    name_a: str,
    name_b: str,
    scene_key: str,
    out_path: Path,
    fps: int = 25,
) -> bool:
    """
    Split-screen debate composition.
    Left: Speaker A | Right: Speaker B
    Interleaves their audio into one mixed track.
    """
    out_path = Path(out_path)
    W, H = RES_16_9
    half_w = W // 2

    # Mix audio (simple concat would desync — instead mix at low volume)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mixed_audio = tmp / "mixed.wav"
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(audio_a), "-i", str(audio_b),
             "-filter_complex", "amix=inputs=2:duration=longest:dropout_transition=3",
             str(mixed_audio)],
            capture_output=True, timeout=120,
        )

        bg_path = _get_background(scene_key, W, H)

        # Lower thirds for each speaker
        lt_a = make_lower_third(name_a, "Speaker A", half_w, tmp / "lt_a.png")
        lt_b = make_lower_third(name_b, "Speaker B", half_w, tmp / "lt_b.png")

        inputs = (
            (["-i", str(bg_path)] if bg_path else ["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={fps}"])
            + ["-i", str(lipsync_a), "-i", str(lipsync_b)]
        )
        a_idx, b_idx = (1, 2) if bg_path else (1, 2)

        av_h = int(H * 0.85)
        filtergraph = (
            f"[0:v]scale={W}:{H}[bg];"
            f"[{a_idx}:v]scale={half_w}:{av_h}:force_original_aspect_ratio=decrease,"
            f"pad={half_w}:{av_h}:(ow-iw)/2:(oh-ih)/2:black@0[ava];"
            f"[{b_idx}:v]scale={half_w}:{av_h}:force_original_aspect_ratio=decrease,"
            f"pad={half_w}:{av_h}:(ow-iw)/2:(oh-ih)/2:black@0[avb];"
            f"[bg][ava]overlay=0:{H - av_h - 10}[left];"
            f"[left][avb]overlay={half_w}:{H - av_h - 10}[out]"
        )

        inputs += ["-i", str(mixed_audio)]
        audio_input = len(inputs) // 2

        cmd = (
            ["ffmpeg", "-y"] + inputs
            + ["-filter_complex", filtergraph,
               "-map", "[out]",
               "-map", f"{audio_input}:a",
               "-c:v", _ENCODER] + _ENCODER_OPTS
            + ["-c:a", "aac", "-b:a", "192k",
               "-r", str(fps),
               "-movflags", "+faststart",
               str(out_path)]
        )
        r = subprocess.run(cmd, capture_output=True, timeout=900)
        ok = r.returncode == 0 and out_path.exists()
        if ok:
            log.info(f"Debate compose OK → {out_path}")
        else:
            log.error(f"Debate compose failed:\n{r.stderr.decode()[-600:]}")
        return ok


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_audio_duration(audio_path: Path) -> str:
    """Get audio duration string for FFmpeg -t flag."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "300"   # 5 min default


def _emergency_mux(video: Path, audio: Path, out: Path) -> bool:
    """Last resort: mux lipsync video with audio, no compositing."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=120,
        )
        return r.returncode == 0 and out.exists()
    except Exception:
        return False


# ── Thumbnail generator ───────────────────────────────────────────────────────

def generate_thumbnail(
    avatar_path: Path,
    title_text: str,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
) -> bool:
    """Generate a YouTube-style thumbnail: avatar + title text overlay."""
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageFilter

        # Load and resize avatar
        avatar = Image.open(str(avatar_path)).convert("RGBA")
        avatar = avatar.resize((height, height), Image.LANCZOS)

        # Create base (dark gradient)
        base = Image.new("RGBA", (width, height), (12, 15, 30, 255))
        # Gradient: darker on left, lighter on right
        for x in range(width):
            alpha = int(180 * (1 - x / width))
            for y in range(height):
                r2, g2, b2, _ = base.getpixel((x, y))
                base.putpixel((x, y), (r2, g2, b2, 255))

        # Paste avatar on right side
        av_x = width - height - 10
        base.paste(avatar, (av_x, 0), avatar)

        draw = ImageDraw.Draw(base)

        # Title text (left side)
        def _font(size):
            for fp in [FONTS_DIR / "NotoSans-Bold.ttf", FONTS_DIR / "Roboto-Bold.ttf"]:
                try:
                    return ImageFont.truetype(str(fp), size)
                except Exception:
                    pass
            return ImageFont.load_default()

        # Wrap long title
        words = title_text.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 20:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))
        lines = lines[:3]

        font_big = _font(72)
        y_pos = height // 2 - len(lines) * 80 // 2
        for line in lines:
            draw.text((40, y_pos), line, font=font_big, fill=(255, 255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0, 200))
            y_pos += 80

        # Saffron accent line
        draw.rectangle([(40, height - 20), (av_x - 20, height - 8)],
                        fill=(232, 146, 10, 255))

        base_rgb = base.convert("RGB")
        base_rgb.save(str(out_path), "JPEG", quality=92)
        log.info(f"Thumbnail OK → {out_path}")
        return True

    except Exception as e:
        log.warning(f"Thumbnail generation failed: {e}")
        return False
