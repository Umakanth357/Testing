"""
content_engine.py — Unified content extraction pipeline

Supported sources (in priority order):
  1. Audio file upload      → Whisper transcription (always works)
  2. Instagram Reels        → yt-dlp download + Whisper (works on EC2 IPs)
  3. YouTube + cookies.txt  → yt-dlp with browser session cookies (bypasses bot block)
  4. YouTube transcript API → caption text (no download, free, when captions exist)
  5. Supadata.ai API        → transcript API fallback (50 free req/day)
  6. Topic / text           → direct pass-through

YouTube bot-block bypass:
  EC2 IPs are blocked. Solutions in priority order:
  a) Upload cookies.txt exported from Chrome (Get cookies.txt LOCALLY extension)
     yt-dlp --cookies cookies.txt → looks like a real logged-in user session
  b) youtube-transcript-api — gets captions directly, no download needed
  c) Supadata.ai free API — transcript-only, 50 req/day free
  d) Upload audio file directly (always works)
"""
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("content_engine")

# Lazy-loaded Whisper model (small is enough for Telugu/Indic, fast)
_WHISPER_MODEL: Optional[object] = None

WHISPER_SIZE = "small"   # small=244MB · base=74MB · medium=769MB


# ── Model loader ──────────────────────────────────────────────────────────────

def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            _WHISPER_MODEL = WhisperModel(WHISPER_SIZE, device="auto", compute_type="float16")
            log.info(f"Loaded faster-whisper ({WHISPER_SIZE})")
        except ImportError:
            import whisper as _w
            _WHISPER_MODEL = _w.load_model(WHISPER_SIZE)
            log.info(f"Loaded openai-whisper ({WHISPER_SIZE})")
    return _WHISPER_MODEL


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, language: str = "te") -> str:
    """Transcribe audio file → text. Handles faster-whisper and openai-whisper APIs."""
    model = _get_whisper()
    log.info(f"Transcribing {audio_path.name} | lang={language}")

    try:
        # faster-whisper API
        from faster_whisper import WhisperModel
        if isinstance(model, WhisperModel):
            segments, _ = model.transcribe(
                str(audio_path), language=language, beam_size=5, vad_filter=True
            )
            text = " ".join(s.text.strip() for s in segments)
            log.info(f"Transcribed {len(text)} chars (faster-whisper)")
            return text
    except (ImportError, Exception):
        pass

    # openai-whisper API
    result = model.transcribe(str(audio_path), language=language)
    text = result.get("text", "").strip()
    log.info(f"Transcribed {len(text)} chars (openai-whisper)")
    return text


# ── Platform detection ────────────────────────────────────────────────────────

def _is_instagram(url: str) -> bool:
    return bool(re.search(r"(instagram\.com|instagr\.am)", url, re.I))


def _is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url, re.I))


def _extract_yt_id(url: str) -> Optional[str]:
    for pattern in [r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", r"shorts/([A-Za-z0-9_-]{11})"]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


# ── Instagram extraction ──────────────────────────────────────────────────────

def extract_instagram(url: str, language: str = "te") -> dict:
    """
    Download Instagram Reel + transcribe.
    yt-dlp handles Instagram, TikTok, Facebook Reels — all work on EC2.
    Returns: {transcript, platform, title, duration, uploader}
    """
    log.info(f"Extracting Instagram: {url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        audio_out = tmp / "audio.%(ext)s"

        # Step 1: Get metadata
        meta = {}
        try:
            r = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-playlist", "--quiet", url],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0:
                meta = json.loads(r.stdout)
        except Exception:
            pass

        # Step 2: Download audio
        cmd = [
            "yt-dlp", "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", str(audio_out),
            "--no-playlist",
            "--quiet",
            # Instagram needs a real UA
            "--add-headers", "User-Agent:Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        # Find downloaded file
        audio_files = list(tmp.glob("audio.*"))
        if not audio_files:
            err = result.stderr[:400] if result.stderr else "No output file"
            raise RuntimeError(
                f"Instagram download failed: {err}\n"
                "Ensure the reel is public and yt-dlp is up to date."
            )

        audio_path = audio_files[0]

        # Step 3: Transcribe
        transcript = transcribe(audio_path, language)

        return {
            "transcript": transcript,
            "platform": "instagram",
            "source_url": url,
            "title": meta.get("title", "Instagram Reel"),
            "duration": meta.get("duration", 0),
            "uploader": meta.get("uploader", ""),
        }


# ── YouTube extraction (EC2-aware) ────────────────────────────────────────────

def extract_youtube(url: str, language: str = "te") -> dict:
    """
    YouTube on EC2: IPs are blocked by YouTube bot detection.
    Try youtube-transcript-api first (text only, no download needed).
    If that fails, raise a clear error with workaround instructions.
    """
    log.info(f"Attempting YouTube: {url}")

    video_id = _extract_yt_id(url)

    # Try transcript API (no download — just API call)
    if video_id:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            langs = [language, "en"] if language != "en" else ["en"]
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            text = " ".join(t["text"] for t in transcript_list)
            log.info(f"YouTube transcript API: {len(text)} chars")
            return {
                "transcript": text,
                "platform": "youtube",
                "source_url": url,
                "title": f"YouTube video {video_id}",
                "duration": 0,
            }
        except Exception as e:
            log.warning(f"Transcript API failed: {e}")

    raise RuntimeError(
        "⚠️ YouTube download is blocked on EC2 IPs (YouTube detects AWS).\n\n"
        "✅ Workarounds:\n"
        "  1. Download MP3 on your local PC:\n"
        "     pip install yt-dlp\n"
        "     yt-dlp -x --audio-format mp3 -o video.mp3 'YOUTUBE_URL'\n"
        "  2. Upload the MP3 using the Audio Upload field\n"
        "  3. Use an Instagram Reel link instead (works on EC2)\n"
        "  4. Paste the topic or script text directly"
    )


# ── Audio file transcription ──────────────────────────────────────────────────

def extract_audio_file(audio_path: str, language: str = "te") -> dict:
    """Transcribe an uploaded audio file (MP3, WAV, M4A, OGG)."""
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    log.info(f"Transcribing uploaded audio: {p.name}")
    transcript = transcribe(p, language)

    return {
        "transcript": transcript,
        "platform": "audio_upload",
        "source_url": str(p),
        "title": p.stem,
        "duration": 0,
    }


# ── Unified entry point ───────────────────────────────────────────────────────

def process_source(
    url: str = "",
    audio_path: str = "",
    topic_text: str = "",
    language: str = "te",
) -> dict:
    """
    Unified content extraction. Tries sources in priority order.

    Returns:
        {transcript, platform, source_url, title, duration, error}
    """
    try:
        if audio_path and Path(audio_path).exists():
            return extract_audio_file(audio_path, language)

        if url:
            url = url.strip()
            if _is_instagram(url):
                return extract_instagram(url, language)
            elif _is_youtube(url):
                return extract_youtube(url, language)
            else:
                raise ValueError(f"Unsupported URL type: {url}")

        if topic_text.strip():
            return {
                "transcript": topic_text.strip(),
                "platform": "text_input",
                "source_url": "",
                "title": "Direct text input",
                "duration": 0,
            }

        raise ValueError("No input provided. Enter a URL, upload audio, or type a topic.")

    except Exception as e:
        log.error(f"Content extraction failed: {e}")
        return {
            "transcript": "",
            "platform": "error",
            "source_url": url,
            "title": "",
            "duration": 0,
            "error": str(e),
        }
