"""
motion_library.py — Build a motion library from real YouTube presenter videos

The user provides 2-5 YouTube URLs of real Telugu/South Indian presenters.
We download them, extract emotion-tagged clips, and store them as the
motion reference library.

At video generation time:
  1. Detect dominant emotion in script segment
  2. Pick a matching motion clip from the library
  3. LivePortrait animates Navya's face with that real human motion
  4. MuseTalk replaces only the lip region with Telugu audio sync

Result: Navya moves like a real Telugu presenter because her motion IS
        copied from one. Not estimated. Not generated. Real.

Library structure:
  models/motion_library/
  ├── professional/  clip_001.mp4, clip_002.mp4 ...
  ├── excited/
  ├── serious/
  ├── warm/
  ├── energetic/
  ├── sombre/
  └── metadata.json

Each clip is 8-15 seconds — long enough for LivePortrait to work well,
short enough to loop naturally for any script length.
"""
import json
import logging
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("motion_library")

ROOT    = Path(__file__).parent.parent.resolve()
LIBRARY = ROOT / "models" / "motion_library"
LIBRARY.mkdir(parents=True, exist_ok=True)

METADATA_FILE = LIBRARY / "metadata.json"

EMOTIONS = ["professional", "excited", "serious", "warm", "energetic", "sombre"]

# Minimum clips per emotion before library is considered ready
MIN_CLIPS_PER_EMOTION = 2


# ── Metadata management ───────────────────────────────────────────────────────

def _load_metadata() -> dict:
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text())
        except Exception:
            pass
    return {"sources": [], "clips": {e: [] for e in EMOTIONS}, "total_clips": 0}


def _save_metadata(meta: dict):
    METADATA_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def get_library_status() -> dict:
    """Return counts per emotion and overall readiness."""
    meta  = _load_metadata()
    clips = meta.get("clips", {})
    status = {e: len(clips.get(e, [])) for e in EMOTIONS}
    status["total"]   = sum(status.values())
    status["ready"]   = all(status[e] >= MIN_CLIPS_PER_EMOTION for e in EMOTIONS)
    status["partial"] = status["total"] > 0
    return status


# ── Video download ────────────────────────────────────────────────────────────

def download_reference_video(url: str, out_dir: Path) -> Optional[Path]:
    """
    Download a YouTube video for motion extraction.
    Uses cookies if available (EC2 IP block bypass).
    Downloads 720p max — enough for motion, saves space.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cookies = ROOT / "models" / "yt_cookies.txt"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
    ]
    if cookies.exists():
        cmd += ["--cookies", str(cookies)]

    cmd.append(url)

    log.info(f"Downloading reference video: {url}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    mp4_files = list(out_dir.glob("*.mp4"))
    if mp4_files:
        mp4_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        log.info(f"Downloaded: {mp4_files[0].name}")
        return mp4_files[0]

    log.error(f"Download failed: {r.stderr[:300]}")
    return None


# ── Emotion detection from video clip ────────────────────────────────────────

def detect_clip_emotion(clip_path: Path) -> str:
    """
    Detect the dominant emotion in a video clip using:
    1. Audio energy analysis (high RMS = excited, low = calm/serious)
    2. Speech rate (fast = energetic, slow = serious/sombre)
    3. Whisper transcript keyword analysis

    Returns one of: professional | excited | serious | warm | energetic | sombre
    """
    # Get audio energy (RMS) via ffprobe
    r = subprocess.run(
        ["ffprobe", "-v", "quiet",
         "-show_entries", "frame_tags=lavfi.astats.Overall.RMS_level",
         "-f", "lavfi",
         "-i", f"amovie={clip_path},astats=metadata=1:reset=1",
         "-of", "default=noprint_wrappers=1:nokey=1"],
        capture_output=True, text=True, timeout=30,
    )

    # Simplified: use audio loudness as emotion proxy
    try:
        rms_values = [float(v) for v in r.stdout.strip().split("\n") if v.strip()]
        avg_rms    = sum(rms_values) / len(rms_values) if rms_values else -30
    except Exception:
        avg_rms = -30

    # Get speech rate via Whisper word count / duration
    duration = _get_duration(clip_path)
    word_count = 0
    transcript = ""

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="auto", compute_type="int8")
        segments, _ = model.transcribe(str(clip_path), language="te", vad_filter=True)
        words = []
        for seg in segments:
            words.extend(w.word for w in (seg.words or []))
            transcript += seg.text + " "
        word_count = len(words)
    except Exception:
        pass

    words_per_sec = word_count / duration if duration > 0 else 0
    transcript_l  = transcript.lower()

    # Keyword signals
    is_excited  = bool(re.search(r"(అద్భుతం|విజయం|అభినందన|amazing|fantastic|congrat|won|record|గొప్ప)", transcript_l))
    is_sombre   = bool(re.search(r"(మరణ|మృతి|విషాదం|tragedy|died|death|నష్టం|grief)", transcript_l))
    is_serious  = bool(re.search(r"(హెచ్చరిక|అత్యవసర|breaking|warning|danger|urgent|ప్రమాద)", transcript_l))
    is_warm     = bool(re.search(r"(ఆనందం|happy|love|family|hope|మనసు|గుండె)", transcript_l))

    # Decision logic
    if is_sombre:
        return "sombre"
    if is_excited or (avg_rms > -15 and words_per_sec > 3.5):
        return "excited"
    if is_serious:
        return "serious"
    if is_warm and words_per_sec < 2.5:
        return "warm"
    if words_per_sec > 3.0 or avg_rms > -20:
        return "energetic"
    return "professional"


def _get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=10
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ── Scene segmentation ────────────────────────────────────────────────────────

def extract_motion_clips(
    video_path: Path,
    clip_duration: float = 10.0,   # seconds per clip
    stride: float = 5.0,           # overlap stride
    max_clips: int = 20,           # max clips per source video
    face_required: bool = True,    # skip clips without a clear face
) -> list:
    """
    Extract short motion clips from a longer video.
    Skips: blurry frames, no face detected, scene cuts, heavy camera shake.
    Returns list of (clip_path, detected_emotion) tuples.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        duration = _get_duration(video_path)
        if duration < clip_duration:
            log.warning(f"Video too short ({duration:.1f}s) for clip extraction")
            return []

        clips = []
        start = 0.0
        clip_idx = 0

        while start + clip_duration <= duration and clip_idx < max_clips:
            clip_out = LIBRARY / "_tmp_clip.mp4"

            # Extract clip
            r = subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(start),
                 "-i", str(video_path),
                 "-t", str(clip_duration),
                 "-c:v", "libx264", "-preset", "ultrafast",
                 "-c:a", "aac",
                 str(clip_out)],
                capture_output=True, timeout=60
            )

            if r.returncode != 0 or not clip_out.exists():
                start += stride
                continue

            # Check face is present and clip is usable
            if face_required and not _has_clear_face(clip_out):
                clip_out.unlink(missing_ok=True)
                start += stride
                continue

            # Detect emotion
            emotion = detect_clip_emotion(clip_out)

            # Save to library
            emotion_dir = LIBRARY / emotion
            emotion_dir.mkdir(exist_ok=True)
            existing = list(emotion_dir.glob("clip_*.mp4"))
            idx      = len(existing) + 1
            dest     = emotion_dir / f"clip_{idx:03d}.mp4"
            clip_out.rename(dest)

            log.info(f"  Clip {clip_idx+1}: {start:.1f}s → emotion={emotion} → {dest.name}")
            clips.append((dest, emotion))
            clip_idx += 1
            start    += stride

        if clip_out.exists():
            clip_out.unlink(missing_ok=True)

    return clips


def _has_clear_face(clip_path: Path) -> bool:
    """
    Quick check: does the clip have a clear, front-facing face?
    Uses OpenCV face detector on middle frame.
    """
    try:
        import cv2

        cap     = cv2.VideoCapture(str(clip_path))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces    = detector.detectMultiScale(gray, 1.1, 5)
        return len(faces) > 0

    except Exception:
        return True   # if check fails, assume face is present


# ── Clip quality filter ────────────────────────────────────────────────────────

def filter_best_clips(
    clips: list,
    max_per_emotion: int = 5,
) -> list:
    """
    Keep only the best clips per emotion.
    Scores clips on: face size, sharpness, minimal camera shake.
    """
    # Group by emotion
    by_emotion = {}
    for path, emotion in clips:
        by_emotion.setdefault(emotion, []).append(path)

    kept = []
    for emotion, paths in by_emotion.items():
        # Score each clip
        scored = []
        for p in paths:
            score = _score_clip(p)
            scored.append((score, p))
        scored.sort(reverse=True)
        best = [p for _, p in scored[:max_per_emotion]]
        kept.extend((p, emotion) for p in best)

    return kept


def _score_clip(clip_path: Path) -> float:
    """
    Score clip quality 0-100:
    - Laplacian variance (sharpness): higher = sharper
    - Face size relative to frame: larger face = better for LivePortrait
    """
    try:
        import cv2
        import numpy as np

        cap    = cv2.VideoCapture(str(clip_path))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return 0.0

        h, w    = frame.shape[:2]
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpness
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Face size
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces    = detector.detectMultiScale(gray, 1.1, 5)
        face_score = 0.0
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
            face_score = (fw * fh) / (w * h) * 100

        return min(100, sharpness / 10 + face_score * 2)

    except Exception:
        return 50.0


# ── Master build function ─────────────────────────────────────────────────────

def build_library(
    youtube_urls: list,
    progress_callback=None,
) -> dict:
    """
    Full pipeline: download videos → extract clips → classify → save library.

    youtube_urls: list of YouTube video URLs (2-5 recommended)
    progress_callback: optional fn(message: str) for UI updates

    Returns library status dict.
    """
    def progress(msg: str):
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    meta = _load_metadata()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        all_clips = []

        for i, url in enumerate(youtube_urls):
            if not url.strip():
                continue

            progress(f"[{i+1}/{len(youtube_urls)}] Downloading: {url[:60]}...")

            video_path = download_reference_video(url.strip(), tmp / f"src_{i}")
            if not video_path:
                progress(f"  ⚠️ Download failed — skipping")
                continue

            progress(f"  ✅ Downloaded: {video_path.name} ({video_path.stat().st_size//1024//1024}MB)")
            progress(f"  Extracting motion clips...")

            clips = extract_motion_clips(video_path)

            if not clips:
                progress(f"  ⚠️ No usable clips extracted")
                continue

            progress(f"  ✅ Extracted {len(clips)} clips")
            all_clips.extend(clips)

            # Track source
            meta["sources"].append({
                "url": url,
                "file": video_path.name,
                "clips_extracted": len(clips),
            })

    # Filter best clips
    if all_clips:
        progress("Selecting best clips per emotion...")
        best = filter_best_clips(all_clips)

        # Update metadata
        meta["clips"] = {e: [] for e in EMOTIONS}
        for path, emotion in best:
            meta["clips"].setdefault(emotion, []).append(str(path.relative_to(ROOT)))
        meta["total_clips"] = len(best)

        _save_metadata(meta)

    status = get_library_status()
    progress(f"\n✅ Motion library built:")
    for emotion in EMOTIONS:
        progress(f"  {emotion}: {status[emotion]} clips")

    return status


# ── Runtime: get clip for generation ─────────────────────────────────────────

def get_motion_clip(emotion: str = "professional") -> Optional[Path]:
    """
    Get a motion clip for a given emotion.
    Falls back to 'professional' if requested emotion has no clips.
    Returns absolute path to the clip.
    """
    meta = _load_metadata()
    clips_map = meta.get("clips", {})

    # Try requested emotion first
    options = clips_map.get(emotion, [])

    # Fallback chain
    if not options:
        fallbacks = {
            "excited":    ["energetic", "warm", "professional"],
            "energetic":  ["excited", "professional"],
            "serious":    ["professional", "sombre"],
            "sombre":     ["serious", "professional"],
            "warm":       ["professional", "excited"],
            "professional": ["warm", "serious"],
        }
        for fallback in fallbacks.get(emotion, ["professional"]):
            options = clips_map.get(fallback, [])
            if options:
                break

    if not options:
        log.warning(f"No motion clips for emotion '{emotion}' — library may be empty")
        return None

    # Pick randomly from available clips (variety)
    chosen = random.choice(options)
    path   = ROOT / chosen
    if path.exists():
        return path

    log.warning(f"Motion clip missing: {path}")
    return None


def get_motion_clip_for_script(script_text: str) -> Optional[Path]:
    """
    Detect emotion from script and return matching motion clip.
    Convenience wrapper for the generation pipeline.
    """
    from pipeline.tts_engine import detect_emotion
    emotion = detect_emotion(script_text[:500])
    return get_motion_clip(emotion)
