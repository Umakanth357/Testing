"""
audio_post.py — Broadcast-quality audio post-processing

Takes raw TTS output and applies a professional broadcast processing chain:

  Raw TTS → High-pass filter → De-esser → Compressor → Presence EQ →
  Limiter → Loudness normalisation → Breath sounds → Room acoustics

This is the difference between "AI voice reading text" and "professional
radio anchor delivering news". The same chain used in radio broadcast.

Why each step:
  high-pass:  Remove low-frequency rumble below 80Hz (no voice content there)
  de-esser:   Tame harsh "S" sounds that TTS over-emphasises
  compressor: Even out loud/soft sections — broadcaster sounds consistent
  presence:   Boost 3-4kHz (the "presence band") — adds clarity and authority
  limiter:    Hard ceiling at -1dBFS — prevents clipping in downstream processing
  loudnorm:   Normalise to -14 LUFS (broadcast standard, matches YouTube/Instagram)
  breath:     Add subtle inhalation sounds at sentence starts — human tell
  room:       Very slight room tone — removes the "anechoic chamber" AI sound

All processing via FFmpeg only — no additional Python dependencies.
"""
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("audio_post")


# ── Core broadcast chain ──────────────────────────────────────────────────────

def process_broadcast(
    in_path: Path,
    out_path: Path,
    anchor_style: str = "tv_news",   # 'tv_news' | 'radio' | 'documentary' | 'podcast'
) -> bool:
    """
    Apply broadcast-quality audio processing chain.

    anchor_style options:
      'tv_news'    — Clean, authoritative, slight presence boost
      'radio'      — Warmer, more compression, boosted low-mids
      'documentary'— Intimate, less aggressive compression
      'podcast'    — Natural, minimal processing

    Returns True if processing succeeded.
    """
    in_path  = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Style-specific EQ/compression parameters
    STYLES = {
        "tv_news": {
            "hp_freq": 85,          # Hz — high-pass cutoff
            "presence_freq": 3500,  # Hz — presence boost frequency
            "presence_gain": 4,     # dB — presence boost amount
            "compress_thresh": -18, # dB
            "compress_ratio": 4,    # 4:1 compression
            "room_delay": 0.08,     # seconds of room delay
            "room_decay": 0.5,      # room decay amount
            "target_lufs": -14.0,   # broadcast standard
        },
        "radio": {
            "hp_freq": 100,
            "presence_freq": 2800,
            "presence_gain": 5,
            "compress_thresh": -20,
            "compress_ratio": 5,
            "room_delay": 0.05,
            "room_decay": 0.3,
            "target_lufs": -14.0,
        },
        "documentary": {
            "hp_freq": 70,
            "presence_freq": 4000,
            "presence_gain": 2,
            "compress_thresh": -15,
            "compress_ratio": 3,
            "room_delay": 0.1,
            "room_decay": 0.4,
            "target_lufs": -16.0,
        },
        "podcast": {
            "hp_freq": 80,
            "presence_freq": 3000,
            "presence_gain": 2,
            "compress_thresh": -20,
            "compress_ratio": 3,
            "room_delay": 0.06,
            "room_decay": 0.3,
            "target_lufs": -16.0,
        },
    }

    p = STYLES.get(anchor_style, STYLES["tv_news"])

    # Build FFmpeg audio filter chain
    # 1. highpass — remove low-frequency rumble
    # 2. equalizer — de-esser (notch at 7kHz, where harsh S lives)
    # 3. equalizer — presence boost
    # 4. compand — broadband compressor
    # 5. aecho — very subtle room tone
    # 6. loudnorm — broadcast loudness normalisation

    filter_chain = (
        # High-pass: remove below 80Hz (rumble, not voice)
        f"highpass=f={p['hp_freq']},"
        # Low-pass: soft roll-off above 12kHz (harsh digititis)
        "lowpass=f=12000,"
        # De-esser: reduce sibilance at 7kHz
        "equalizer=f=7000:width_type=o:width=1.5:g=-4,"
        # Presence boost: adds authority/clarity
        f"equalizer=f={p['presence_freq']}:width_type=o:width=2:g={p['presence_gain']},"
        # Compressor: even out dynamics
        f"compand="
        f"attacks=0.005:decays=0.2:"
        f"points=-80/-80|{p['compress_thresh']}/{p['compress_thresh']}|0/-{int(p['compress_ratio'])}:"
        f"gain=3,"
        # Subtle room acoustics (removes the "AI anechoic" sound)
        f"aecho=0.8:0.88:{int(p['room_delay']*1000)}:{p['room_decay']},"
        # Final limiter
        "alimiter=level_in=1:level_out=0.9:limit=0.9:attack=5:release=50:asc=1,"
        # Loudness normalisation to broadcast standard
        f"loudnorm=I={p['target_lufs']}:TP=-1.5:LRA=11"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-af", filter_chain,
        "-ar", "44100",
        "-ac", "1",       # mono for voice
        "-c:a", "pcm_s16le",
        str(out_path),
    ]

    log.info(f"Broadcast processing | style={anchor_style} | {in_path.name}")
    r = subprocess.run(cmd, capture_output=True, timeout=120)

    if r.returncode != 0:
        log.error(f"Broadcast processing failed: {r.stderr.decode()[-300:]}")
        # Fallback: just normalise loudness
        return _loudnorm_only(in_path, out_path)

    ok = out_path.exists() and out_path.stat().st_size > 0
    if ok:
        log.info(f"Broadcast processing OK → {out_path.name}")
    return ok


def _loudnorm_only(in_path: Path, out_path: Path) -> bool:
    """Minimal fallback: loudness normalisation only."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         "-ar", "44100", "-ac", "1",
         str(out_path)],
        capture_output=True, timeout=60,
    )
    return r.returncode == 0 and out_path.exists()


# ── Breath sounds injection ───────────────────────────────────────────────────

def add_breathing(
    audio_path: Path,
    out_path: Path,
    interval_approx_sec: float = 6.0,   # approx seconds between breath sounds
    breath_volume: float = 0.08,         # 0.0-1.0, very subtle
) -> bool:
    """
    Synthesise and inject subtle breath sounds at sentence boundaries.
    Breath sounds are generated using FFmpeg's anoisesrc + envelope.
    This adds a subtle but critical human tell — AI TTS never breathes.

    breath_volume: 0.05-0.15 is barely audible but effective subconsciously
    """
    audio_path = Path(audio_path)
    out_path   = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get audio duration
    dur_r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True, timeout=15,
    )
    try:
        duration = float(dur_r.stdout.strip())
    except (ValueError, AttributeError):
        log.warning("Could not get audio duration — skipping breath sounds")
        import shutil; shutil.copy2(str(audio_path), str(out_path))
        return True

    if duration < 3.0:
        import shutil; shutil.copy2(str(audio_path), str(out_path))
        return True

    # Generate breath positions (irregular timing, like real speech)
    positions = []
    t = interval_approx_sec * 0.7  # first breath fairly early
    while t < duration - 1.0:
        positions.append(t)
        # Vary interval by ±30% to avoid mechanical rhythm
        jitter = random.uniform(0.7, 1.3)
        t += interval_approx_sec * jitter

    if not positions:
        import shutil; shutil.copy2(str(audio_path), str(out_path))
        return True

    # Build FFmpeg filter: synthesise a breath sound at each position
    # Breath: short pink noise burst (0.2s) with fast attack + decay envelope
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        breath_clips = []

        for i, pos in enumerate(positions):
            bc = tmp / f"breath_{i}.wav"
            # Pink noise, 0.15s, enveloped to sound like an inhale
            r = subprocess.run(
                ["ffmpeg", "-y",
                 "-f", "lavfi", "-i",
                 "anoisesrc=color=pink:duration=0.18:amplitude=0.3",
                 "-af",
                 "afade=t=in:ss=0:d=0.04,afade=t=out:st=0.10:d=0.08,"
                 "highpass=f=200,lowpass=f=3000,"
                 f"volume={breath_volume}",
                 str(bc)],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                breath_clips.append((pos, bc))

        if not breath_clips:
            import shutil; shutil.copy2(str(audio_path), str(out_path))
            return True

        # Mix breath clips into main audio at their positions
        # Build amix filter
        inputs = ["-i", str(audio_path)]
        for _, bc in breath_clips:
            inputs += ["-i", str(bc)]

        # Use amerge + delay per breath
        filter_parts = []
        for idx, (pos, _) in enumerate(breath_clips):
            delay_ms = int(pos * 1000)
            filter_parts.append(
                f"[{idx+1}:a]adelay={delay_ms}|{delay_ms}[breath{idx}]"
            )

        streams = "[0:a]" + "".join(f"[breath{i}]" for i in range(len(breath_clips)))
        n = 1 + len(breath_clips)
        filter_parts.append(f"{streams}amix=inputs={n}:duration=first:normalize=0[out]")
        filter_str = ";".join(filter_parts)

        mix_cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + ["-filter_complex", filter_str,
               "-map", "[out]",
               "-ar", "44100", "-ac", "1",
               str(out_path)]
        )

        r2 = subprocess.run(mix_cmd, capture_output=True, timeout=120)
        if r2.returncode != 0:
            log.warning(f"Breath mixing failed — using clean audio: {r2.stderr.decode()[-200:]}")
            import shutil; shutil.copy2(str(audio_path), str(out_path))
            return True

    log.info(f"Added {len(breath_clips)} breath sounds → {out_path.name}")
    return True


# ── Emotion-aware pitch/rate adjustment ──────────────────────────────────────

def adjust_for_emotion(
    audio_path: Path,
    out_path: Path,
    emotion: str = "professional",
) -> bool:
    """
    Fine-tune pitch and tempo for emotional delivery.
    edge-tts supports rate/pitch at synthesis time; this handles
    post-synthesis adjustments for emotions that weren't settable.
    """
    audio_path = Path(audio_path)
    out_path   = Path(out_path)

    # Emotion → (pitch shift semitones, tempo multiplier)
    EMOTION_PARAMS = {
        "excited":      (1.5, 1.08),   # slightly higher, faster
        "energetic":    (1.0, 1.05),
        "professional": (0.0, 1.00),   # no change
        "calm":         (-0.5, 0.95),  # slightly lower, slower
        "serious":      (-1.0, 0.93),  # lower, deliberate
        "warm":         (0.0, 0.97),   # same pitch, slightly slower
        "sombre":       (-1.5, 0.88),  # lower, slow
    }

    semitones, tempo = EMOTION_PARAMS.get(emotion, (0.0, 1.00))

    # Build filter only if adjustments are needed
    filters = []
    if abs(semitones) > 0.1:
        # rubberband for high-quality pitch shift (if available)
        # fallback: asetrate trick (lower quality but no deps)
        try:
            r_check = subprocess.run(
                ["ffmpeg", "-filters"], capture_output=True, text=True
            )
            if "rubberband" in r_check.stdout:
                filters.append(f"rubberband=pitch={2**(semitones/12):.4f}")
            else:
                rate = int(44100 * (2 ** (semitones / 12)))
                filters.append(f"asetrate={rate},aresample=44100")
        except Exception:
            pass

    if abs(tempo - 1.0) > 0.01:
        filters.append(f"atempo={tempo:.3f}")

    if not filters:
        import shutil; shutil.copy2(str(audio_path), str(out_path))
        return True

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-af", ",".join(filters),
        "-ar", "44100", "-ac", "1",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        import shutil; shutil.copy2(str(audio_path), str(out_path))
    return True


# ── Master function ───────────────────────────────────────────────────────────

def master_audio(
    raw_tts_path: Path,
    out_path: Path,
    emotion: str = "professional",
    anchor_style: str = "tv_news",
    add_breaths: bool = True,
) -> Path:
    """
    Full audio post-processing pipeline:
      raw TTS → emotion adjust → broadcast chain → [breath sounds] → final

    Returns the processed audio path (out_path).
    """
    raw_tts_path = Path(raw_tts_path)
    out_path     = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        step1 = tmp / "emotion.wav"
        step2 = tmp / "broadcast.wav"

        # Step 1: Emotion pitch/tempo adjustment
        adjust_for_emotion(raw_tts_path, step1, emotion)

        # Step 2: Broadcast processing chain
        src = step1 if step1.exists() else raw_tts_path
        process_broadcast(src, step2, anchor_style)

        # Step 3: Add breathing sounds
        src2 = step2 if step2.exists() else src
        if add_breaths:
            add_breathing(src2, out_path)
        else:
            import shutil; shutil.copy2(str(src2), str(out_path))

    log.info(f"Master audio complete → {out_path.name}")
    return out_path
