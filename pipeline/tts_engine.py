"""
tts_engine.py — Text-to-Speech pipeline

Priority chain:
  1. edge-tts (Microsoft Azure Neural)  — PRIMARY. Fast, natural, Telugu/Hindi/Tamil/Kannada.
                                          No API key. Free. Async HTTP. ~2-5s for 3min audio.
  2. Indic Parler-TTS (ai4bharat)       — Best quality. 1806h Telugu, 6 emotion params.
                                          Needs HF gated access approval + 4GB model.
  3. gTTS                               — Last resort. Always works. Robotic but reliable.

Speed: edge-tts produces 3min of audio in ~3 seconds. It is the right choice.

Post-processing (always applied):
  - Normalise loudness to -14 LUFS (broadcast standard)
  - Optional room acoustics (pedalboard reverb)
"""
import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("tts_engine")

# ── Voice map: lang × gender → Microsoft Neural voice ────────────────────────
EDGE_VOICES = {
    "te": {"female": "te-IN-ShrutiNeural",  "male": "te-IN-MohanNeural"},
    "hi": {"female": "hi-IN-SwaraNeural",   "male": "hi-IN-MadhurNeural"},
    "ta": {"female": "ta-IN-PallaviNeural", "male": "ta-IN-ValluvarNeural"},
    "kn": {"female": "kn-IN-SapnaNeural",   "male": "kn-IN-GaganNeural"},
    "ml": {"female": "ml-IN-SobhanaNeural", "male": "ml-IN-MidhunNeural"},
    "mr": {"female": "mr-IN-AarohiNeural",  "male": "mr-IN-ManoharNeural"},
    "bn": {"female": "bn-IN-TanishaaNeural","male": "bn-IN-BashkarNeural"},
    "en": {"female": "en-US-JennyNeural",   "male": "en-US-GuyNeural"},
}

# Voice profile → (lang, gender, speaking_rate_offset, pitch_offset_hz)
PROFILE_CONFIG = {
    "te_female_professional": ("te", "female", "+0%",  "+0Hz"),
    "te_male_professional":   ("te", "male",   "-5%",  "-5Hz"),
    "ta_male_professional":   ("ta", "male",   "-5%",  "-3Hz"),
    "kn_female_professional": ("kn", "female", "+0%",  "+0Hz"),
    "hi_female_professional": ("hi", "female", "+0%",  "+0Hz"),
    "en_male_professional":   ("en", "male",   "-5%",  "-5Hz"),
    "en_female_professional": ("en", "female", "+0%",  "+0Hz"),
}


# ── edge-tts (PRIMARY) ────────────────────────────────────────────────────────

async def _edge_async(text: str, voice: str, mp3_path: Path, rate: str, pitch: str) -> bool:
    """Core async edge-tts call. Saves MP3, returns success."""
    try:
        import edge_tts
        comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
        await comm.save(str(mp3_path))
        return mp3_path.exists() and mp3_path.stat().st_size > 1024
    except Exception as e:
        log.warning(f"edge-tts async failed: {e}")
        return False


def _synth_edge(text: str, profile: str, out_wav: Path) -> bool:
    """Synthesize via edge-tts. MP3 → WAV via FFmpeg."""
    lang, gender, rate, pitch = PROFILE_CONFIG.get(
        profile, ("te", "female", "+0%", "+0Hz")
    )
    voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])[gender]
    mp3_path = out_wav.with_suffix(".mp3")
    log.info(f"edge-tts | voice={voice} rate={rate}")

    ok = asyncio.run(_edge_async(text, voice, mp3_path, rate, pitch))
    if not ok:
        return False

    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "44100", "-ac", "1", str(out_wav)],
        capture_output=True, timeout=60,
    )
    mp3_path.unlink(missing_ok=True)
    success = r.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 1024
    if not success:
        log.error(f"FFmpeg MP3→WAV: {r.stderr.decode()[:200]}")
    return success


# ── Indic Parler-TTS (OPTIONAL HIGH QUALITY) ──────────────────────────────────

_PARLER_MODEL = None
_PARLER_TOKENIZER = None

_PARLER_DESCRIPTIONS = {
    "te_female_professional": (
        "A warm, confident South Indian Telugu woman speaks professionally. "
        "Clear pronunciation, engaging tone, studio quality."
    ),
    "te_male_professional": (
        "A deep, authoritative Telugu man speaks calmly and professionally. "
        "Moderate pace, clear and articulate, studio quality."
    ),
}


def _load_parler() -> bool:
    global _PARLER_MODEL, _PARLER_TOKENIZER
    if _PARLER_MODEL is not None:
        return True
    try:
        import torch
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration
        model_id = "ai4bharat/indic-parler-tts"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _PARLER_MODEL = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        _PARLER_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        log.info(f"Parler-TTS loaded on {device}")
        return True
    except Exception as e:
        log.warning(f"Parler-TTS unavailable: {e}")
        return False


def _synth_parler(text: str, profile: str, out_wav: Path) -> bool:
    if not _load_parler():
        return False
    try:
        import torch, soundfile as sf
        device = "cuda" if torch.cuda.is_available() else "cpu"
        desc = _PARLER_DESCRIPTIONS.get(
            profile, "A professional voice speaking clearly. Studio quality."
        )
        inp = _PARLER_TOKENIZER(desc, return_tensors="pt").input_ids.to(device)
        pmt = _PARLER_TOKENIZER(text, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            gen = _PARLER_MODEL.generate(input_ids=inp, prompt_input_ids=pmt)
        audio = gen.cpu().numpy().squeeze()
        sf.write(str(out_wav), audio, _PARLER_MODEL.config.sampling_rate)
        log.info(f"Parler-TTS OK → {out_wav}")
        return True
    except Exception as e:
        log.error(f"Parler-TTS synthesis: {e}")
        return False


# ── gTTS (LAST RESORT) ────────────────────────────────────────────────────────

def _synth_gtts(text: str, profile: str, out_wav: Path) -> bool:
    try:
        from gtts import gTTS
        lang = PROFILE_CONFIG.get(profile, ("te",))[0]
        mp3 = out_wav.with_suffix(".mp3")
        gTTS(text=text, lang=lang, slow=False).save(str(mp3))
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3), "-ar", "44100", "-ac", "1", str(out_wav)],
            capture_output=True, timeout=60,
        )
        mp3.unlink(missing_ok=True)
        log.info(f"gTTS fallback OK → {out_wav}")
        return out_wav.exists() and out_wav.stat().st_size > 1024
    except Exception as e:
        log.error(f"gTTS failed: {e}")
        return False


# ── Pre-processing ────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean script text for TTS: remove stage directions, speaker labels, normalise whitespace."""
    text = re.sub(r'^(NAVYA|ARJUN|SPEAKER_[AB]):\s*', '', text, flags=re.MULTILINE | re.I)
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)
    text = text.replace('...', ',')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Post-processing ───────────────────────────────────────────────────────────

def normalise_loudness(wav_in: Path, wav_out: Path, target_lufs: float = -14.0) -> bool:
    """Normalise loudness to -14 LUFS (broadcast standard) via FFmpeg loudnorm."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_in),
             "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
             "-ar", "44100", "-ac", "1", str(wav_out)],
            capture_output=True, timeout=120,
        )
        return r.returncode == 0 and wav_out.exists()
    except Exception as e:
        log.warning(f"Loudnorm failed: {e}")
        return False


def add_room_acoustics(wav_in: Path, reverb_type: str, wav_out: Path) -> bool:
    """Add subtle room reverb via FFmpeg aecho filter."""
    PARAMS = {
        "dead":        None,
        "small_room":  "0.8:0.88:60:0.4",
        "medium_room": "0.8:0.88:100:0.5",
        "large_hall":  "0.8:0.88:200:0.7",
        "outdoors":    "0.8:0.88:80:0.3",
    }
    params = PARAMS.get(reverb_type)
    if params is None:
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_in), str(wav_out)], capture_output=True)
        return wav_out.exists()
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_in), "-af", f"aecho={params}", str(wav_out)],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0 and wav_out.exists()
    except Exception as e:
        log.warning(f"Room acoustics failed: {e}")
        import shutil; shutil.copy2(str(wav_in), str(wav_out))
        return True


# ── Main synthesize() ─────────────────────────────────────────────────────────

def synthesize(
    text: str,
    voice_profile: str,
    out_path: Path,
    ref_audio: Optional[str] = None,
    use_parler: bool = False,
) -> bool:
    """
    Main synthesis entrypoint.
    Chain: edge-tts (primary) → Parler-TTS (if use_parler) → gTTS (fallback).
    Returns True if audio generated successfully.
    """
    if not text.strip():
        log.error("synthesize: empty text")
        return False

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = preprocess_text(text)
    raw = out_path.with_stem(out_path.stem + "_raw")

    success = False

    if use_parler:
        log.info("Trying Parler-TTS (high quality)...")
        success = _synth_parler(clean, voice_profile, raw)

    if not success:
        log.info("Trying edge-tts (Microsoft Neural)...")
        success = _synth_edge(clean, voice_profile, raw)

    if not success:
        log.warning("edge-tts failed — gTTS fallback")
        success = _synth_gtts(clean, voice_profile, raw)

    if not success:
        log.error("All TTS engines failed")
        return False

    ok = normalise_loudness(raw, out_path)
    if not ok:
        import shutil; shutil.copy2(str(raw), str(out_path))

    raw.unlink(missing_ok=True)
    final_ok = out_path.exists() and out_path.stat().st_size > 1024
    if final_ok:
        log.info(f"TTS done: {out_path} ({out_path.stat().st_size//1024}KB)")
    return final_ok
