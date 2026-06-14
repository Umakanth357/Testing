"""
TTS Engine — Text to speech with emotion, breathing, and voice cloning.

Priority chain:
  Indic languages: IndicF5 → Coqui XTTS v2 → gTTS
  English:         Chatterbox → Coqui XTTS v2 → gTTS
"""
import re
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from config import VOICES_DIR, VOICE_PROFILES, INDIC_TTS_MODEL, DEVICE

log = logging.getLogger("tts_engine")

# Silence durations (ms)
BREATH_PAUSE_MS  = 350   # [breath] marker
COMMA_PAUSE_MS   = 180
PERIOD_PAUSE_MS  = 400
ELLIPSIS_PAUSE_MS = 600


# ── Public API ────────────────────────────────────────────────────────────────

def synthesize(text: str, voice_profile: str, out_path: Path,
               ref_audio: Optional[Path] = None) -> bool:
    """
    Main TTS entry point.
    Returns True if audio was generated successfully, False otherwise.
    """
    profile = VOICE_PROFILES.get(voice_profile)
    if not profile:
        log.error(f"Unknown voice profile: {voice_profile}")
        return False

    # Inject breathing pauses and normalise text
    prepared = _prepare_text(text)
    segments  = _split_into_segments(prepared)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tts_"))

    engine  = profile["engine"]
    lang    = profile["lang"]

    # Auto-select reference audio from voices dir if not provided
    if ref_audio is None:
        candidates = list(VOICES_DIR.glob(f"{lang}_*.wav")) + list(VOICES_DIR.glob(f"*.wav"))
        if candidates:
            ref_audio = candidates[0]
            log.info(f"Using voice reference: {ref_audio.name}")

    segment_files = []
    for i, seg in enumerate(segments):
        if not seg.strip() or seg == "[breath]":
            # Generate silence
            seg_path = tmp_dir / f"seg_{i:04d}.wav"
            _write_silence(seg_path, BREATH_PAUSE_MS if seg == "[breath]" else COMMA_PAUSE_MS)
            segment_files.append(seg_path)
            continue

        seg_path = tmp_dir / f"seg_{i:04d}.wav"
        success = False

        if engine == "indic":
            success = _synth_indic(seg, lang, seg_path, ref_audio)
            if not success:
                success = _synth_coqui(seg, lang, seg_path, ref_audio)
            if not success:
                success = _synth_gtts(seg, lang, seg_path)

        elif engine == "chatterbox":
            success = _synth_chatterbox(seg, seg_path, ref_audio)
            if not success:
                success = _synth_coqui(seg, "en", seg_path, ref_audio)
            if not success:
                success = _synth_gtts(seg, "en", seg_path)

        else:
            success = _synth_gtts(seg, lang, seg_path)

        if success:
            segment_files.append(seg_path)
        else:
            log.error(f"All TTS engines failed for segment: {seg[:50]}")

    if not segment_files:
        log.error("No audio segments generated")
        return False

    # Concatenate all segments
    _concatenate_wavs(segment_files, out_path)
    log.info(f"TTS complete: {out_path} ({out_path.stat().st_size // 1024}KB)")
    return True


# ── Text Preparation ──────────────────────────────────────────────────────────

def _prepare_text(text: str) -> str:
    """Clean text, inject breathing markers before long sentences."""
    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove existing stage directions in brackets (we add our own)
    text = re.sub(r'\[(?!breath)[^\]]+\]', '', text)

    # Inject [breath] before sentences longer than 15 words
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for i, sent in enumerate(sentences):
        word_count = len(sent.split())
        if i > 0 and word_count > 15:
            result.append("[breath]")
        result.append(sent)

    return " ".join(result)


def _split_into_segments(text: str) -> list[str]:
    """
    Split prepared text into TTS-safe segments.
    Splits on: [breath], sentence boundaries, commas.
    """
    # Split on [breath] markers first
    parts = re.split(r'\[breath\]', text)
    segments = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        segments.append("[breath]")  # insert pause between parts
        # Split long parts on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', part)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                segments.append(sent)

    # Remove leading [breath]
    while segments and segments[0] == "[breath]":
        segments.pop(0)

    return segments


# ── IndicF5 ───────────────────────────────────────────────────────────────────

_indic_pipeline = None

def _synth_indic(text: str, lang: str, out_path: Path, ref_audio: Optional[Path]) -> bool:
    global _indic_pipeline
    try:
        if _indic_pipeline is None:
            from transformers import pipeline
            log.info("Loading IndicF5 model...")
            _indic_pipeline = pipeline(
                "text-to-speech",
                model=INDIC_TTS_MODEL,
                device=0 if DEVICE == "cuda" else -1,
            )

        # IndicF5 expects language prefix in some variants
        lang_prefixed = f"[{lang}] {text}" if lang != "en" else text

        kwargs = {}
        if ref_audio and ref_audio.exists():
            # Zero-shot voice cloning
            kwargs["forward_params"] = {"reference_audio": str(ref_audio)}

        output = _indic_pipeline(lang_prefixed, **kwargs)
        audio  = output["audio"]
        sr     = output["sampling_rate"]

        if audio.ndim > 1:
            audio = audio.squeeze()
        sf.write(str(out_path), audio, sr)
        return True
    except Exception as e:
        log.warning(f"IndicF5 failed: {e}")
        _indic_pipeline = None  # Reset so next call retries load
        return False


# ── Chatterbox ────────────────────────────────────────────────────────────────

_chatterbox_model = None

def _synth_chatterbox(text: str, out_path: Path, ref_audio: Optional[Path]) -> bool:
    global _chatterbox_model
    try:
        if _chatterbox_model is None:
            from chatterbox.tts import ChatterboxTTS
            log.info("Loading Chatterbox TTS...")
            _chatterbox_model = ChatterboxTTS.from_pretrained(device=DEVICE)

        kwargs = {"exaggeration": 0.5, "cfg_weight": 0.5}
        if ref_audio and ref_audio.exists():
            kwargs["audio_prompt_path"] = str(ref_audio)

        wav = _chatterbox_model.generate(text, **kwargs)
        sf.write(str(out_path), wav.squeeze().cpu().numpy(), _chatterbox_model.sr)
        return True
    except Exception as e:
        log.warning(f"Chatterbox failed: {e}")
        _chatterbox_model = None
        return False


# ── Coqui XTTS v2 ─────────────────────────────────────────────────────────────

_coqui_model = None

def _synth_coqui(text: str, lang: str, out_path: Path, ref_audio: Optional[Path]) -> bool:
    global _coqui_model
    try:
        if _coqui_model is None:
            from TTS.api import TTS
            log.info("Loading Coqui XTTS v2...")
            _coqui_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

        # XTTS v2 supported languages
        lang_map = {"te": "te", "kn": "kn", "ta": "ta", "hi": "hi",
                    "en": "en", "ml": "ml", "mr": "mr", "bn": "bn"}
        tts_lang = lang_map.get(lang, "en")

        if ref_audio and ref_audio.exists():
            _coqui_model.tts_to_file(
                text=text, language=tts_lang,
                speaker_wav=str(ref_audio),
                file_path=str(out_path),
            )
        else:
            _coqui_model.tts_to_file(
                text=text, language=tts_lang,
                file_path=str(out_path),
            )
        return True
    except Exception as e:
        log.warning(f"Coqui XTTS failed: {e}")
        _coqui_model = None
        return False


# ── gTTS Fallback ─────────────────────────────────────────────────────────────

def _synth_gtts(text: str, lang: str, out_path: Path) -> bool:
    try:
        from gtts import gTTS
        lang_map = {"te": "te", "kn": "kn", "ta": "ta", "hi": "hi",
                    "en": "en", "ml": "ml", "mr": "mr", "bn": "bn"}
        tts_lang = lang_map.get(lang, "en")
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        mp3_path = out_path.with_suffix(".mp3")
        tts.save(str(mp3_path))
        # Convert MP3 → WAV
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio.export(str(out_path), format="wav")
        mp3_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        log.warning(f"gTTS failed: {e}")
        return False


# ── Audio Utilities ───────────────────────────────────────────────────────────

def _write_silence(path: Path, duration_ms: int) -> None:
    silence = AudioSegment.silent(duration=duration_ms)
    silence.export(str(path), format="wav")


def _concatenate_wavs(files: list[Path], out_path: Path) -> None:
    combined = AudioSegment.empty()
    for f in files:
        try:
            seg = AudioSegment.from_wav(str(f))
            combined += seg
        except Exception as e:
            log.warning(f"Skip corrupt segment {f.name}: {e}")
    # Normalise to -16 LUFS (broadcast standard)
    combined = _normalise(combined)
    combined.export(str(out_path), format="wav")


def _normalise(audio: AudioSegment, target_dBFS: float = -16.0) -> AudioSegment:
    diff = target_dBFS - audio.dBFS
    return audio.apply_gain(diff)


def add_room_acoustics(audio_path: Path, reverb_type: str, out_path: Path) -> bool:
    """Apply room acoustics using Pedalboard. Returns True on success."""
    try:
        from pedalboard import Pedalboard, Reverb, HighpassFilter, LowpassFilter, Compressor
        import soundfile as sf

        REVERB_PRESETS = {
            "dead":        Reverb(room_size=0.0, damping=1.0, wet_level=0.0),
            "small_room":  Reverb(room_size=0.15, damping=0.7, wet_level=0.08),
            "medium_room": Reverb(room_size=0.3,  damping=0.6, wet_level=0.12),
            "large_hall":  Reverb(room_size=0.7,  damping=0.4, wet_level=0.20),
            "outdoors":    Reverb(room_size=0.5,  damping=0.3, wet_level=0.10),
        }

        reverb = REVERB_PRESETS.get(reverb_type, REVERB_PRESETS["small_room"])
        board  = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),   # Remove low rumble
            Compressor(threshold_db=-18, ratio=3.0),  # Even dynamics
            reverb,
            LowpassFilter(cutoff_frequency_hz=14000), # Natural roll-off
        ])

        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        processed = board(audio.T, sr).T
        sf.write(str(out_path), processed, sr)
        return True
    except Exception as e:
        log.warning(f"Room acoustics failed: {e} — using raw audio")
        import shutil
        shutil.copy(audio_path, out_path)
        return False
