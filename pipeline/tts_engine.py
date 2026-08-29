"""
tts_engine.py — Emotion-aware TTS pipeline

Primary:  edge-tts  (Microsoft Azure Neural — te-IN-ShrutiNeural / MohanNeural)
          Natural Telugu prosody, ~3s for 3min script, zero cost, works immediately.

Enhanced: Indic Parler-TTS (ai4bharat/indic-parler-tts)
          Best Telugu quality in 2025 — 1806h training, 6 emotion parameters.
          Requires HF gated access: huggingface.co/ai4bharat/indic-parler-tts
          When available, emotion description ("Priya speaks in an excited tone...")
          is passed to model for natural emotional delivery.

Fallback: gTTS (Google) — always works, somewhat robotic.

Emotion detection:
  Script is analysed sentence by sentence. Keywords and punctuation patterns
  identify emotional context. Each segment gets an emotion tag used to:
    a) Set edge-tts rate/pitch parameters
    b) Build Parler-TTS description prompt when Parler is available

Audio post-processing (pipeline/audio_post.py) runs after TTS to add
broadcast-quality processing and breathing sounds.
"""
import asyncio
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("tts_engine")

ROOT = Path(__file__).parent.parent.resolve()

# ── Voice map ─────────────────────────────────────────────────────────────────
# edge-tts voice IDs by language + gender
EDGE_VOICES = {
    "te": {"female": "te-IN-ShrutiNeural",  "male": "te-IN-MohanNeural"},
    "hi": {"female": "hi-IN-SwaraNeural",   "male": "hi-IN-MadhurNeural"},
    "ta": {"female": "ta-IN-PallaviNeural", "male": "ta-IN-ValluvarNeural"},
    "kn": {"female": "kn-IN-SapnaNeural",   "male": "kn-IN-GaganNeural"},
    "ml": {"female": "ml-IN-SobhanaNeural", "male": "ml-IN-MidhunNeural"},
    "mr": {"female": "mr-IN-AarohiNeural",  "male": "mr-IN-ManoharNeural"},
    "bn": {"female": "bn-IN-TanishaaNeural","male": "bn-IN-BashkarNeural"},
    "en": {"female": "en-IN-NeerjaNeural",  "male": "en-IN-PrabhatNeural"},
}

# ── Voice profiles ────────────────────────────────────────────────────────────
# Maps persona voice_profile → (language, gender, base_rate, base_pitch)
VOICE_PROFILES = {
    "navya_telugu": {
        "lang": "te", "gender": "female",
        "rate": "+0%", "pitch": "+2Hz",
        "parler_desc": "Navya speaks in a warm, professional Telugu news anchor voice.",
    },
    "arjun_telugu": {
        "lang": "te", "gender": "male",
        "rate": "-5%", "pitch": "-2Hz",
        "parler_desc": "Arjun speaks in a confident, authoritative Telugu anchor voice.",
    },
    "priya_telugu": {
        "lang": "te", "gender": "female",
        "rate": "+5%", "pitch": "+4Hz",
        "parler_desc": "Priya speaks in an energetic, youthful Telugu presenter voice.",
    },
    "default_female_te": {
        "lang": "te", "gender": "female",
        "rate": "+0%", "pitch": "+0Hz",
        "parler_desc": "A professional Telugu female news reader.",
    },
    "default_male_te": {
        "lang": "te", "gender": "male",
        "rate": "+0%", "pitch": "+0Hz",
        "parler_desc": "A professional Telugu male news reader.",
    },
}

# ── Emotion → edge-tts rate/pitch modifiers ───────────────────────────────────
EMOTION_MODIFIERS = {
    "excited":      {"rate": "+20%",  "pitch": "+6Hz"},
    "energetic":    {"rate": "+12%",  "pitch": "+4Hz"},
    "professional": {"rate": "+0%",   "pitch": "+0Hz"},
    "warm":         {"rate": "-5%",   "pitch": "+2Hz"},
    "calm":         {"rate": "-10%",  "pitch": "-2Hz"},
    "serious":      {"rate": "-8%",   "pitch": "-4Hz"},
    "sombre":       {"rate": "-15%",  "pitch": "-6Hz"},
    "questioning":  {"rate": "+3%",   "pitch": "+8Hz"},
}

# ── Emotion → Parler-TTS description fragments ───────────────────────────────
EMOTION_PARLER = {
    "excited":      "in an excited, enthusiastic tone with rising intonation",
    "energetic":    "in an energetic, fast-paced tone full of energy",
    "professional": "in a clear, professional, measured tone",
    "warm":         "in a warm, friendly, conversational tone",
    "calm":         "in a calm, reassuring, slow and measured tone",
    "serious":      "in a serious, grave, authoritative tone",
    "sombre":       "in a solemn, low-energy, respectful tone",
    "questioning":  "in an inquisitive tone with rising pitch at the end",
}


# ── Emotion detection ─────────────────────────────────────────────────────────

def detect_emotion(text: str) -> str:
    """
    Detect the dominant emotion in a text segment.
    Returns emotion string: 'excited' | 'professional' | 'serious' | etc.
    Simple keyword + punctuation heuristic — fast, no model required.
    """
    t = text.lower().strip()

    # Sombre / sad signals
    if re.search(r"(మరణ|మృతి|ప్రాణ నష్టం|విషాదం|tragedy|tragic|sad|died|death|fatal|గుండె|దుఃఖ)", t):
        return "sombre"

    # Serious / breaking news
    if re.search(r"(breaking|అత్యవసర|హెచ్చరిక|serious|urgent|critical|danger|ప్రమాద)", t):
        return "serious"

    # Excited / celebration
    if re.search(r"(అద్భుతం|అభినందన|విజయం|fantastic|amazing|celebrate|won|victory|achievement|record|గొప్ప)", t):
        return "excited"

    # Warm / human interest
    if re.search(r"(ఆనందం|happy|joy|love|family|children|hope|inspiring|heart|feel)", t):
        return "warm"

    # Questioning / analysis
    if text.strip().endswith("?") or re.search(r"(why|how|what does|అర్థం|ఎందుకు|ఎలా)", t):
        return "questioning"

    # Default: professional delivery
    return "professional"


def detect_anchor_style(voice_profile: str) -> str:
    """Map voice profile to broadcast anchor style."""
    if "arjun" in voice_profile.lower():
        return "tv_news"
    if "priya" in voice_profile.lower():
        return "radio"
    return "tv_news"


# ── Text pre-processing ───────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean script text before TTS:
      - Remove stage directions: [EMOTION:excited], (pause), **bold**
      - Remove speaker labels: NAVYA:, HOST:
      - Normalise whitespace
      - Remove markdown formatting
    """
    # Remove [EMOTION:xxx] tags
    text = re.sub(r"\[EMOTION:[^\]]+\]", "", text, flags=re.I)
    # Remove (stage directions in parentheses)
    text = re.sub(r"\([^)]{1,40}\)", "", text)
    # Remove speaker labels at line start: "NAVYA:", "HOST:", "ANCHOR:"
    text = re.sub(r"^[A-Z\s]{2,12}:\s*", "", text, flags=re.MULTILINE)
    # Remove markdown bold/italic
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    # Remove markdown headers
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── edge-tts (primary) ────────────────────────────────────────────────────────

async def _edge_synthesise(
    text: str,
    voice: str,
    out_path: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bool:
    """Async edge-tts synthesis."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(str(out_path))
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        log.error(f"edge-tts error: {e}")
        return False


def synthesise_edge_tts(
    text: str,
    voice_profile: str,
    out_path: Path,
    emotion: str = "professional",
) -> bool:
    """Synthesise with edge-tts using emotion-aware rate/pitch modifiers."""
    profile   = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["default_female_te"])
    lang      = profile["lang"]
    gender    = profile["gender"]
    voice     = EDGE_VOICES.get(lang, EDGE_VOICES["te"])[gender]

    # Merge base rate/pitch with emotion modifiers
    base_rate  = profile["rate"]
    base_pitch = profile["pitch"]
    emo        = EMOTION_MODIFIERS.get(emotion, EMOTION_MODIFIERS["professional"])

    # Combine: parse percentage numbers and add them
    def _combine_rate(a: str, b: str) -> str:
        va = int(re.sub(r"[^-\d]", "", a) or "0")
        vb = int(re.sub(r"[^-\d]", "", b) or "0")
        total = max(-50, min(50, va + vb))
        return f"{total:+d}%"

    def _combine_pitch(a: str, b: str) -> str:
        va = int(re.sub(r"[^-\d]", "", a) or "0")
        vb = int(re.sub(r"[^-\d]", "", b) or "0")
        total = max(-20, min(20, va + vb))
        return f"{total:+d}Hz"

    final_rate  = _combine_rate(base_rate, emo["rate"])
    final_pitch = _combine_pitch(base_pitch, emo["pitch"])

    text_clean = preprocess_text(text)
    if not text_clean:
        return False

    log.info(f"edge-tts | voice={voice} rate={final_rate} pitch={final_pitch} emotion={emotion}")
    ok = asyncio.run(
        _edge_synthesise(text_clean, voice, out_path, final_rate, final_pitch)
    )

    # Convert to WAV for pipeline compatibility
    if ok and out_path.suffix.lower() != ".wav":
        wav = out_path.with_suffix(".wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-ar", "44100", "-ac", "1", str(wav)],
            capture_output=True, timeout=30
        )
        if r.returncode == 0:
            out_path.unlink(missing_ok=True)
            wav.rename(out_path)

    return ok


# ── Indic Parler-TTS (enhanced) ───────────────────────────────────────────────

_PARLER_MODEL = None
_PARLER_TOKENISER = None


def _load_parler():
    """Lazy-load Indic Parler-TTS model (ai4bharat/indic-parler-tts)."""
    global _PARLER_MODEL, _PARLER_TOKENISER
    if _PARLER_MODEL is not None:
        return True
    try:
        import torch
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration

        model_id = "ai4bharat/indic-parler-tts"
        log.info("Loading Indic Parler-TTS model (~4GB)...")
        _PARLER_TOKENISER = AutoTokenizer.from_pretrained(model_id)
        _PARLER_MODEL = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Indic Parler-TTS loaded")
        return True
    except Exception as e:
        log.warning(f"Indic Parler-TTS not available: {e}")
        return False


def synthesise_parler(
    text: str,
    voice_profile: str,
    out_path: Path,
    emotion: str = "professional",
) -> bool:
    """Synthesise with Indic Parler-TTS using emotion description."""
    if not _load_parler():
        return False

    profile = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["default_female_te"])
    emotion_frag = EMOTION_PARLER.get(emotion, EMOTION_PARLER["professional"])
    description  = f"{profile['parler_desc']} She speaks {emotion_frag}. " \
                   "The recording is clear and high quality with minimal background noise."

    text_clean = preprocess_text(text)
    if not text_clean:
        return False

    try:
        import torch, soundfile as sf

        inputs = _PARLER_TOKENISER(description, return_tensors="pt").to(_PARLER_MODEL.device)
        prompt = _PARLER_TOKENISER(text_clean, return_tensors="pt").to(_PARLER_MODEL.device)

        with torch.no_grad():
            generation = _PARLER_MODEL.generate(
                input_ids=inputs.input_ids,
                prompt_input_ids=prompt.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_attention_mask=prompt.attention_mask,
            )

        audio = generation.cpu().numpy().squeeze()
        sr    = _PARLER_MODEL.config.sampling_rate

        sf.write(str(out_path), audio, sr)
        log.info(f"Parler-TTS OK | emotion={emotion} → {out_path.name}")
        return out_path.exists() and out_path.stat().st_size > 0

    except Exception as e:
        log.error(f"Parler-TTS inference failed: {e}")
        return False


# ── gTTS fallback ─────────────────────────────────────────────────────────────

def synthesise_gtts(text: str, voice_profile: str, out_path: Path) -> bool:
    """gTTS fallback — robotic but always works."""
    try:
        from gtts import gTTS
        profile = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["default_female_te"])
        lang    = profile["lang"]
        tts     = gTTS(text=preprocess_text(text), lang=lang, slow=False)
        tmp_mp3 = out_path.with_suffix(".mp3")
        tts.save(str(tmp_mp3))
        # Convert to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_mp3), "-ar", "44100", "-ac", "1", str(out_path)],
            capture_output=True, timeout=30
        )
        tmp_mp3.unlink(missing_ok=True)
        log.info(f"gTTS fallback OK → {out_path.name}")
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        log.error(f"gTTS failed: {e}")
        return False


# ── Master synthesise function ────────────────────────────────────────────────

def synthesize(
    text: str,
    voice_profile: str,
    out_path: str,
    emotion: Optional[str] = None,
    use_parler: bool = False,
    apply_broadcast: bool = True,
) -> bool:
    """
    Main TTS entry point.

    Chain:
      1. Detect emotion from text if not provided
      2. Try Indic Parler-TTS (if use_parler=True and model available)
      3. Fall back to edge-tts (primary — always attempted first if not parler)
      4. Fall back to gTTS
      5. Apply broadcast audio post-processing

    Returns True if audio was generated successfully.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not text.strip():
        log.error("Empty text — nothing to synthesise")
        return False

    # Auto-detect emotion
    if emotion is None:
        emotion = detect_emotion(text)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_out = Path(tmpdir) / "raw_tts.wav"

        # TTS synthesis
        ok = False
        if use_parler:
            ok = synthesise_parler(text, voice_profile, raw_out, emotion)
        if not ok:
            ok = synthesise_edge_tts(text, voice_profile, raw_out, emotion)
        if not ok:
            log.warning("edge-tts failed — falling back to gTTS")
            ok = synthesise_gtts(text, voice_profile, raw_out)

        if not ok:
            log.error("All TTS engines failed")
            return False

        # Apply broadcast processing
        if apply_broadcast:
            try:
                from pipeline.audio_post import master_audio
                anchor_style = detect_anchor_style(voice_profile)
                master_audio(
                    raw_tts_path=raw_out,
                    out_path=out_path,
                    emotion=emotion,
                    anchor_style=anchor_style,
                    add_breaths=True,
                )
            except Exception as e:
                log.warning(f"Broadcast processing error: {e} — using raw TTS")
                import shutil; shutil.copy2(str(raw_out), str(out_path))
        else:
            import shutil; shutil.copy2(str(raw_out), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 0
    if ok:
        log.info(f"TTS complete | profile={voice_profile} emotion={emotion} → {out_path.name}")
    return ok


# ── Segmented synthesis (for long scripts) ───────────────────────────────────

def synthesize_segmented(
    text: str,
    voice_profile: str,
    out_path: str,
    use_parler: bool = False,
) -> bool:
    """
    Split long scripts into paragraphs, synthesise each with detected emotion,
    then concatenate. Gives per-paragraph emotion variation for long videos.
    """
    import tempfile as _tf

    out_path = Path(out_path)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    if len(paragraphs) <= 1:
        return synthesize(text, voice_profile, out_path, use_parler=use_parler)

    log.info(f"Segmented TTS: {len(paragraphs)} paragraphs")

    with _tf.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        seg_files = []

        for i, para in enumerate(paragraphs):
            emotion = detect_emotion(para)
            seg_out = tmp / f"seg_{i:03d}.wav"
            ok = synthesize(para, voice_profile, seg_out,
                            emotion=emotion, use_parler=use_parler,
                            apply_broadcast=False)  # broadcast applied once at end
            if ok:
                seg_files.append(seg_out)
            else:
                log.warning(f"Segment {i} TTS failed — skipping")

        if not seg_files:
            return False

        # Concatenate segments
        concat_list = tmp / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{f}'" for f in seg_files)
        )
        concat_raw = tmp / "concat_raw.wav"
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(concat_raw)],
            capture_output=True, timeout=120
        )
        if r.returncode != 0:
            log.error("Concat failed")
            return False

        # Apply broadcast processing once on the full audio
        try:
            from pipeline.audio_post import master_audio
            master_audio(concat_raw, out_path,
                         emotion="professional",
                         anchor_style=detect_anchor_style(voice_profile))
        except Exception:
            import shutil; shutil.copy2(str(concat_raw), str(out_path))

    ok = out_path.exists() and out_path.stat().st_size > 0
    if ok:
        log.info(f"Segmented TTS complete → {out_path.name}")
    return ok
