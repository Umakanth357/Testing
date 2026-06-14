"""
Script Engine — Content ingestion, copyright protection, script generation.

Pipeline:
  YouTube URL / text → transcribe → extract facts → similarity check
  → rewrite fresh script → detect format → split for debate if needed
"""
import re
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import httpx
import requests

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, LANGUAGES, BRAND_SCENE_MAP

log = logging.getLogger("script_engine")

# Similarity threshold — above this triggers a forced rewrite
SIMILARITY_THRESHOLD = 0.35

# ── YouTube Ingestion ─────────────────────────────────────────────────────────

def download_youtube_audio(url: str, out_dir: Path) -> Optional[Path]:
    """Download audio from YouTube URL using yt-dlp. Returns WAV path or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "source_audio.wav"

    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", str(out_path.with_suffix("")),  # yt-dlp adds extension
        "--no-playlist",
        "--quiet",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.error(f"yt-dlp failed: {result.stderr[:500]}")
            return None
        # yt-dlp may name file slightly differently
        candidates = list(out_dir.glob("source_audio*"))
        return candidates[0] if candidates else None
    except subprocess.TimeoutExpired:
        log.error("yt-dlp timed out after 300s")
        return None
    except FileNotFoundError:
        log.error("yt-dlp not installed")
        return None


def transcribe_audio(audio_path: Path, language: str = "te") -> Optional[str]:
    """Transcribe audio using faster-whisper. Returns full transcript text."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device="auto", compute_type="float16")
        lang_code = language if language != "en" else None  # None = auto-detect for English
        segments, _ = model.transcribe(str(audio_path), language=lang_code, beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments)
        log.info(f"Transcribed {len(transcript)} characters")
        return transcript
    except ImportError:
        log.warning("faster-whisper not available, falling back to whisper")
        return _transcribe_whisper_fallback(audio_path, language)
    except Exception as e:
        log.error(f"Transcription failed: {e}")
        return None


def _transcribe_whisper_fallback(audio_path: Path, language: str) -> Optional[str]:
    try:
        import whisper
        model = whisper.load_model("large-v3")
        result = model.transcribe(str(audio_path), language=language)
        return result.get("text", "")
    except Exception as e:
        log.error(f"Whisper fallback failed: {e}")
        return None


# ── Copyright Protection — 3-Layer System ────────────────────────────────────

def extract_facts_only(transcript: str) -> list[str]:
    """
    Layer 1: Strip all original sentences. Extract only facts, numbers, names.
    Returns list of raw fact strings. All original language/structure is discarded.
    """
    prompt = f"""You are a fact extractor. Extract ONLY raw facts from this transcript.
Output a JSON array of fact strings. Each fact = one discrete piece of information.
Include: names, numbers, dates, statistics, product names, locations, events.
DISCARD: all sentences, opinions, transitions, filler words, original structure.
Do NOT quote or paraphrase original sentences.

Transcript:
{transcript[:4000]}

Output ONLY valid JSON array, nothing else. Example: ["fact1", "fact2"]"""

    response = _ollama(prompt)
    if not response:
        return []
    try:
        # Extract JSON array from response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    # Fallback: split by newlines
    return [line.strip("- ").strip() for line in response.splitlines() if line.strip()]


def check_similarity(original: str, rewritten: str) -> float:
    """
    Layer 2: Compute semantic similarity between original transcript and rewritten script.
    Returns 0.0 (completely different) to 1.0 (identical).
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        orig_chunks = _chunk_text(original, 512)
        new_chunks = _chunk_text(rewritten, 512)
        orig_emb = model.encode(orig_chunks[:10], convert_to_tensor=True)
        new_emb = model.encode(new_chunks[:10], convert_to_tensor=True)
        scores = util.cos_sim(orig_emb, new_emb)
        return float(scores.max().item())
    except Exception as e:
        log.warning(f"Similarity check failed: {e} — defaulting to 0.0 (safe)")
        return 0.0


def generate_script(facts: list[str], language: str, format_type: str,
                    topic: str = "", tone: str = "professional",
                    duration_sec: int = 180) -> str:
    """
    Layer 3: Generate completely fresh script from facts only.
    New structure, new angle, new transitions — original source unrecognisable.
    """
    lang_name = LANGUAGES.get(language, "English")
    word_count = int(duration_sec * 2.5)   # ~2.5 words/sec average speech rate

    format_instructions = {
        "monologue":   "Single presenter. Natural conversational flow. Use first person.",
        "debate":      "TWO speakers: SPEAKER_A (pro/for) and SPEAKER_B (con/against). Label each line. Make it engaging.",
        "news_anchor": "Professional news anchor style. Formal, authoritative. Short punchy sentences.",
        "seminar":     "Academic/professional presentation. Well-structured with transitions. Include agenda.",
        "short":       "Hook in first sentence. Very concise. Max 150 words total. High energy.",
    }

    agenda_instruction = ""
    if format_type in ("monologue", "seminar", "news_anchor") and len(facts) > 3:
        agenda_instruction = "Start with a compelling hook line, then say 'Today we cover:' and list the top 3-4 topics naturally."

    prompt = f"""Write a complete video script in {lang_name} language.

Format: {format_instructions.get(format_type, format_instructions['monologue'])}
Tone: {tone}
Target length: approximately {word_count} words
Topic: {topic or 'General information'}
{agenda_instruction}

Facts to cover (use ALL of them, but in your OWN words and structure):
{chr(10).join(f'- {f}' for f in facts[:30])}

Requirements:
- Write ENTIRELY in {lang_name}
- Use natural speech rhythm with commas and pauses
- Include emotional transitions (excitement, gravity, curiosity)
- Add breathing indicators [breath] before long sentences
- Do NOT copy any original source sentences
- Start with a strong hook that grabs attention in 5 seconds
- End with a clear call to action or memorable closing line
{"- Mark each speaker change as SPEAKER_A: or SPEAKER_B:" if format_type == "debate" else ""}

Write the complete script now:"""

    script = _ollama(prompt, temperature=0.7)
    if not script:
        log.error("Script generation failed")
        return ""
    return script.strip()


def rewrite_if_similar(original_transcript: str, script: str, facts: list[str],
                        language: str, format_type: str, topic: str) -> str:
    """Force rewrite if script is too similar to original. Max 3 attempts."""
    for attempt in range(3):
        similarity = check_similarity(original_transcript, script)
        log.info(f"Script similarity attempt {attempt+1}: {similarity:.2f}")
        if similarity < SIMILARITY_THRESHOLD:
            log.info("Similarity check passed")
            return script
        log.warning(f"Similarity {similarity:.2f} > {SIMILARITY_THRESHOLD} — rewriting")
        script = generate_script(facts, language, format_type, topic,
                                  tone=["energetic", "analytical", "storytelling"][attempt % 3])
    log.info("Rewrite loop complete — returning best attempt")
    return script


# ── Format Detection ──────────────────────────────────────────────────────────

def detect_format(text: str) -> dict:
    """
    Analyse content and suggest best video format + scene.
    Returns: {format, scene, reason, entities}
    """
    prompt = f"""Analyse this content and return a JSON object with:
- "format": best video format (one of: monologue, debate, news_anchor, seminar, short)
- "scene": best background scene key (examples: professional/office, nature/beach, brand/google_stage, professional/news_desk)
- "reason": one sentence explanation
- "entities": list of brand/place names mentioned (e.g. ["Google", "Apple", "Red Fort"])
- "tone": overall tone (professional, casual, excited, serious)
- "debate_topic": if debate, what is the central debate question

Content:
{text[:1000]}

Return ONLY valid JSON, nothing else."""

    response = _ollama(prompt)
    if not response:
        return {"format": "monologue", "scene": "professional/office", "reason": "default", "entities": [], "tone": "professional", "debate_topic": ""}

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"format": "monologue", "scene": "professional/office", "reason": "fallback", "entities": [], "tone": "professional", "debate_topic": ""}


# ── Debate Script Splitter ────────────────────────────────────────────────────

def split_debate_script(script: str) -> tuple[list[str], list[str]]:
    """
    Split a debate script into two speaker tracks.
    Returns (speaker_a_lines, speaker_b_lines)
    """
    lines_a, lines_b = [], []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SPEAKER_A:"):
            lines_a.append(line[len("SPEAKER_A:"):].strip())
        elif line.upper().startswith("SPEAKER_B:"):
            lines_b.append(line[len("SPEAKER_B:"):].strip())
        else:
            # Untagged line — assign alternately
            if len(lines_a) <= len(lines_b):
                lines_a.append(line)
            else:
                lines_b.append(line)
    return lines_a, lines_b


# ── Scene Sequence Builder ────────────────────────────────────────────────────

def build_scene_sequence(script: str, entities: list[str], default_scene: str) -> list[dict]:
    """
    Split script into segments, assign background per segment based on entity mentions.
    Returns list of {text, scene} dicts.
    """
    # Split on paragraph breaks or topic transitions
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', script) if p.strip()]
    if len(paragraphs) < 2:
        # Single paragraph — no scene switching needed
        return [{"text": script, "scene": default_scene}]

    segments = []
    for para in paragraphs:
        scene = default_scene
        para_lower = para.lower()
        for keyword, scene_key in BRAND_SCENE_MAP.items():
            if keyword in para_lower:
                scene = scene_key
                break
        segments.append({"text": para, "scene": scene})

    log.info(f"Built {len(segments)} scene segments")
    return segments


# ── Agenda Extractor ──────────────────────────────────────────────────────────

def extract_agenda(script: str) -> list[str]:
    """Extract top-level agenda items from script for on-screen display."""
    prompt = f"""Extract the main agenda items / topics from this script.
Return a JSON array of 3-5 short topic titles (max 5 words each).
These will appear as on-screen agenda items.

Script:
{script[:2000]}

Return ONLY a JSON array. Example: ["AI Tools Update", "Market News", "Product Review"]"""

    response = _ollama(prompt)
    if not response:
        return []
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def process_content(
    source: str,              # YouTube URL or raw text
    language: str = "te",
    format_type: str = "auto",
    duration_sec: int = 180,
    topic: str = "",
    work_dir: Optional[Path] = None,
) -> dict:
    """
    Full pipeline: source → approved script ready for TTS.
    Returns dict with: script, format, scene_sequence, agenda, debate_parts, metadata
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="avatarjob_"))

    result = {
        "script": "", "format": format_type, "scene_sequence": [],
        "agenda": [], "debate_parts": None, "metadata": {},
        "error": None,
    }

    # ── Step 1: Get raw text ──────────────────────────────────────────────────
    original_transcript = ""
    if source.startswith("http"):
        log.info(f"Downloading YouTube: {source}")
        audio_path = download_youtube_audio(source, work_dir / "audio")
        if not audio_path:
            result["error"] = "Failed to download YouTube audio. Check URL and yt-dlp installation."
            return result
        original_transcript = transcribe_audio(audio_path, language)
        if not original_transcript:
            result["error"] = "Transcription failed. Check Whisper installation and audio quality."
            return result
        log.info(f"Transcribed: {len(original_transcript)} chars")
    else:
        original_transcript = source

    # ── Step 2: Detect format if auto ────────────────────────────────────────
    detected = detect_format(original_transcript)
    if format_type == "auto":
        format_type = detected.get("format", "monologue")
        result["format"] = format_type
    default_scene = detected.get("scene", "professional/office")
    entities = detected.get("entities", [])
    result["metadata"] = {
        "detected_format": detected.get("format"),
        "detected_scene": default_scene,
        "tone": detected.get("tone", "professional"),
        "entities": entities,
        "debate_topic": detected.get("debate_topic", ""),
        "original_length": len(original_transcript),
    }

    # ── Step 3: Extract facts (copyright layer 1) ─────────────────────────────
    log.info("Extracting facts from transcript...")
    facts = extract_facts_only(original_transcript)
    if not facts:
        log.warning("No facts extracted — using full transcript as base")
        facts = [s.strip() for s in original_transcript.split('. ') if s.strip()][:30]

    # ── Step 4: Generate fresh script ────────────────────────────────────────
    log.info(f"Generating {format_type} script in {language}...")
    script = generate_script(facts, language, format_type, topic, detected.get("tone", "professional"), duration_sec)
    if not script:
        result["error"] = "Script generation failed. Check Ollama is running."
        return result

    # ── Step 5: Similarity check + rewrite if needed (copyright layer 2) ─────
    if source.startswith("http"):
        script = rewrite_if_similar(original_transcript, script, facts, language, format_type, topic)

    result["script"] = script

    # ── Step 6: Build scene sequence ─────────────────────────────────────────
    result["scene_sequence"] = build_scene_sequence(script, entities, default_scene)

    # ── Step 7: Extract agenda ────────────────────────────────────────────────
    result["agenda"] = extract_agenda(script)

    # ── Step 8: Split debate if needed ────────────────────────────────────────
    if format_type == "debate":
        lines_a, lines_b = split_debate_script(script)
        result["debate_parts"] = {
            "speaker_a": " ".join(lines_a),
            "speaker_b": " ".join(lines_b),
        }

    log.info("Script pipeline complete")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ollama(prompt: str, temperature: float = 0.3) -> Optional[str]:
    """Call local Ollama with retry."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 2048},
    }
    for attempt in range(3):
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            log.warning(f"Ollama attempt {attempt+1} failed: {e}")
    return None


def _chunk_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(" ".join(current)) >= max_chars:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks
