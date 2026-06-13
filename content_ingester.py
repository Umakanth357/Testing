"""
content_ingester.py
Handles all content sources:
  - YouTube URL → audio → Whisper transcription → Ollama summarise → script
  - Plain text / document paste → direct to script
  - Web article URL → newspaper3k → Ollama summarise → script
"""
import os
import re
import json
import logging
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def _call_ollama(prompt: str, model: str = "gemma3:4b") -> str:
    """Call local Ollama. Falls back to Groq free API if Ollama unavailable."""
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 2000}
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama failed: {e}. Trying Groq fallback...")
        return _call_groq(prompt)


def _call_groq(prompt: str) -> str:
    """Groq free API — 14,400 req/day free, no credit card."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return ""
    try:
        payload = json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.3
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Groq API also failed: {e}")
        return ""


def ingest_youtube(
    url: str,
    language: str = "en",
    tone: str = "professional",
    target_duration_mins: int = 3,
    content_type: str = "KT"
) -> Dict:
    """
    Full pipeline: YouTube URL → transcript → summarise → narration script.
    Returns dict with keys: transcript, summary, script, title, source_url
    """
    logger.info(f"Ingesting YouTube: {url}")
    tmpdir = tempfile.mkdtemp(prefix="yt_ingest_")

    # ── Step 1: Try native captions first (fastest) ──────────────
    transcript = _get_youtube_captions(url)

    # ── Step 2: Download audio + Whisper if no captions ──────────
    if not transcript:
        logger.info("No captions found — downloading audio for Whisper transcription")
        audio_path = _download_youtube_audio(url, tmpdir)
        if audio_path:
            transcript = _whisper_transcribe(audio_path, language)

    if not transcript:
        return {"error": "Could not extract content from YouTube URL", "script": ""}

    # ── Step 3: Get video title ───────────────────────────────────
    title = _get_youtube_title(url)

    # ── Step 4: LLM — summarise + rewrite as narration script ────
    script = _generate_script(
        transcript=transcript,
        title=title,
        source_url=url,
        language=language,
        tone=tone,
        target_duration_mins=target_duration_mins,
        content_type=content_type
    )

    return {
        "transcript": transcript[:2000] + "..." if len(transcript) > 2000 else transcript,
        "summary":    _quick_summary(transcript),
        "script":     script,
        "title":      title,
        "source_url": url,
    }


def ingest_text(
    text: str,
    language: str = "en",
    tone: str = "professional",
    target_duration_mins: int = 3,
    content_type: str = "KT"
) -> Dict:
    """Direct text/document → script."""
    script = _generate_script(
        transcript=text,
        title="Custom Script",
        source_url="",
        language=language,
        tone=tone,
        target_duration_mins=target_duration_mins,
        content_type=content_type
    )
    return {"transcript": text, "script": script, "title": "Custom Script", "source_url": ""}


def _get_youtube_captions(url: str) -> str:
    """Try to get native captions via youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        video_id = _extract_video_id(url)
        if not video_id:
            return ""
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "te", "hi", "ta", "kn", "ml"]
        )
        return " ".join([t["text"] for t in transcript_list])
    except Exception as e:
        logger.debug(f"Caption extraction failed: {e}")
        return ""


def _download_youtube_audio(url: str, tmpdir: str) -> Optional[str]:
    """Download audio using yt-dlp."""
    output_template = os.path.join(tmpdir, "audio.%(ext)s")
    try:
        result = subprocess.run([
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--output", output_template,
            "--no-playlist",
            "--quiet",
            url
        ], capture_output=True, text=True, timeout=300)

        wav_files = list(Path(tmpdir).glob("*.wav"))
        if wav_files:
            return str(wav_files[0])

        logger.error(f"yt-dlp failed: {result.stderr[:300]}")
        return None
    except Exception as e:
        logger.error(f"YouTube download error: {e}")
        return None


def _whisper_transcribe(audio_path: str, language: str = "en") -> str:
    """Transcribe audio using OpenAI Whisper (local)."""
    try:
        import whisper
        logger.info(f"Transcribing with Whisper ({audio_path})...")
        model = whisper.load_model("large-v3")
        result = model.transcribe(
            audio_path,
            language=language if language != "en" else None,
            task="transcribe",
            verbose=False
        )
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return ""


def _get_youtube_title(url: str) -> str:
    """Get YouTube video title via yt-dlp."""
    try:
        result = subprocess.run([
            "yt-dlp", "--get-title", "--no-playlist", "--quiet", url
        ], capture_output=True, text=True, timeout=30)
        return result.stdout.strip() or "YouTube Video"
    except Exception:
        return "YouTube Video"


def _extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _quick_summary(transcript: str) -> str:
    """Quick 3-sentence summary for display."""
    prompt = f"""Summarise the following content in exactly 3 sentences. Be concise and factual.

Content:
{transcript[:3000]}

3-sentence summary:"""
    return _call_ollama(prompt) or transcript[:300] + "..."


def _generate_script(
    transcript: str,
    title: str,
    source_url: str,
    language: str,
    tone: str,
    target_duration_mins: int,
    content_type: str
) -> str:
    """
    Core LLM prompt that converts any source material into a narration script.
    Key copyright protection: explicitly instructed NOT to copy phrases.
    Key quality: per-sentence emotion guidance.
    """

    tone_descriptions = {
        "professional": "calm, clear, authoritative corporate presenter",
        "friendly":     "warm, approachable, conversational colleague",
        "excited":      "enthusiastic, energetic, genuinely excited presenter",
        "serious":      "serious, measured, weight-bearing delivery",
        "reviewer":     "honest reviewer sharing genuine personal opinion",
    }

    content_type_instructions = {
        "KT":           "Focus on teaching. Explain what, why, and how. Use examples. Structure: intro → key concepts → demo walk-through → summary → next steps.",
        "tool_review":  "Give an honest balanced assessment. Cover: what it is, key features, what's great, what's missing, who should use it.",
        "news":         "Report the facts clearly. Cover: what happened, why it matters, what comes next.",
        "movie_review": "Give a genuine opinion. Cover: plot summary (no spoilers), performance, direction, verdict. Add personal recommendation.",
        "event_summary":"Summarise the key announcements. Cover: biggest reveals, technical details, what it means for the industry.",
        "notification": "Be brief and direct. One clear message. Action if needed. Warm close.",
    }

    lang_instruction = {
        "te": "Write the ENTIRE script in natural Telugu (తెలుగు). Use conversational Telugu, not formal/academic. Numbers and technical terms can be in English.",
        "kn": "Write the ENTIRE script in natural Kannada (ಕನ್ನಡ).",
        "ta": "Write the ENTIRE script in natural Tamil (தமிழ்).",
        "hi": "Write the ENTIRE script in natural Hindi (हिंदी).",
        "en": "Write in clear, natural spoken English.",
    }.get(language, "Write in clear English.")

    word_count = target_duration_mins * 130  # ~130 words/min for natural speech

    source_credit = f"\n\nSource material is from: {source_url}" if source_url else ""

    prompt = f"""You are a professional script writer for an AI avatar video system.

Task: Rewrite the following source material as a spoken narration script.

Content type: {content_type}
Target duration: {target_duration_mins} minutes (~{word_count} words)
Tone: {tone_descriptions.get(tone, tone_descriptions['professional'])}
Language: {lang_instruction}

{content_type_instructions.get(content_type, content_type_instructions['KT'])}

CRITICAL RULES:
1. DO NOT copy any phrases verbatim from the source. Express all ideas in completely fresh language.
2. Write as natural SPOKEN words — not written prose. Short sentences. Conversational rhythm.
3. Add natural filler transitions: "So", "Now", "Here's the thing", "What's interesting is", "Right", "Let me show you"
4. Vary sentence length. Mix short punchy sentences with slightly longer explanatory ones.
5. The script will be read by an AI avatar — make it sound like a real person talking, not reading.
6. DO NOT include stage directions, [pause], (emotion tags), or any markup — plain text only.
7. End with a warm close that feels natural, not scripted.
8. Keep to approximately {word_count} words.{source_credit}

Source material:
---
{transcript[:4000]}
---

Narration script (plain text, spoken words only):"""

    script = _call_ollama(prompt)

    if not script or len(script) < 100:
        # Fallback: return a condensed version of transcript
        logger.warning("LLM script generation returned empty — using transcript excerpt")
        return transcript[:word_count * 5]  # rough chars

    return script


if __name__ == "__main__":
    # Self-test with a plain text input (no YouTube needed)
    test_text = """
    MuseTalk is an open-source AI lip-sync model by Tencent Music Entertainment.
    It generates realistic talking-face animations at 30 frames per second.
    The model uses latent space inpainting which produces sharper teeth and mouth
    regions compared to older GAN-based approaches like Wav2Lip.
    It supports any audio language including Telugu, English, and Hindi.
    """
    print("Testing content ingester with plain text input...\n")
    result = ingest_text(test_text, language="en", tone="professional", content_type="tool_review")
    print("Generated script:\n")
    print(result["script"][:500] + "..." if len(result.get("script","")) > 500 else result.get("script","FAILED"))
    print("\n✓ Content ingester self-test complete")
