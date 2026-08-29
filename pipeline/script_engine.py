"""
script_engine.py — LLM-powered Telugu script generation with emotion tagging

Uses llama3.1:8b via Ollama to generate anchor-style Telugu scripts from:
  - Transcribed content (Instagram/YouTube/audio)
  - Topic text
  - Debate format (two speakers)

Emotion tagging:
  Each paragraph in the generated script gets an [EMOTION:xxx] tag prepended.
  This is used by the TTS engine to match vocal delivery to content.
  The LLM is instructed to insert these tags naturally.

Tone profiles per character:
  Navya Reddy — warm, personable, breaking news urgency when needed
  Arjun Varma — authoritative, analytical, measured delivery
  Priya Sharma — energetic, youthful, entertainment-forward
"""
import logging
import re
import time
from typing import Optional

import requests

log = logging.getLogger("script_engine")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST    = "http://127.0.0.1:11434"
PRIMARY_MODEL  = "llama3.1:8b"
FALLBACK_MODEL = "gemma3:4b"
TIMEOUT        = 180   # seconds

# ── Character system prompts ──────────────────────────────────────────────────
CHARACTER_PROMPTS = {
    "navya_telugu_f": """You are Navya Reddy, a popular Telugu news anchor known for your warm,
approachable style and ability to explain complex topics clearly.
Your delivery is professional yet personal. You connect emotionally with the audience.
For breaking news: urgent and clear. For feature stories: warm and engaging.
Always write in natural spoken Telugu, not written formal Telugu.""",

    "arjun_telugu_m": """You are Arjun Varma, a senior Telugu journalist and political analyst.
Known for authoritative, fact-driven reporting with deep analysis.
Your delivery is measured, confident, and commands respect.
You use precise Telugu vocabulary and structure arguments clearly.
Write in formal but accessible Telugu suitable for educated audiences.""",

    "priya_telugu_f": """You are Priya Sharma, a Telugu entertainment and lifestyle presenter.
Young, energetic, and enthusiastic. Your style is conversational and fun.
You make content relatable to young Telugu-speaking audiences.
Mix in trending references, keep energy high, and be expressive.
Write in modern spoken Telugu mixed with some English when natural.""",
}

# ── Duration → word count mapping ────────────────────────────────────────────
DURATION_WORDS = {
    "30s":  75,
    "60s":  150,
    "2min": 300,
    "3min": 450,
    "5min": 750,
    "8min": 1200,
    "10min": 1500,
}


# ── Ollama helper ─────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, model: str = PRIMARY_MODEL) -> str:
    """Send prompt to Ollama and return generated text."""
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama timed out after {TIMEOUT}s. Model may still be loading."
        )
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _generate_with_fallback(prompt: str) -> str:
    """Try primary model, fall back to gemma3:4b."""
    try:
        return _ollama_generate(prompt, PRIMARY_MODEL)
    except RuntimeError as e:
        if "not found" in str(e).lower() or "model" in str(e).lower():
            log.warning(f"Primary model failed: {e} — trying {FALLBACK_MODEL}")
            return _ollama_generate(prompt, FALLBACK_MODEL)
        raise


# ── Emotion tag insertion ─────────────────────────────────────────────────────

def tag_emotions_llm(script: str) -> str:
    """
    Ask the LLM to add [EMOTION:xxx] tags to each paragraph.
    Tags: excited | professional | serious | warm | sombre | calm | energetic
    """
    prompt = f"""You are a script editor. Read the following Telugu script and add emotion delivery tags.
Before each paragraph, add exactly one tag from this list:
[EMOTION:excited] — good news, achievements, celebrations
[EMOTION:professional] — news delivery, analysis, neutral information
[EMOTION:serious] — important news, warnings, alerts
[EMOTION:warm] — human interest, positive stories, emotional moments
[EMOTION:sombre] — tragedy, bad news, respectful reporting
[EMOTION:energetic] — entertainment news, sports, exciting events

Rules:
- One tag per paragraph only
- Tags must be exactly as shown above
- Do not add any other text or explanations
- Keep all Telugu text exactly as-is

SCRIPT:
{script}

OUTPUT (script with tags added):"""

    try:
        tagged = _generate_with_fallback(prompt)
        # Validate tags are present
        if "[EMOTION:" in tagged:
            return tagged
        # LLM didn't add tags — use simple heuristic fallback
        log.warning("LLM emotion tagging returned no tags — using heuristic")
    except Exception as e:
        log.warning(f"Emotion tagging failed: {e} — using heuristic")

    return _tag_emotions_heuristic(script)


def _tag_emotions_heuristic(script: str) -> str:
    """Fallback: heuristic emotion tagging by keyword scanning."""
    from pipeline.tts_engine import detect_emotion
    paragraphs = [p.strip() for p in script.split("\n\n") if p.strip()]
    tagged = []
    for p in paragraphs:
        emotion = detect_emotion(p)
        tagged.append(f"[EMOTION:{emotion}]\n{p}")
    return "\n\n".join(tagged)


def strip_emotion_tags(script: str) -> str:
    """Remove [EMOTION:xxx] tags from script for display."""
    return re.sub(r"\[EMOTION:[^\]]+\]\n?", "", script).strip()


# ── Main script generation ────────────────────────────────────────────────────

def generate_script(
    transcript: str,
    persona_id: str,
    duration: str = "3min",
    language: str = "te",
    content_type: str = "news",    # 'news' | 'analysis' | 'entertainment' | 'educational'
    add_emotion_tags: bool = True,
) -> dict:
    """
    Generate a Telugu anchor script from source transcript.

    Returns:
        {
            script: str (with emotion tags if add_emotion_tags=True),
            script_clean: str (without tags, for display),
            word_count: int,
            estimated_duration_sec: int,
            emotion_map: list of {text, emotion},
        }
    """
    target_words  = DURATION_WORDS.get(duration, 450)
    char_prompt   = CHARACTER_PROMPTS.get(persona_id, CHARACTER_PROMPTS["navya_telugu_f"])
    persona_name  = persona_id.replace("_telugu_f", "").replace("_telugu_m", "").replace("_", " ").title()

    # Content-type specific instructions
    CONTENT_TYPE_HINTS = {
        "news":          "This is a news report. Be factual, clear, and authoritative.",
        "analysis":      "This is a news analysis. Provide context, background, and your perspective.",
        "entertainment": "This is entertainment news. Keep it fun, engaging, and relatable.",
        "educational":   "This is an educational explainer. Break down complex topics simply.",
    }
    content_hint = CONTENT_TYPE_HINTS.get(content_type, CONTENT_TYPE_HINTS["news"])

    prompt = f"""{char_prompt}

{content_hint}

Your task: Rewrite the following content as a professional Telugu video script for {persona_name}.

Requirements:
- Write in natural spoken Telugu (not written/formal Telugu)
- Target length: approximately {target_words} words
- Structure: Opening hook → Main content (3-4 points) → Closing call-to-action
- Opening: Grab attention in first 2 sentences
- Use paragraph breaks between major points
- Closing: End with a memorable line and "likes, shares, subscribe" call
- Do NOT include stage directions, timestamps, or speaker labels
- Write ONLY the script text that will be spoken

SOURCE CONTENT:
{transcript[:3000]}

TELUGU SCRIPT:"""

    log.info(f"Generating script | persona={persona_id} duration={duration} target={target_words}w")
    start = time.time()
    script = _generate_with_fallback(prompt)
    elapsed = time.time() - start
    log.info(f"Script generated in {elapsed:.1f}s | {len(script.split())} words")

    # Add emotion tags
    if add_emotion_tags:
        script_tagged = tag_emotions_llm(script)
    else:
        script_tagged = script

    script_clean = strip_emotion_tags(script_tagged)
    word_count   = len(script_clean.split())
    # Telugu speech rate: ~130 words/min
    est_duration = int((word_count / 130) * 60)

    # Build emotion map
    emotion_map = []
    for para in script_tagged.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"\[EMOTION:([^\]]+)\]", para)
        emotion  = m.group(1) if m else "professional"
        text     = re.sub(r"\[EMOTION:[^\]]+\]\n?", "", para).strip()
        if text:
            emotion_map.append({"text": text, "emotion": emotion})

    return {
        "script":               script_tagged,
        "script_clean":         script_clean,
        "word_count":           word_count,
        "estimated_duration_sec": est_duration,
        "emotion_map":          emotion_map,
    }


# ── Debate script generation ──────────────────────────────────────────────────

def generate_debate(
    transcript: str,
    persona_a_id: str,
    persona_b_id: str,
    duration: str = "5min",
    topic_override: str = "",
) -> dict:
    """
    Generate a debate-style script between two anchors.
    Returns dict with 'script' containing alternating speaker lines.
    """
    target_words = DURATION_WORDS.get(duration, 750)
    name_a = persona_a_id.replace("_telugu_f","").replace("_telugu_m","").replace("_"," ").title()
    name_b = persona_b_id.replace("_telugu_f","").replace("_telugu_m","").replace("_"," ").title()

    topic = topic_override or transcript[:500]

    prompt = f"""Write a Telugu debate-style dialogue between two news anchors discussing:

TOPIC: {topic}

{name_a}: Opens, presents one perspective (3-4 sentences)
{name_b}: Responds with a different viewpoint (3-4 sentences)
{name_a}: Counters with evidence (3-4 sentences)
{name_b}: Makes a strong point (3-4 sentences)
{name_a}: Summarizes and closes (2-3 sentences)

Rules:
- Write in natural spoken Telugu
- Target: {target_words} words total
- Format: Start each line with the speaker's name followed by colon
- No stage directions, just dialogue
- Make it engaging and substantive

DIALOGUE:"""

    script = _generate_with_fallback(prompt)
    script_clean = strip_emotion_tags(script)

    return {
        "script":       script,
        "script_clean": script_clean,
        "word_count":   len(script_clean.split()),
        "format":       "debate",
        "speakers":     [persona_a_id, persona_b_id],
    }


# ── Script quality check ──────────────────────────────────────────────────────

def check_script_quality(script: str) -> dict:
    """
    Basic quality checks on generated script.
    Returns dict with {passed, issues, word_count, has_telugu}.
    """
    clean   = strip_emotion_tags(script)
    words   = clean.split()
    issues  = []

    if len(words) < 50:
        issues.append("Script too short — less than 50 words")

    # Check for Telugu characters
    has_telugu = bool(re.search(r"[ఀ-౿]", clean))
    if not has_telugu:
        issues.append("No Telugu characters found — script may be in wrong language")

    # Check for stage directions that leaked through
    if re.search(r"\(pause\)|\[fade\]|ANCHOR:", clean, re.I):
        issues.append("Stage directions found in script — clean before TTS")

    return {
        "passed":      len(issues) == 0,
        "issues":      issues,
        "word_count":  len(words),
        "has_telugu":  has_telugu,
        "est_duration_sec": int((len(words) / 130) * 60),
    }
