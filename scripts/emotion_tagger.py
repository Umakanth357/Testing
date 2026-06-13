"""
emotion_tagger.py
Analyses a script sentence by sentence and injects emotion + SSML tags.
Primary: Ollama (Gemma3 4B) — 100% local, free, no API key
Fallback: HuggingFace j-hartmann classifier (28MB, CPU)
Fallback 2: Rule-based keyword matching
"""
import re
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── Emotion → Svara/Chatterbox tag mapping ────────────────────────
EMOTION_TAGS = {
    "excited":     {"svara": "<excited>",   "chatterbox": 0.85, "rate": "fast",   "pitch": "+2st"},
    "happy":       {"svara": "<happy>",     "chatterbox": 0.70, "rate": "medium", "pitch": "+1st"},
    "serious":     {"svara": "<clear>",     "chatterbox": 0.30, "rate": "slow",   "pitch": "-1st"},
    "urgent":      {"svara": "<anger>",     "chatterbox": 0.80, "rate": "fast",   "pitch": "+0st"},
    "sad":         {"svara": "<sad>",       "chatterbox": 0.60, "rate": "slow",   "pitch": "-2st"},
    "curious":     {"svara": "<happy>",     "chatterbox": 0.50, "rate": "medium", "pitch": "+0st"},
    "surprised":   {"svara": "<excited>",   "chatterbox": 0.75, "rate": "fast",   "pitch": "+2st"},
    "calm":        {"svara": "<clear>",     "chatterbox": 0.20, "rate": "slow",   "pitch": "-1st"},
    "neutral":     {"svara": "",            "chatterbox": 0.40, "rate": "medium", "pitch": "+0st"},
    "warm":        {"svara": "<happy>",     "chatterbox": 0.45, "rate": "medium", "pitch": "+1st"},
    "concerned":   {"svara": "<sad>",       "chatterbox": 0.55, "rate": "slow",   "pitch": "-1st"},
    "enthusiastic":{"svara": "<excited>",   "chatterbox": 0.80, "rate": "fast",   "pitch": "+2st"},
}

# Rule-based keyword → emotion (fallback)
KEYWORD_MAP = {
    "excited":    ["incredible", "amazing", "wow", "fantastic", "awesome", "brilliant", "love", "best"],
    "urgent":     ["critical", "important", "immediately", "alert", "warning", "must", "urgent", "now"],
    "happy":      ["good", "great", "nice", "excellent", "congratulations", "well done", "proud"],
    "sad":        ["unfortunately", "sadly", "failed", "problem", "issue", "concern", "disappointed"],
    "curious":    ["interesting", "wonder", "think about", "consider", "what if", "imagine", "notice"],
    "surprised":  ["surprisingly", "unexpected", "suddenly", "wait", "actually", "turns out"],
    "serious":    ["note", "remember", "important", "security", "critical", "must know", "attention"],
    "warm":       ["welcome", "thank", "appreciate", "glad", "hope", "looking forward", "together"],
}


def _rule_based_emotion(sentence: str) -> str:
    """Fast keyword-based emotion detection. No model needed."""
    s = sentence.lower()
    scores = {e: 0 for e in KEYWORD_MAP}
    for emotion, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in s:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


def _hf_emotion(sentences: List[str]) -> List[str]:
    """HuggingFace distilroberta emotion classifier — 28MB CPU model."""
    try:
        from transformers import pipeline
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )
        results = classifier(sentences, truncation=True, max_length=128)
        label_map = {
            "joy": "happy", "anger": "urgent", "fear": "concerned",
            "sadness": "sad", "surprise": "surprised", "disgust": "serious",
            "neutral": "neutral"
        }
        return [label_map.get(r[0]["label"].lower(), "neutral") for r in results]
    except Exception as e:
        logger.warning(f"HF classifier failed: {e}. Using rule-based.")
        return [_rule_based_emotion(s) for s in sentences]


def _ollama_emotion(sentences: List[str]) -> List[str]:
    """Ollama Gemma3 4B — best quality, fully local."""
    try:
        import urllib.request
        prompt = f"""You are an emotion detector for a text-to-speech system.
For each sentence below, output ONLY a JSON array of emotion labels.
Choose from: excited, happy, serious, urgent, sad, curious, surprised, calm, neutral, warm, concerned, enthusiastic

Sentences:
{json.dumps(sentences, ensure_ascii=False)}

Respond with ONLY a JSON array, example: ["happy", "neutral", "urgent"]
No explanation, no markdown, just the JSON array."""

        payload = json.dumps({
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200}
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        response_text = data.get("response", "")

        # Parse JSON array from response
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            emotions = json.loads(match.group())
            if len(emotions) == len(sentences):
                # Validate all are known emotions
                valid = [e if e in EMOTION_TAGS else "neutral" for e in emotions]
                return valid
        raise ValueError("Could not parse emotion array from Ollama response")
    except Exception as e:
        logger.warning(f"Ollama emotion tagging failed: {e}. Falling back to HF.")
        return _hf_emotion(sentences)


def tag_script(script: str) -> List[Dict]:
    """
    Main entry point. Takes a plain script string.
    Returns list of dicts: {sentence, emotion, svara_tag, chatterbox_exaggeration, ssml_rate, ssml_pitch, ssml_sentence}

    Example output item:
    {
        "sentence": "Today we had an incredible hackathon!",
        "emotion": "excited",
        "svara_tag": "<excited>",
        "chatterbox_exaggeration": 0.85,
        "ssml_rate": "fast",
        "ssml_pitch": "+2st",
        "ssml_sentence": '<prosody rate="fast" pitch="+2st">Today we had an incredible hackathon!</prosody>'
    }
    """
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script.strip()) if s.strip()]
    if not sentences:
        return []

    logger.info(f"Tagging {len(sentences)} sentences for emotion...")

    # Try Ollama first, cascade through fallbacks
    emotions = _ollama_emotion(sentences)

    result = []
    for sentence, emotion in zip(sentences, emotions):
        tag_info = EMOTION_TAGS.get(emotion, EMOTION_TAGS["neutral"])

        # Add SSML prosody + natural breath pause
        rate = tag_info["rate"]
        pitch = tag_info["pitch"]
        ssml = f'<prosody rate="{rate}" pitch="{pitch}">{sentence}</prosody>'

        # Add sentence-end pause (longer for serious/urgent, shorter for excited)
        pause_ms = {"slow": 600, "medium": 400, "fast": 200}.get(rate, 400)
        ssml += f'<break time="{pause_ms}ms"/>'

        result.append({
            "sentence": sentence,
            "emotion": emotion,
            "svara_tag": tag_info["svara"],
            "chatterbox_exaggeration": tag_info["chatterbox"],
            "ssml_rate": rate,
            "ssml_pitch": pitch,
            "ssml_sentence": ssml,
        })

    logger.info("Emotion tagging complete")
    return result


def build_tagged_script(tagged: List[Dict], language: str = "en") -> str:
    """
    Build the final TTS-ready string with emotion tags.
    For Svara TTS: appends <emotion> tags per sentence.
    For Chatterbox: returns plain sentences (exaggeration set per-segment in TTS layer).
    """
    parts = []
    for item in tagged:
        sentence = item["sentence"]
        tag = item["svara_tag"]
        if language in ["te", "kn", "ta", "hi", "ml"] and tag:
            parts.append(f"{sentence} {tag}")
        else:
            parts.append(sentence)
    return " ".join(parts)


if __name__ == "__main__":
    # Quick self-test
    test_script = """
    Welcome everyone to today's KT session on our new deployment pipeline.
    This is incredibly exciting — we've reduced deployment time by 80 percent!
    However, there is one critical security concern you must be aware of.
    If you ignore this warning, production systems could be affected.
    I'm so proud of the team for pulling this off in just two weeks.
    Let's walk through each component carefully.
    """
    print("Running emotion tagger self-test...\n")
    tagged = tag_script(test_script)
    for item in tagged:
        print(f"  [{item['emotion']:12s}] {item['sentence'][:60]}...")
    print(f"\n✓ Tagged {len(tagged)} sentences successfully")
