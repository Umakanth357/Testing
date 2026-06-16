"""
Prediction Tracker — Arjun and Kavya make predictions. We track them. Publicly.

This is a key engagement mechanic:
  - Characters make confident predictions on Bigg Boss, tech, movies, etc.
  - Predictions shown on-screen with confidence level and episode reference
  - When resolved: callback episode references the original prediction
  - Running accuracy score shown each episode ("Kavya: 72% correct")
  - Wrong predictions acknowledged publicly — builds authenticity

Integration:
  - Script engine extracts predictions from generated script
  - Prediction is stored before episode goes live
  - Resolution monitored manually (for now) — future: auto-resolve from YouTube scraping
  - Callback lines auto-injected into next relevant episode
"""

import re
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("prediction_tracker")

ROOT    = Path(__file__).parent.parent.resolve()


# ── Prediction Extraction ─────────────────────────────────────────────────────

PREDICTION_TRIGGERS = [
    r"naadu prediction",
    r"nenu cheptunna",
    r"mark chesukundi",
    r"ee week lo",
    r"confirm ga",
    r"i predict",
    r"my prediction",
    r"based on.*pattern",
    r"meeru note chesukundi",
    r"Ep record",
    r"crystal ball",
    r"wrong aite.*sorry",
    r"confident.*cheptunna",
]

CONFIDENCE_KEYWORDS = {
    "high":   ["confirm ga", "100%", "definitely", "no doubt", "pakka"],
    "medium": ["nenu anukuntunna", "i think", "likely", "probably", "anipistundi"],
    "low":    ["maybe", "em teliyadu", "not sure", "perhaps", "ayyuntundi"],
}


def extract_predictions_from_script(script: str, character_id: str) -> list[dict]:
    """
    Extract prediction statements from a generated script.
    Returns list of {text, confidence, entity_hint} dicts.
    """
    predictions = []
    pattern     = re.compile(
        "|".join(PREDICTION_TRIGGERS), re.IGNORECASE
    )

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', script)
    for i, sentence in enumerate(sentences):
        if not pattern.search(sentence):
            continue

        # Grab this sentence + next one (prediction often spans 2 sentences)
        text = sentence
        if i + 1 < len(sentences):
            text = f"{sentence} {sentences[i+1]}"

        # Determine confidence
        confidence = "medium"
        text_lower = text.lower()
        for level, keywords in CONFIDENCE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                confidence = level
                break

        # Extract entity hint (capitalized words)
        entities = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        entity_hint = entities[0] if entities else None

        predictions.append({
            "character":    character_id,
            "text":         text.strip(),
            "confidence":   confidence,
            "entity_hint":  entity_hint,
        })

    log.info(f"Extracted {len(predictions)} predictions for {character_id}")
    return predictions


def save_predictions_from_script(script: str, character_id: str,
                                  episode_id: int, format_type: str = "monologue"):
    """
    Extract predictions from script and save them to memory DB.
    Call this after script approval, before TTS.
    """
    from pipeline.memory_db import save_prediction, get_or_create_entity

    if format_type == "debate":
        # Extract for both characters
        for char_id in ["kavya", "arjun"]:
            # Filter lines for this character
            char_lines = []
            for line in script.splitlines():
                if line.upper().startswith(f"{char_id.upper()}:"):
                    char_lines.append(line[len(char_id)+1:].strip())
            char_script = " ".join(char_lines)
            preds = extract_predictions_from_script(char_script, char_id)
            _save_pred_list(preds, char_id, episode_id)
    else:
        preds = extract_predictions_from_script(script, character_id)
        _save_pred_list(preds, character_id, episode_id)


def _save_pred_list(preds: list[dict], character_id: str, episode_id: int):
    from pipeline.memory_db import save_prediction, get_or_create_entity
    for p in preds:
        entity_id = None
        if p.get("entity_hint"):
            try:
                entity_id = get_or_create_entity(p["entity_hint"], "entity")
            except Exception:
                pass
        try:
            save_prediction(
                episode_id      = episode_id,
                character       = character_id,
                prediction_text = p["text"],
                confidence      = p["confidence"],
            )
        except Exception as e:
            log.warning(f"Could not save prediction: {e}")


# ── Callback Generation ───────────────────────────────────────────────────────

def build_prediction_callback_lines(topic: str, character_id: str) -> str:
    """
    Build natural callback lines for open predictions relevant to this topic.
    Injected into the script context so LLM can reference them organically.
    """
    from pipeline.memory_db import get_open_predictions, get_recently_wrong_predictions
    from pipeline.character_bible import get_character

    c      = get_character(character_id)
    lines  = []

    open_preds = get_open_predictions(character_id)
    # Filter only those that might relate to this topic
    relevant = [
        p for p in open_preds
        if topic.lower() in (p.get("prediction_text") or "").lower()
        or any(w in (p.get("prediction_text") or "").lower()
               for w in topic.lower().split())
    ]

    if relevant:
        lines.append(f"\n{c.display_name}'s OPEN PREDICTION TO REFERENCE:")
        for p in relevant[:2]:
            lines.append(
                f"  Ep {p['episode_number']}: \"{p['prediction_text']}\" "
                f"(confidence: {p['confidence']})"
            )
        lines.append("→ If this episode resolves it, mention it naturally")

    wrong = get_recently_wrong_predictions(30)
    char_wrong = [w for w in wrong if w["character"] == character_id]
    if char_wrong:
        wrong_p = char_wrong[0]
        lines.append(f"\nRECENT WRONG PREDICTION (acknowledge for authenticity):")
        lines.append(f"  \"{wrong_p['prediction_text']}\" — was wrong ({wrong_p['resolution_note']})")

    return "\n".join(lines)


# ── On-Screen Display Data ────────────────────────────────────────────────────

def get_prediction_scoreboard() -> dict:
    """
    Get current prediction accuracy for both characters.
    Used by compose_engine to render the prediction scorecard graphic.
    """
    from pipeline.memory_db import get_character_accuracy
    stats = get_character_accuracy()
    return {
        "kavya": {
            "total":    stats.get("kavya", {}).get("total_predictions", 0),
            "correct":  stats.get("kavya", {}).get("correct_predictions", 0),
            "accuracy": stats.get("kavya", {}).get("accuracy_pct", 0.0),
        },
        "arjun": {
            "total":    stats.get("arjun", {}).get("total_predictions", 0),
            "correct":  stats.get("arjun", {}).get("correct_predictions", 0),
            "accuracy": stats.get("arjun", {}).get("accuracy_pct", 0.0),
        },
    }


def format_scoreboard_text(scoreboard: dict) -> str:
    """Format for FFmpeg drawtext overlay."""
    k = scoreboard["kavya"]
    a = scoreboard["arjun"]
    return (
        f"Kavya: {k['accuracy']}% ({k['correct']}/{k['total']})  |  "
        f"Arjun: {a['accuracy']}% ({a['correct']}/{a['total']})"
    )


# ── Manual Resolution CLI ─────────────────────────────────────────────────────

def resolve_interactively():
    """
    CLI tool to resolve open predictions.
    Run: python -m pipeline.prediction_tracker
    """
    from pipeline.memory_db import get_open_predictions, resolve_prediction

    open_preds = get_open_predictions()
    if not open_preds:
        print("No open predictions.")
        return

    print(f"\n=== Open Predictions ({len(open_preds)}) ===\n")
    for i, p in enumerate(open_preds):
        print(f"{i+1}. [{p['character'].upper()}] Ep {p['episode_number']}: {p['prediction_text']}")
        print(f"   Confidence: {p['confidence']} | Made: {p['created_at'][:10]}")
        print()

    idx = input("Enter prediction number to resolve (or 'q' to quit): ").strip()
    if idx.lower() == 'q':
        return

    try:
        p = open_preds[int(idx) - 1]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    outcome = input(f"Was this prediction CORRECT? (y/n): ").strip().lower()
    correct = outcome == 'y'
    note    = input("Resolution note (what actually happened): ").strip()
    ep_num  = input("Which episode is resolving this?: ").strip()

    try:
        ep_id = int(ep_num)
    except ValueError:
        ep_id = None

    resolve_prediction(p["id"], correct, note, ep_id)
    result = "CORRECT ✓" if correct else "WRONG ✗"
    print(f"\nResolved: {result}")
    print("Prediction tracker updated. Next episode script will acknowledge this.")


if __name__ == "__main__":
    resolve_interactively()
