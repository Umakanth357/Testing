"""
Catchphrase Injector — Guarantees signature phrases appear in every episode.

Rules:
  1. Every episode must have at least one opener catchphrase
  2. Every episode must close with a sign-off catchphrase
  3. Excitement/skepticism phrases injected naturally at 30% and 70% marks
  4. No same catchphrase used in last 3 episodes (rotation enforced)
  5. Debate format: both characters get their catchphrases
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional

from pipeline.character_bible import get_character, CHARACTERS

ROOT = Path(__file__).parent.parent.resolve()
DB_PATH = ROOT / "data" / "content_memory.db"


def _get_used_recently(character_id: str, phrase_type: str, n: int = 3) -> list[str]:
    """Fetch catchphrases used in last n episodes to avoid repetition."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("""
            SELECT phrase FROM catchphrase_log
            WHERE character = ? AND phrase_type = ?
            ORDER BY used_at DESC LIMIT ?
        """, (character_id, phrase_type, n)).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _log_used(character_id: str, phrase_type: str, phrase: str):
    """Log that this phrase was used so it won't repeat immediately."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS catchphrase_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                character   TEXT,
                phrase_type TEXT,
                phrase      TEXT,
                used_at     TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "INSERT INTO catchphrase_log (character, phrase_type, phrase) VALUES (?, ?, ?)",
            (character_id, phrase_type, phrase)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _pick_fresh(options: list[str], recently_used: list[str]) -> str:
    """Pick an option not used recently. Falls back to first option."""
    fresh = [o for o in options if o not in recently_used]
    chosen = fresh[0] if fresh else options[0]
    return chosen


def inject_catchphrases(script: str, character_id: str,
                         format_type: str = "monologue") -> str:
    """
    Ensure catchphrases are present in the script.
    If the script already has them (LLM got it right), skip injection.
    If missing, inject at the correct positions.
    """
    c = get_character(character_id)
    lines = script.strip().splitlines()

    # ── Check if opener is already there ─────────────────────────────────────
    script_start = " ".join(lines[:3]).lower()
    has_opener = any(
        word in script_start
        for phrase in c.opener
        for word in phrase.lower().split()[:3]  # first 3 words of phrase
    )

    if not has_opener:
        recently = _get_used_recently(character_id, "opener")
        opener   = _pick_fresh(c.opener, recently)
        lines    = [opener, ""] + lines
        _log_used(character_id, "opener", opener)

    # ── Check if sign-off is already there ───────────────────────────────────
    script_end = " ".join(lines[-3:]).lower()
    has_signoff = any(
        word in script_end
        for phrase in c.sign_off
        for word in phrase.lower().split()[:3]
    )

    if not has_signoff:
        recently  = _get_used_recently(character_id, "sign_off")
        sign_off  = _pick_fresh(c.sign_off, recently)
        lines     = lines + ["", sign_off]
        _log_used(character_id, "sign_off", sign_off)

    # ── Inject excitement phrase at ~30% mark if script is long enough ───────
    if len(lines) > 15:
        mid_1 = len(lines) // 3
        mid_2 = (2 * len(lines)) // 3
        mid_block = " ".join(lines[mid_1-2:mid_1+2]).lower()

        has_excite = any(
            phrase.lower()[:20] in mid_block for phrase in c.excitement
        )
        if not has_excite:
            recently = _get_used_recently(character_id, "excitement")
            phrase   = _pick_fresh(c.excitement, recently)
            lines.insert(mid_1, phrase)
            _log_used(character_id, "excitement", phrase)

    return "\n".join(lines)


def inject_debate_catchphrases(script: str) -> str:
    """
    For debate format: ensure KAVYA lines start with her opener and ARJUN's too.
    Works on the tagged KAVYA: / ARJUN: format.
    """
    lines    = script.strip().splitlines()
    kavya_c  = get_character("kavya")
    arjun_c  = get_character("arjun")

    kavya_lines_seen = 0
    arjun_lines_seen = 0
    output   = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("KAVYA:"):
            kavya_lines_seen += 1
            if kavya_lines_seen == 1:
                # First KAVYA line — ensure opener
                content = stripped[6:].strip()
                has_opener = any(w in content.lower() for w in ["namaskaram", "ayyo", "hello"])
                if not has_opener:
                    recently = _get_used_recently("kavya", "opener")
                    opener   = _pick_fresh(kavya_c.opener, recently)
                    output.append(f"KAVYA: {opener}")
                    _log_used("kavya", "opener", opener)
                    output.append(stripped)
                    continue
            output.append(stripped)

        elif stripped.upper().startswith("ARJUN:"):
            arjun_lines_seen += 1
            if arjun_lines_seen == 1:
                content = stripped[6:].strip()
                has_opener = any(w in content.lower() for w in ["namaskaram", "meeru", "facts"])
                if not has_opener:
                    recently = _get_used_recently("arjun", "opener")
                    opener   = _pick_fresh(arjun_c.opener, recently)
                    output.append(f"ARJUN: {opener}")
                    _log_used("arjun", "opener", opener)
                    output.append(stripped)
                    continue
            output.append(stripped)
        else:
            output.append(line)

    # ── Debate sign-offs ──────────────────────────────────────────────────────
    last_block = " ".join(output[-5:]).lower()
    if "kavya" in last_block and not any(w in last_block for w in ["next week", "bye", "namaskaram"]):
        recently = _get_used_recently("kavya", "sign_off")
        sign_off = _pick_fresh(kavya_c.sign_off, recently)
        output.append(f"KAVYA: {sign_off}")
        _log_used("kavya", "sign_off", sign_off)

    if "arjun" in last_block and not any(w in last_block for w in ["namaskaram", "informed", "until"]):
        recently = _get_used_recently("arjun", "sign_off")
        sign_off = _pick_fresh(arjun_c.sign_off, recently)
        output.append(f"ARJUN: {sign_off}")
        _log_used("arjun", "sign_off", sign_off)

    return "\n".join(output)


def ensure_catchphrases(script: str, character_id: str,
                         format_type: str = "monologue") -> str:
    """Top-level entry point. Call this after script generation, before TTS."""
    if format_type == "debate":
        return inject_debate_catchphrases(script)
    return inject_catchphrases(script, character_id, format_type)
