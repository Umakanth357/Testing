"""
Content Calendar — Format rotation and AI slop prevention.

Problems this solves:
  1. Same format 3 episodes in a row → viewers tune out
  2. Same topic category 3 in a row → channel becomes one-note
  3. No variety in scene backgrounds → feels low-budget
  4. Missing content types (no debate this week?) → misses engagement peaks
  5. Kavya only, no Arjun episodes → character development stalls

Rules:
  - Never same format 3x in a row
  - Bigg Boss content: max 3 per week (audience burns out)
  - At least 1 debate per 5 episodes
  - At least 1 tech review per 7 episodes
  - Suggest next content type based on recent history
"""

from pathlib import Path
from typing import Optional
import sqlite3
import logging

log = logging.getLogger("content_calendar")

ROOT      = Path(__file__).parent.parent.resolve()
DB_PATH   = ROOT / "data" / "content_memory.db"

# Format and category rotation rules
FORMAT_RULES = {
    "monologue":   {"max_consecutive": 2, "weight": 3},
    "debate":      {"max_consecutive": 2, "weight": 2, "min_interval": 5},
    "news_anchor": {"max_consecutive": 2, "weight": 2},
    "seminar":     {"max_consecutive": 1, "weight": 1, "min_interval": 7},
    "short":       {"max_consecutive": 3, "weight": 1},
}

CATEGORY_RULES = {
    "bigg_boss":      {"max_per_week": 3, "max_consecutive": 3},
    "movie_review":   {"max_per_week": 4, "max_consecutive": 2},
    "tech_review":    {"max_per_week": 5, "max_interval": 7},
    "debate":         {"max_per_week": 2, "min_interval": 5},
    "festival":       {"max_per_week": 2},
    "general":        {"max_per_week": 10},
}

FORMAT_SEQUENCE_SUGGESTIONS = {
    # after N consecutive monologues → suggest debate
    ("monologue", "monologue"):    ["debate", "news_anchor"],
    ("debate", "debate"):          ["monologue", "seminar"],
    ("news_anchor", "news_anchor"):["monologue", "debate"],
    ("monologue",):                ["debate", "monologue", "news_anchor"],
}

CATEGORY_SEQUENCE_SUGGESTIONS = {
    ("bigg_boss", "bigg_boss", "bigg_boss"): ["movie_review", "tech_review"],
    ("tech_review", "tech_review"):           ["bigg_boss", "debate"],
    ("movie_review", "movie_review"):         ["bigg_boss", "tech_review"],
}


def _get_recent(table_col: str, n: int = 5) -> list[str]:
    """Get recent values from episode history."""
    try:
        conn  = sqlite3.connect(str(DB_PATH))
        col   = table_col.split(".")[-1]
        table = table_col.split(".")[0] if "." in table_col else "episodes"
        rows  = conn.execute(f"""
            SELECT {col} FROM {table}
            ORDER BY rowid DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def get_recent_formats(n: int = 5) -> list[str]:
    """Recent episode format_types, newest first."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("""
            SELECT ef.format_type FROM episode_formats_log ef
            JOIN episodes ep ON ep.id = ef.episode_id
            ORDER BY ep.episode_number DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def get_recent_categories(n: int = 5) -> list[str]:
    """Recent episode topic_categories, newest first."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("""
            SELECT topic_category FROM episodes
            ORDER BY episode_number DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def check_format_allowed(proposed_format: str) -> tuple[bool, str]:
    """
    Check if a format is allowed given recent history.
    Returns (allowed, reason).
    """
    recent = get_recent_formats(5)
    if not recent:
        return True, "No history"

    rule      = FORMAT_RULES.get(proposed_format, {})
    max_consec = rule.get("max_consecutive", 99)

    # Count consecutive at start of history
    consecutive = 0
    for f in recent:
        if f == proposed_format:
            consecutive += 1
        else:
            break

    if consecutive >= max_consec:
        return False, f"Already used {proposed_format} {consecutive}x in a row (max {max_consec})"

    # Check min_interval for rare formats (debate, seminar)
    min_interval = rule.get("min_interval", 0)
    if min_interval > 0 and proposed_format in recent[:min_interval]:
        last_pos = recent.index(proposed_format) + 1
        return False, f"{proposed_format} was used {last_pos} episodes ago (needs {min_interval} gap)"

    return True, "OK"


def suggest_next_format(topic_category: str = "general") -> str:
    """
    Suggest the best format for next episode based on history.
    Returns format string.
    """
    recent = get_recent_formats(5)
    recent_tuple = tuple(recent[:2]) if len(recent) >= 2 else tuple(recent[:1])

    # Use sequence rules
    if recent_tuple in FORMAT_SEQUENCE_SUGGESTIONS:
        candidates = FORMAT_SEQUENCE_SUGGESTIONS[recent_tuple]
        for candidate in candidates:
            allowed, _ = check_format_allowed(candidate)
            if allowed:
                return candidate

    # Category-based override: if tech review → seminar works well
    if topic_category in ("tech_review", "market_analysis"):
        allowed, _ = check_format_allowed("seminar")
        if allowed:
            return "seminar"

    if topic_category in ("bigg_boss", "entertainment", "celebrity_gossip"):
        allowed, _ = check_format_allowed("monologue")
        if allowed:
            return "monologue"

    # Default: pick most underused
    counts = {f: recent.count(f) for f in FORMAT_RULES}
    allowed_formats = [f for f in FORMAT_RULES if check_format_allowed(f)[0]]
    if allowed_formats:
        return min(allowed_formats, key=lambda f: counts.get(f, 0))

    return "monologue"


def check_category_allowed(topic_category: str) -> tuple[bool, str]:
    """
    Check if a topic category is allowed this week.
    Returns (allowed, reason).
    """
    rule = CATEGORY_RULES.get(topic_category, CATEGORY_RULES["general"])
    recent_cats = get_recent_categories(10)

    # Check consecutive
    max_consec = rule.get("max_consecutive", 99)
    consecutive = 0
    for cat in recent_cats:
        if cat == topic_category:
            consecutive += 1
        else:
            break

    if consecutive >= max_consec:
        return False, f"Already {consecutive} consecutive {topic_category} episodes"

    # Check min_interval
    min_interval = rule.get("min_interval", 0)
    if min_interval and topic_category in recent_cats[:min_interval]:
        return False, f"{topic_category} too recent"

    return True, "OK"


def suggest_next_character(topic_category: str = "general") -> tuple[str, str]:
    """
    Suggest which character should be primary for next episode.
    Returns (primary, secondary).
    """
    from pipeline.character_bible import who_leads_topic
    return who_leads_topic(topic_category)


def get_content_health_report() -> dict:
    """
    Full calendar health check — call before scheduling next episode.
    Returns dict with warnings, suggestions, and content gaps.
    """
    recent_formats    = get_recent_formats(10)
    recent_categories = get_recent_categories(10)

    warnings     = []
    suggestions  = []

    # Check debate gap
    debate_positions = [i for i, f in enumerate(recent_formats) if f == "debate"]
    if not debate_positions or debate_positions[0] > 5:
        warnings.append("No debate in last 5 episodes — audience engagement dropping")
        suggestions.append("Schedule a debate episode next")

    # Check Bigg Boss saturation
    bb_count = recent_categories[:7].count("bigg_boss")
    if bb_count > 3:
        warnings.append(f"Bigg Boss appeared {bb_count}x in last 7 episodes — diversify")
        suggestions.append("Add a tech review or movie review")

    # Check tech review gap
    if "tech_review" not in recent_categories[:7]:
        suggestions.append("No tech review in 7 episodes — consider adding one")

    # Check format variety
    unique_formats = len(set(recent_formats[:5]))
    if unique_formats == 1:
        warnings.append(f"All last 5 episodes used same format: {recent_formats[0]}")
        suggestions.append(f"Next episode should NOT be {recent_formats[0]}")

    return {
        "recent_formats":       recent_formats[:5],
        "recent_categories":    recent_categories[:5],
        "warnings":             warnings,
        "suggestions":          suggestions,
        "next_format_suggest":  suggest_next_format(),
        "format_variety_score": unique_formats / 5.0,
    }
