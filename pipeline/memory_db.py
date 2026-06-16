"""
Content Knowledge Database — The Script Writer's Memory

This is NOT anchor memory. This is CONTENT intelligence.
The script writer queries this before writing every episode.

What it stores:
  - Every topic ever covered (with episode reference)
  - Entity tracking (tools, movies, contestants, people)
  - Claims made on air (verified or refuted later)
  - Predictions (with outcomes)
  - Topic correlations (Tool A → similar to Tool B)
  - Keyword index for fast retrieval

How it's used:
  - "We are reviewing Tool X → find related past coverage"
  - "We predicted Contestant Y would win → check if resolved"
  - "We said Tool Z had bad Telugu support in Ep 12 → follow up?"
  - "What episode formats have we done last 5 episodes?"
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional

ROOT = Path(__file__).parent.parent.resolve()
DB_PATH = ROOT / "data" / "content_memory.db"

log = logging.getLogger("memory_db")


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_number  INTEGER UNIQUE,
    date            TEXT NOT NULL,
    title           TEXT NOT NULL,
    topic_category  TEXT,           -- bigg_boss | movie_review | tech_review | debate | festival
    format_type     TEXT,           -- monologue | debate | news_anchor | seminar
    summary         TEXT,           -- 2-3 sentence summary for future context injection
    script_path     TEXT,
    video_path      TEXT,
    duration_sec    INTEGER,
    youtube_url     TEXT,
    view_count      INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,  -- tool | movie | person | contestant | place | brand | concept
    aliases         TEXT,           -- JSON array of alternate names
    first_episode   INTEGER REFERENCES episodes(id),
    last_episode    INTEGER REFERENCES episodes(id),
    mention_count   INTEGER DEFAULT 1,
    notes           TEXT            -- any important facts about this entity
);

CREATE TABLE IF NOT EXISTS episode_entities (
    episode_id      INTEGER REFERENCES episodes(id),
    entity_id       INTEGER REFERENCES entities(id),
    context         TEXT,           -- how entity was mentioned in this episode
    sentiment       TEXT,           -- positive | negative | neutral | mixed
    PRIMARY KEY (episode_id, entity_id)
);

CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER REFERENCES episodes(id),
    claim_text      TEXT NOT NULL,
    entity_id       INTEGER REFERENCES entities(id),
    claim_type      TEXT,           -- fact | opinion | prediction | comparison
    verified        INTEGER DEFAULT 0,  -- 0=unverified, 1=confirmed, -1=refuted
    verification_episode INTEGER REFERENCES episodes(id),
    verification_note TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER REFERENCES episodes(id),
    character       TEXT NOT NULL,  -- kavya | arjun | both
    prediction_text TEXT NOT NULL,
    entity_id       INTEGER REFERENCES entities(id),
    confidence      TEXT,           -- high | medium | low
    resolved        INTEGER DEFAULT 0,
    outcome_correct INTEGER,        -- 1=correct, 0=wrong, NULL=unresolved
    resolution_date TEXT,
    resolution_episode INTEGER REFERENCES episodes(id),
    resolution_note TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topic_correlations (
    entity_a        INTEGER REFERENCES entities(id),
    entity_b        INTEGER REFERENCES entities(id),
    correlation_type TEXT,          -- same_category | competitor | sequel | related_person
    strength        REAL DEFAULT 0.5,  -- 0.0 to 1.0
    notes           TEXT,
    PRIMARY KEY (entity_a, entity_b)
);

CREATE TABLE IF NOT EXISTS episode_formats_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER REFERENCES episodes(id),
    format_type     TEXT NOT NULL,
    scene_key       TEXT,
    avatar_a        TEXT,
    avatar_b        TEXT,
    background      TEXT
);

CREATE TABLE IF NOT EXISTS character_stats (
    character       TEXT PRIMARY KEY,
    total_predictions   INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_pct        REAL DEFAULT 0.0,
    signature_wins      INTEGER DEFAULT 0,  -- prediction everyone doubted but correct
    last_updated        TEXT
);

-- Full text search index on episode summaries and claims
CREATE VIRTUAL TABLE IF NOT EXISTS fts_content
    USING fts5(episode_id, text_content, entity_names);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_claims_episode ON claims(episode_id);
CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions(resolved);
CREATE INDEX IF NOT EXISTS idx_episode_date ON episodes(date);
"""


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables. Safe to call multiple times."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    log.info(f"Content memory DB ready: {DB_PATH}")


# ── Episode Operations ────────────────────────────────────────────────────────

def save_episode(
    title: str,
    topic_category: str,
    format_type: str,
    summary: str,
    date_str: str = None,
    script_path: str = None,
    video_path: str = None,
    duration_sec: int = None,
    youtube_url: str = None,
) -> int:
    """Save a new episode. Returns episode ID."""
    if date_str is None:
        date_str = date.today().isoformat()

    with get_connection() as conn:
        # Get next episode number
        row = conn.execute("SELECT MAX(episode_number) FROM episodes").fetchone()
        ep_num = (row[0] or 0) + 1

        cur = conn.execute("""
            INSERT INTO episodes
              (episode_number, date, title, topic_category, format_type,
               summary, script_path, video_path, duration_sec, youtube_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ep_num, date_str, title, topic_category, format_type,
              summary, script_path, video_path, duration_sec, youtube_url))

        ep_id = cur.lastrowid
        log.info(f"Saved episode #{ep_num}: {title} (id={ep_id})")
        return ep_id


def get_recent_episodes(n: int = 5) -> list[dict]:
    """Get n most recent episodes for context injection."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, episode_number, date, title, topic_category,
                   format_type, summary
            FROM episodes
            ORDER BY episode_number DESC
            LIMIT ?
        """, (n,)).fetchall()
        return [dict(r) for r in rows]


def get_episodes_by_category(category: str, limit: int = 10) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, episode_number, date, title, summary
            FROM episodes WHERE topic_category = ?
            ORDER BY episode_number DESC LIMIT ?
        """, (category, limit)).fetchall()
        return [dict(r) for r in rows]


# ── Entity Operations ─────────────────────────────────────────────────────────

def get_or_create_entity(name: str, entity_type: str,
                          aliases: list[str] = None) -> int:
    """Get existing entity or create new one. Returns entity ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM entities WHERE name = ? OR aliases LIKE ?",
            (name, f'%"{name}"%')
        ).fetchone()

        if row:
            conn.execute(
                "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                (row["id"],)
            )
            return row["id"]

        aliases_json = json.dumps(aliases or [])
        cur = conn.execute("""
            INSERT INTO entities (name, entity_type, aliases)
            VALUES (?, ?, ?)
        """, (name, entity_type, aliases_json))
        return cur.lastrowid


def link_entity_to_episode(episode_id: int, entity_id: int,
                             context: str = "", sentiment: str = "neutral"):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO episode_entities
              (episode_id, entity_id, context, sentiment)
            VALUES (?, ?, ?, ?)
        """, (episode_id, entity_id, context, sentiment))
        conn.execute("""
            UPDATE entities SET last_episode = ?, mention_count = mention_count + 1
            WHERE id = ?
        """, (episode_id, entity_id))


def find_related_entities(entity_name: str, limit: int = 5) -> list[dict]:
    """Find entities correlated to this one — for cross-episode references."""
    with get_connection() as conn:
        entity = conn.execute(
            "SELECT id FROM entities WHERE name LIKE ?", (f"%{entity_name}%",)
        ).fetchone()

        if not entity:
            return []

        # Direct correlations
        rows = conn.execute("""
            SELECT e.name, e.entity_type, tc.correlation_type, tc.strength,
                   ep.episode_number, ep.title
            FROM topic_correlations tc
            JOIN entities e ON (
                CASE WHEN tc.entity_a = ? THEN tc.entity_b ELSE tc.entity_a END = e.id
            )
            JOIN episode_entities ee ON ee.entity_id = e.id
            JOIN episodes ep ON ep.id = ee.episode_id
            WHERE tc.entity_a = ? OR tc.entity_b = ?
            ORDER BY tc.strength DESC, ep.episode_number DESC
            LIMIT ?
        """, (entity["id"], entity["id"], entity["id"], limit)).fetchall()

        return [dict(r) for r in rows]


def add_correlation(entity_a_name: str, entity_b_name: str,
                    correlation_type: str, strength: float = 0.7, notes: str = ""):
    """Add a correlation between two entities."""
    with get_connection() as conn:
        a = conn.execute("SELECT id FROM entities WHERE name = ?",
                         (entity_a_name,)).fetchone()
        b = conn.execute("SELECT id FROM entities WHERE name = ?",
                         (entity_b_name,)).fetchone()
        if not a or not b:
            return

        conn.execute("""
            INSERT OR REPLACE INTO topic_correlations
              (entity_a, entity_b, correlation_type, strength, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (a["id"], b["id"], correlation_type, strength, notes))


# ── Claims ────────────────────────────────────────────────────────────────────

def save_claim(episode_id: int, claim_text: str, claim_type: str,
               entity_name: str = None) -> int:
    """Save a claim made in an episode."""
    with get_connection() as conn:
        entity_id = None
        if entity_name:
            row = conn.execute(
                "SELECT id FROM entities WHERE name LIKE ?", (f"%{entity_name}%",)
            ).fetchone()
            if row:
                entity_id = row["id"]

        cur = conn.execute("""
            INSERT INTO claims (episode_id, claim_text, entity_id, claim_type)
            VALUES (?, ?, ?, ?)
        """, (episode_id, claim_text, entity_id, claim_type))
        return cur.lastrowid


def get_unverified_claims(entity_name: str = None) -> list[dict]:
    """Get claims that haven't been verified yet — candidates for follow-up."""
    with get_connection() as conn:
        if entity_name:
            rows = conn.execute("""
                SELECT c.id, c.claim_text, c.claim_type, c.created_at,
                       ep.episode_number, ep.title, e.name as entity_name
                FROM claims c
                JOIN episodes ep ON ep.id = c.episode_id
                LEFT JOIN entities e ON e.id = c.entity_id
                WHERE c.verified = 0
                  AND (e.name LIKE ? OR ? IS NULL)
                ORDER BY ep.episode_number DESC
            """, (f"%{entity_name}%", entity_name)).fetchall()
        else:
            rows = conn.execute("""
                SELECT c.id, c.claim_text, c.claim_type, c.created_at,
                       ep.episode_number, ep.title
                FROM claims c
                JOIN episodes ep ON ep.id = c.episode_id
                WHERE c.verified = 0
                ORDER BY ep.episode_number DESC LIMIT 20
            """).fetchall()
        return [dict(r) for r in rows]


def resolve_claim(claim_id: int, verified: bool, note: str, resolution_episode_id: int):
    with get_connection() as conn:
        conn.execute("""
            UPDATE claims SET verified = ?, verification_note = ?,
                              verification_episode = ?
            WHERE id = ?
        """, (1 if verified else -1, note, resolution_episode_id, claim_id))


# ── Predictions ───────────────────────────────────────────────────────────────

def save_prediction(episode_id: int, character: str, prediction_text: str,
                    confidence: str = "medium", entity_name: str = None) -> int:
    with get_connection() as conn:
        entity_id = None
        if entity_name:
            row = conn.execute(
                "SELECT id FROM entities WHERE name LIKE ?", (f"%{entity_name}%",)
            ).fetchone()
            if row:
                entity_id = row["id"]

        cur = conn.execute("""
            INSERT INTO predictions
              (episode_id, character, prediction_text, confidence, entity_id)
            VALUES (?, ?, ?, ?, ?)
        """, (episode_id, character, prediction_text, confidence, entity_id))

        # Update character stats
        conn.execute("""
            INSERT OR IGNORE INTO character_stats (character) VALUES (?)
        """, (character,))
        conn.execute("""
            UPDATE character_stats SET
                total_predictions = total_predictions + 1,
                last_updated = CURRENT_TIMESTAMP
            WHERE character = ?
        """, (character,))

        return cur.lastrowid


def get_open_predictions(character: str = None) -> list[dict]:
    """Get unresolved predictions — inject into script for callbacks."""
    with get_connection() as conn:
        if character:
            rows = conn.execute("""
                SELECT p.id, p.character, p.prediction_text, p.confidence,
                       p.created_at, ep.episode_number, ep.title,
                       e.name as entity_name
                FROM predictions p
                JOIN episodes ep ON ep.id = p.episode_id
                LEFT JOIN entities e ON e.id = p.entity_id
                WHERE p.resolved = 0 AND p.character = ?
                ORDER BY ep.episode_number DESC
            """, (character,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT p.id, p.character, p.prediction_text, p.confidence,
                       p.created_at, ep.episode_number, ep.title
                FROM predictions p
                JOIN episodes ep ON ep.id = p.episode_id
                WHERE p.resolved = 0
                ORDER BY ep.episode_number DESC LIMIT 20
            """).fetchall()
        return [dict(r) for r in rows]


def resolve_prediction(prediction_id: int, correct: bool, note: str,
                        resolution_episode_id: int):
    """Resolve a prediction. Updates character accuracy stats."""
    with get_connection() as conn:
        # Get character name
        row = conn.execute(
            "SELECT character FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if not row:
            return

        character = row["character"]
        conn.execute("""
            UPDATE predictions SET
                resolved = 1,
                outcome_correct = ?,
                resolution_date = CURRENT_TIMESTAMP,
                resolution_episode = ?,
                resolution_note = ?
            WHERE id = ?
        """, (1 if correct else 0, resolution_episode_id, note, prediction_id))

        # Update accuracy stats
        if correct:
            conn.execute("""
                UPDATE character_stats SET
                    correct_predictions = correct_predictions + 1,
                    accuracy_pct = ROUND(
                        CAST(correct_predictions + 1 AS REAL) /
                        CAST(total_predictions AS REAL) * 100, 1
                    ),
                    last_updated = CURRENT_TIMESTAMP
                WHERE character = ?
            """, (character,))
        else:
            conn.execute("""
                UPDATE character_stats SET
                    accuracy_pct = ROUND(
                        CAST(correct_predictions AS REAL) /
                        CAST(total_predictions AS REAL) * 100, 1
                    ),
                    last_updated = CURRENT_TIMESTAMP
                WHERE character = ?
            """, (character,))


def get_character_accuracy() -> dict:
    """Get prediction accuracy for all characters — shown on screen."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT character, total_predictions, correct_predictions, accuracy_pct
            FROM character_stats
        """).fetchall()
        return {r["character"]: dict(r) for r in rows}


def get_recently_wrong_predictions(days: int = 30) -> list[dict]:
    """Get predictions that were wrong recently — for acknowledgment episodes."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT p.character, p.prediction_text, ep.episode_number, ep.title,
                   p.resolution_note, p.resolution_date
            FROM predictions p
            JOIN episodes ep ON ep.id = p.episode_id
            WHERE p.resolved = 1
              AND p.outcome_correct = 0
              AND p.resolution_date >= datetime('now', ? || ' days')
            ORDER BY p.resolution_date DESC
        """, (f"-{days}",)).fetchall()
        return [dict(r) for r in rows]


# ── Format History ────────────────────────────────────────────────────────────

def log_episode_format(episode_id: int, format_type: str, scene_key: str,
                        avatar_a: str = None, avatar_b: str = None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO episode_formats_log
              (episode_id, format_type, scene_key, avatar_a, avatar_b)
            VALUES (?, ?, ?, ?, ?)
        """, (episode_id, format_type, scene_key, avatar_a, avatar_b))


def get_recent_formats(n: int = 5) -> list[str]:
    """Get format types of last n episodes — used by content calendar."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT ef.format_type
            FROM episode_formats_log ef
            JOIN episodes ep ON ep.id = ef.episode_id
            ORDER BY ep.episode_number DESC LIMIT ?
        """, (n,)).fetchall()
        return [r["format_type"] for r in rows]


# ── Full Text Search ──────────────────────────────────────────────────────────

def search_past_coverage(query: str, limit: int = 5) -> list[dict]:
    """Search all past episode content for a topic. Used before every script."""
    with get_connection() as conn:
        # Search episode summaries
        rows = conn.execute("""
            SELECT ep.episode_number, ep.title, ep.summary, ep.date,
                   ep.topic_category
            FROM episodes ep
            WHERE ep.title LIKE ?
               OR ep.summary LIKE ?
               OR ep.topic_category LIKE ?
            ORDER BY ep.episode_number DESC LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()

        results = [dict(r) for r in rows]

        # Also search entities
        entity_rows = conn.execute("""
            SELECT DISTINCT ep.episode_number, ep.title, ep.summary,
                   e.name as entity_name, ee.context, ee.sentiment
            FROM entities e
            JOIN episode_entities ee ON ee.entity_id = e.id
            JOIN episodes ep ON ep.id = ee.episode_id
            WHERE e.name LIKE ? OR e.aliases LIKE ?
            ORDER BY ep.episode_number DESC LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit)).fetchall()

        for r in entity_rows:
            if not any(x["episode_number"] == r["episode_number"] for x in results):
                results.append(dict(r))

        return results[:limit]


# ── Context Package for Script Engine ────────────────────────────────────────

def build_script_context(topic: str, entity_names: list[str] = None) -> dict:
    """
    Build the full context package the script engine injects into Ollama.
    Call this before every script generation.
    """
    context = {
        "recent_episodes":    get_recent_episodes(5),
        "past_coverage":      search_past_coverage(topic, 3),
        "open_predictions":   get_open_predictions(),
        "recent_wrong":       get_recently_wrong_predictions(30),
        "character_accuracy": get_character_accuracy(),
        "recent_formats":     get_recent_formats(5),
        "related_entities":   [],
        "unverified_claims":  [],
    }

    if entity_names:
        for name in entity_names[:3]:
            related = find_related_entities(name, 3)
            context["related_entities"].extend(related)
            claims  = get_unverified_claims(name)
            context["unverified_claims"].extend(claims[:2])

    return context


def format_context_for_prompt(ctx: dict) -> str:
    """Convert context dict to natural language for Ollama prompt injection."""
    lines = []

    if ctx["recent_episodes"]:
        lines.append("RECENT EPISODES (for continuity):")
        for ep in ctx["recent_episodes"]:
            lines.append(f"  Ep {ep['episode_number']}: {ep['title']} — {ep.get('summary', '')[:100]}")

    if ctx["past_coverage"]:
        lines.append("\nPAST COVERAGE ON THIS TOPIC:")
        for ep in ctx["past_coverage"]:
            lines.append(f"  Ep {ep['episode_number']}: {ep['title']} ({ep.get('date', '')})")

    if ctx["open_predictions"]:
        lines.append("\nOPEN PREDICTIONS (mention if relevant):")
        for p in ctx["open_predictions"][:3]:
            lines.append(f"  {p['character'].title()} predicted (Ep {p['episode_number']}): {p['prediction_text']}")

    if ctx["recent_wrong"]:
        lines.append("\nRECENT WRONG PREDICTIONS (can reference for humility):")
        for p in ctx["recent_wrong"][:2]:
            lines.append(f"  {p['character'].title()} was wrong: {p['prediction_text']}")

    if ctx["character_accuracy"]:
        lines.append("\nCURRENT PREDICTION ACCURACY:")
        for char, stats in ctx["character_accuracy"].items():
            lines.append(f"  {char.title()}: {stats['accuracy_pct']}% ({stats['correct_predictions']}/{stats['total_predictions']})")

    if ctx["related_entities"]:
        lines.append("\nRELATED TOPICS WE COVERED (use for comparisons):")
        for e in ctx["related_entities"][:3]:
            lines.append(f"  {e['name']} ({e['entity_type']}) — {e.get('correlation_type', 'related')}, Ep {e['episode_number']}")

    if ctx["unverified_claims"]:
        lines.append("\nUNVERIFIED CLAIMS (follow up if possible):")
        for c in ctx["unverified_claims"][:2]:
            lines.append(f"  Ep {c['episode_number']}: \"{c['claim_text']}\" — not yet verified")

    recent_formats = ctx.get("recent_formats", [])
    if recent_formats:
        lines.append(f"\nRECENT FORMATS: {', '.join(recent_formats)} — vary if pattern detected")

    return "\n".join(lines)


# ── Init on import ────────────────────────────────────────────────────────────
init_db()
