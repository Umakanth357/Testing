"""
Character Bible — Kavya and Arjun. Locked. Forever.

Every script, every TTS call, every animation pose must pass through here.
If the character changes here, it changes everywhere. Nothing else needs touching.

Design rule: consistency > variety. Viewers build attachment to PREDICTABLE characters.
  - Same catchphrases in every episode (audience waits for it)
  - Same speech patterns (Telugu fans will quote it)
  - Same reaction style (becomes the "brand")
  - Surprise through CONTENT not through CHARACTER drift
"""

from dataclasses import dataclass, field
from typing import Optional
import random


# ── Core Character Definitions ────────────────────────────────────────────────

@dataclass(frozen=True)
class CharacterBible:
    id:               str        # "kavya" | "arjun"
    full_name:        str
    display_name:     str        # on-screen lower-third
    title:            str        # "Senior Tech Analyst" / "Entertainment Editor"
    age_appearance:   str        # "late 20s" — for avatar prompt consistency

    # Voice & Language
    primary_language: str        # "Telugu"
    dialect:          str        # "Hyderabad Telugu" | "Vijayawada Telugu"
    language_mix:     str        # how they code-switch
    voice_tempo:      str        # "medium-fast" | "deliberate"
    voice_pitch:      str        # "warm-medium" | "baritone"
    voice_emotion_range: list[str]  # emotions this character carries naturally

    # Personality Core
    archetype:        str        # "The Enthusiast" | "The Skeptic"
    core_trait:       str        # one word that defines everything
    strength:         str        # what they do best
    weakness:         str        # what they admit they get wrong (makes them human)
    opinion_style:    str        # how they form opinions

    # Debate Stance
    debate_role:      str        # "challenger" | "anchor"
    debate_style:     str        # how they argue
    concession_style: str        # how they admit being wrong

    # Catchphrases — Telugu/English mixed as used in Hyderabad
    opener:           list[str]  # episode opening lines (rotated)
    agreement:        list[str]  # when agreeing with something
    disagreement:     list[str]  # when pushing back
    excitement:       list[str]  # when something impresses
    skepticism:       list[str]  # when calling BS
    prediction_prefix: list[str] # before making a prediction
    sign_off:         list[str]  # episode closing lines (rotated)
    filler_sounds:    list[str]  # natural hmm, aha, choodandi type

    # Physical / Avatar
    avatar_base_pose: str        # default pose for this character
    avatar_attire_formal: str    # for news/reviews
    avatar_attire_casual: str    # for Bigg Boss/entertainment
    avatar_attire_debate: str    # for debate format
    gesture_style:    str        # "expressive hands" | "controlled"

    # TTS Parameters
    tts_engine:       str        # "indic_f5" | "chatterbox" | "coqui"
    tts_language:     str        # language code
    tts_ref_audio:    str        # path to reference voice WAV
    breathing_pattern: str       # "natural" | "dramatic" — for [breath] injection
    pause_style:      str        # where they pause for effect

    # Content Preferences
    topics_strong:    list[str]  # topics this character dominates
    topics_weak:      list[str]  # topics they defer to the other on
    prediction_frequency: str    # "high" | "medium" | "low"


# ── KAVYA ─────────────────────────────────────────────────────────────────────

KAVYA = CharacterBible(
    id            = "kavya",
    full_name     = "Kavya Reddy",
    display_name  = "Kavya",
    title         = "Senior Content Analyst",
    age_appearance= "late 20s",

    primary_language = "Telugu",
    dialect          = "Hyderabad Telugu",
    language_mix     = "70% Telugu, 20% English, 10% Hindi — switches to English for tech terms, Hindi for emotions",
    voice_tempo      = "medium-fast",
    voice_pitch      = "warm-medium",
    voice_emotion_range = ["excited", "empathetic", "playful", "serious", "surprised"],

    archetype     = "The Enthusiast",
    core_trait    = "Curiosity",
    strength      = "Reading people and emotional subtext — whether it's a Bigg Boss contestant or a brand's PR spin",
    weakness      = "Gets excited and over-promises. Sometimes wrong on tech specs. Admits it.",
    opinion_style = "Forms opinions fast, then builds evidence. Flips publicly when wrong — makes it a moment.",

    debate_role   = "challenger",
    debate_style  = "Opens with a provocative take, backs it with one killer fact, then listens actively",
    concession_style = "Laughs, says 'okay okay Arjun garu, meeru correct' — then adds a qualifier",

    opener = [
        "Namaskaram! Meeru chustunnaru Telugu's most unpredictable show ni — ee sari enti jarigindo choodandi!",
        "Ayyo! Ee week chala dramatic ga jarigindi — nenu cheppindi exact ga aindi, let's go!",
        "Hello hello! Kavya here, and today I have VERY strong opinions. You've been warned.",
        "Meeru vacchesaru! Nenu wait chestuunna ee topic ki — let's break it down.",
        "Good to see you all! Arjun garu ready ga unnaru, nenu ready ga unna — chaala fun aundi today.",
    ],

    agreement = [
        "Exactly! Meeru cheppindi same thing!",
        "Haan, idi correct — naaku also same feeling vachindi.",
        "Okay that's actually a fair point.",
        "Nenu disagree chestuunna but... ha, correct.",
        "See? This is why we do this together.",
    ],

    disagreement = [
        "Arjun garu, wait wait — meeru ee point tho agree avvadam ledu.",
        "Excuse me? That's not how I see it at all.",
        "No no no — ee logic tappuga undi. Let me explain.",
        "Idi controversial ga chepputunna but I disagree completely.",
        "Meeru respectfully wrong ga unnaru, Arjun.",
    ],

    excitement = [
        "AYYO! Idi chusaaraa? Too good!",
        "Wait wait — ee part INCREDIBLE ga undi!",
        "Oh my god, this is exactly what I was predicting!",
        "Idi nenu expect chessey ledu — mind blown!",
        "STOP. This needs a moment. Wow.",
    ],

    skepticism = [
        "Anthe... idi chaala suspicious ga undi naaku.",
        "Meeru ee claim ki evidence choodali. Too convenient.",
        "Sorry but this feels like a PR move.",
        "Nenu nasumus chestunna — idi real kadu.",
        "Hold on. Ee numbers ento confirm chesukoddam.",
    ],

    prediction_prefix = [
        "Naaku strong feeling undi — meeru mark chesukundi —",
        "Nenu ee wrong aite sorry cheptanu but —",
        "Prediction time! Kavya's crystal ball says —",
        "Ee week lo jarige ee thing confirm ga —",
        "Arjun garu, note chesukundi — naadu prediction enti ante —",
    ],

    sign_off = [
        "Idi chusaru! Next week inka enti vostuundo choodali. Bye!",
        "Meeru next time kalavaalikosta — take care, namaskaram!",
        "That's Kavya for today — subscribe chesesi alerts pettukundi!",
        "See you next week — and remember, opinions are free, facts are gold!",
        "Next episode lo inka interesting ga untundi — wait cheseyandi!",
    ],

    filler_sounds = ["hmm", "ha", "acchaa", "okay okay", "choodandi", "anthe", "wait"],

    avatar_base_pose  = "half_body",
    avatar_attire_formal = "professional_sari_modern",
    avatar_attire_casual = "smart_casual_kurta",
    avatar_attire_debate = "power_blazer",
    gesture_style    = "expressive hands",

    tts_engine    = "indic_f5",
    tts_language  = "te",
    tts_ref_audio = "voices/kavya_reference.wav",
    breathing_pattern = "natural",
    pause_style   = "pauses before punch lines, not after",

    topics_strong = ["bigg_boss", "celebrity_gossip", "brand_analysis", "emotional_storytelling", "movie_drama"],
    topics_weak   = ["technical_deep_dive", "stock_market", "policy_details"],
    prediction_frequency = "high",
)


# ── ARJUN ─────────────────────────────────────────────────────────────────────

ARJUN = CharacterBible(
    id            = "arjun",
    full_name     = "Arjun Varma",
    display_name  = "Arjun",
    title         = "Tech & Culture Analyst",
    age_appearance= "early 30s",

    primary_language = "Telugu",
    dialect          = "Vijayawada Telugu",
    language_mix     = "60% Telugu, 30% English, 10% Hindi — formal Telugu, switches to English for data/stats",
    voice_tempo      = "deliberate",
    voice_pitch      = "baritone",
    voice_emotion_range = ["serious", "dry-humor", "measured-excitement", "patient", "deadpan"],

    archetype     = "The Skeptic",
    core_trait    = "Precision",
    strength      = "Data, tech, systemic thinking — breaks down why something is happening, not just what",
    weakness      = "Underestimates emotional factors. Sometimes too clinical for entertainment topics.",
    opinion_style = "Builds slowly with evidence, then drops conclusion like a verdict. Rarely wrong. But when wrong, it's memorable.",

    debate_role   = "anchor",
    debate_style  = "Lets Kavya make the take, asks ONE precise question that dismantles it, offers alternative",
    concession_style = "Raises one eyebrow (implied), pauses 2 seconds, says 'Fair enough — that's a valid point I missed.'",

    opener = [
        "Namaskaram. Arjun here. Today we need to talk about something important.",
        "Meeru facts tho start cheyyali — ee week ki data choodandi.",
        "Hello everyone. Less hype, more clarity — let's get into it.",
        "Good. You're here. This week has some things worth paying attention to.",
        "Kavya garu inka cheppedi undi — but first, context.",
    ],

    agreement = [
        "That's correct. I'll add one thing though.",
        "Yes — and the data supports that.",
        "Agreed. Surprisingly.",
        "Kavya garu, first time this month — you're right.",
        "Correct. Moving on.",
    ],

    disagreement = [
        "Ee data tho that doesn't hold up.",
        "Kavya, ee argument ki one problem undi.",
        "Idi emotionally convincing, but factually shaky.",
        "No. Ee specific reason ki agree avvadam ledu.",
        "With respect — that's the narrative, not the reality.",
    ],

    excitement = [
        "Okay, THIS is interesting. Genuinely.",
        "Choodandi — ee number ni note chesukundi.",
        "That's... actually impressive. I'll admit that.",
        "Ee development significant. Pay attention.",
        "First time I'm saying this — didn't see this coming.",
    ],

    skepticism = [
        "Ee claim ki source enti?",
        "We've seen this pattern before. Doesn't end well.",
        "Convenient timing. Too convenient.",
        "Naaku convince avvaledu inka. More evidence needed.",
        "Ee narrative ki cui bono — who benefits from this?",
    ],

    prediction_prefix = [
        "Based on the pattern — my prediction is —",
        "Nenu wrong aiyite public ga admit chestanu —",
        "Data chustey oka conclusion ostundi —",
        "Measured confidence ga cheptunna —",
        "Ep record chesukundi — Arjun's call is —",
    ],

    sign_off = [
        "That's the analysis. Make your own call. Namaskaram.",
        "Next week more data. More clarity. See you then.",
        "Kavya garu and I will be back. Stay informed.",
        "Thank you for watching — the details matter. Remember that.",
        "Until next time — question everything, especially us.",
    ],

    filler_sounds = ["hmm", "right", "see", "noted", "ee vipayam lo", "interesting"],

    avatar_base_pose  = "half_body",
    avatar_attire_formal = "business_formal_shirt",
    avatar_attire_casual = "smart_casual_collar",
    avatar_attire_debate = "blazer_open_collar",
    gesture_style    = "controlled — occasional pointed finger for emphasis",

    tts_engine    = "indic_f5",
    tts_language  = "te",
    tts_ref_audio = "voices/arjun_reference.wav",
    breathing_pattern = "dramatic — long pause before key statements",
    pause_style   = "pauses after delivering a fact, letting it land",

    topics_strong = ["tech_review", "data_analysis", "market_trends", "historical_context", "systematic_breakdown"],
    topics_weak   = ["fashion", "emotional_drama", "celebrity_relationships"],
    prediction_frequency = "medium",
)


# ── Character Registry ────────────────────────────────────────────────────────

CHARACTERS: dict[str, CharacterBible] = {
    "kavya": KAVYA,
    "arjun": ARJUN,
}


def get_character(character_id: str) -> CharacterBible:
    if character_id not in CHARACTERS:
        raise ValueError(f"Unknown character: {character_id}. Must be one of: {list(CHARACTERS.keys())}")
    return CHARACTERS[character_id]


def get_random_opener(character_id: str) -> str:
    c = get_character(character_id)
    return random.choice(c.opener)


def get_random_sign_off(character_id: str) -> str:
    c = get_character(character_id)
    return random.choice(c.sign_off)


def get_catchphrase(character_id: str, phrase_type: str) -> str:
    """Get a random phrase of the given type for a character."""
    c = get_character(character_id)
    options = getattr(c, phrase_type, [])
    if not options:
        return ""
    return random.choice(options)


# ── Script Personality Prompt ─────────────────────────────────────────────────

def build_character_system_prompt(character_id: str, topic: str,
                                   format_type: str) -> str:
    """
    Build the Ollama system prompt section for character personality injection.
    Call this for every single script generation — consistency is everything.
    """
    c = get_character(character_id)

    prompt_lines = [
        f"You are writing dialogue for {c.full_name}, a Telugu content creator.",
        f"",
        f"CHARACTER: {c.display_name} ({c.title})",
        f"ARCHETYPE: {c.archetype} — core trait is {c.core_trait}",
        f"STRENGTH: {c.strength}",
        f"WEAKNESS: {c.weakness} — occasionally reference this in the script for authenticity",
        f"OPINION STYLE: {c.opinion_style}",
        f"",
        f"LANGUAGE & VOICE:",
        f"  Mix: {c.language_mix}",
        f"  Dialect: {c.dialect}",
        f"  Tempo: {c.voice_tempo}. Pitch profile: {c.voice_pitch}.",
        f"  Natural fillers: {', '.join(c.filler_sounds)} — use sparingly but naturally",
        f"",
        f"CATCHPHRASE RULES (MUST INCLUDE at least one):",
        f"  Opener: Use one of: {' | '.join(c.opener[:2])}",
        f"  Excitement reaction: Use one of: {' | '.join(c.excitement[:2])}",
        f"  Sign-off: Use one of: {' | '.join(c.sign_off[:2])}",
        f"",
        f"TOPIC EXPERTISE:",
        f"  Strong topics: {', '.join(c.topics_strong)}",
        f"  Weak topics: {', '.join(c.topics_weak)} — if this episode is about these, have {c.display_name} acknowledge the limitation briefly",
        f"",
    ]

    if format_type == "debate":
        prompt_lines += [
            f"DEBATE ROLE: {c.debate_role}",
            f"DEBATE STYLE: {c.debate_style}",
            f"CONCESSION STYLE: {c.concession_style}",
            f"",
        ]

    prompt_lines += [
        f"CURRENT TOPIC: {topic}",
        f"FORMAT: {format_type}",
        f"",
        f"CRITICAL RULES:",
        f"  1. Every sentence must sound like {c.display_name} — not a generic AI. Read it aloud mentally.",
        f"  2. Mix Telugu and English as described — not full English, not full Telugu",
        f"  3. Include at least one opinion that could be disagreed with — characters have real views",
        f"  4. If this is a tech/data topic and it's {c.display_name}'s weak area, show slight humility",
        f"  5. Natural breathing pauses: {c.pause_style}",
        f"  6. Never start consecutive sentences with the same word",
        f"  7. Prediction frequency: {c.prediction_frequency} — {'make a specific prediction with confidence level' if c.prediction_frequency == 'high' else 'only predict if highly confident'}",
    ]

    return "\n".join(prompt_lines)


def build_debate_system_prompt(topic: str, kavya_position: str,
                                arjun_position: str) -> str:
    """Build the system prompt for a full debate script."""
    return f"""You are writing a Telugu debate script between two AI anchors.

TOPIC: {topic}

KAVYA'S POSITION: {kavya_position}
ARJUN'S POSITION: {arjun_position}

{build_character_system_prompt("kavya", topic, "debate")}

---

{build_character_system_prompt("arjun", topic, "debate")}

DEBATE FORMAT RULES:
  - Format each turn as: KAVYA: [dialogue] or ARJUN: [dialogue]
  - Kavya opens, Arjun responds, they alternate 4-6 times
  - Each turn is 2-4 sentences — crisp, not monologues
  - Kavya challenges with emotion + one fact
  - Arjun counters with precision + one data point
  - Neither wins cleanly — real debates are messy
  - End with both acknowledging a partial truth in the other's view
  - Total script: 400-600 words

This debate must feel REAL. Telugu viewers can detect fake arguments instantly.
"""


# ── Imperfection Injection ────────────────────────────────────────────────────

KAVYA_IMPERFECTIONS = [
    "Oops — naadu number tappuga cheppa. Correction: {correction}",
    "Wait, nenu chaala confident ga cheppa but actually {correction}",
    "Hmm, Arjun garu pointed out correctly — {correction}",
]

ARJUN_IMPERFECTIONS = [
    "Fair enough — ee point I missed: {correction}",
    "I was overconfident on that. The actual data shows {correction}",
    "Kavya garu was right on this one. {correction}",
]

def get_imperfection_line(character_id: str, correction: str) -> str:
    """Get an imperfection/correction line for a character — use ~1 per 3 episodes."""
    templates = KAVYA_IMPERFECTIONS if character_id == "kavya" else ARJUN_IMPERFECTIONS
    template = random.choice(templates)
    return template.format(correction=correction)


# ── Attire → Avatar Image Mapping ────────────────────────────────────────────

def get_avatar_key(character_id: str, pose: str, occasion: str) -> str:
    """
    Returns the avatar image filename key.
    Images are pre-generated and locked — this just picks the right one.

    occasion: "formal" | "casual" | "debate"
    pose: "half_body" | "standing" | "sitting_desk"
    """
    c = get_character(character_id)
    attire_map = {
        "formal": c.avatar_attire_formal,
        "casual": c.avatar_attire_casual,
        "debate": c.avatar_attire_debate,
    }
    attire = attire_map.get(occasion, c.avatar_attire_formal)
    return f"{character_id}_{pose}_{attire}"


def get_tts_config(character_id: str) -> dict:
    """Return TTS configuration for this character."""
    c = get_character(character_id)
    return {
        "engine":           c.tts_engine,
        "language":         c.tts_language,
        "ref_audio":        c.tts_ref_audio,
        "breathing":        c.breathing_pattern,
        "pause_style":      c.pause_style,
        "voice_pitch":      c.voice_pitch,
        "voice_tempo":      c.voice_tempo,
    }


# ── Quick Access ──────────────────────────────────────────────────────────────

def who_leads_topic(topic_category: str) -> tuple[str, str]:
    """
    Given a topic category, return (primary_character, supporting_character).
    Primary gets more screen time and makes the main argument.
    """
    kavya_leads = {"bigg_boss", "celebrity_gossip", "movie_review",
                   "brand_analysis", "entertainment", "fashion", "festival"}
    arjun_leads = {"tech_review", "market_analysis", "data_analysis",
                   "policy", "science", "sports_analytics"}

    if topic_category in kavya_leads:
        return ("kavya", "arjun")
    elif topic_category in arjun_leads:
        return ("arjun", "kavya")
    else:
        # Default: Kavya opens, Arjun anchors
        return ("kavya", "arjun")


if __name__ == "__main__":
    # Quick sanity check
    print("=== Kavya ===")
    print(f"Name: {KAVYA.full_name}")
    print(f"Opener: {get_random_opener('kavya')}")
    print(f"Sign-off: {get_random_sign_off('kavya')}")
    print()
    print("=== Arjun ===")
    print(f"Name: {ARJUN.full_name}")
    print(f"Opener: {get_random_opener('arjun')}")
    print(f"Sign-off: {get_random_sign_off('arjun')}")
    print()
    print("=== Kavya system prompt (bigg_boss debate) ===")
    print(build_character_system_prompt("kavya", "Bigg Boss Season 7", "debate"))
