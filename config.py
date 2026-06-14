"""
Avatar Studio — Central Configuration
All paths, model configs, scene definitions, avatar personas.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).parent.resolve()

# ── Directories ───────────────────────────────────────────────────────────────
MODELS_DIR      = ROOT / "models"
OUTPUTS_DIR     = ROOT / "outputs"
ASSETS_DIR      = ROOT / "assets"
LOGS_DIR        = ROOT / "logs"

AVATARS_DIR     = MODELS_DIR / "avatars"
VOICES_DIR      = MODELS_DIR / "voices"
BACKGROUNDS_DIR = MODELS_DIR / "backgrounds"
MUSIC_DIR       = MODELS_DIR / "music"
WEIGHTS_DIR     = MODELS_DIR / "weights"

ECHOMIMIC_DIR   = MODELS_DIR / "EchoMimicV2"
LATENTSYNC_DIR  = MODELS_DIR / "LatentSync"
MUSECPOSE_DIR   = MODELS_DIR / "MusePose"
CYBERHOST_DIR   = MODELS_DIR / "CyberHost"
REALESRGAN_DIR  = MODELS_DIR / "Real-ESRGAN"

GRAPHICS_DIR    = ASSETS_DIR / "graphics"
PROPS_DIR       = ASSETS_DIR / "props"
SFX_DIR         = ASSETS_DIR / "sfx"
LUTS_DIR        = ASSETS_DIR / "luts"

for d in [MODELS_DIR, OUTPUTS_DIR, ASSETS_DIR, LOGS_DIR,
          AVATARS_DIR, VOICES_DIR, BACKGROUNDS_DIR, MUSIC_DIR, WEIGHTS_DIR,
          GRAPHICS_DIR, PROPS_DIR, SFX_DIR, LUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32
VRAM_GB  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if DEVICE == "cuda" else 0

# ── Model IDs ─────────────────────────────────────────────────────────────────
FLUX_MODEL       = "black-forest-labs/FLUX.1-schnell"   # Apache 2.0 — commercial safe
FLUX_LORA        = "prithivMLmods/Desi-Espresso-Flux"   # South Indian faces LoRA
FLUX_STEPS       = 4                                    # schnell needs only 4 steps

WHISPER_MODEL    = "large-v3"
OLLAMA_MODEL     = "gemma3:4b"
OLLAMA_URL       = "http://localhost:11434"
OLLAMA_TIMEOUT   = 120

INDIC_TTS_MODEL  = "ai4bharat/IndicF5"
REALESRGAN_MODEL = "RealESRGAN_x4plus"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FPS         = 25
OUTPUT_RESOLUTION  = (1920, 1080)   # 16:9
VERTICAL_RES       = (1080, 1920)   # 9:16 Reels/Shorts
SQUARE_RES         = (1080, 1080)   # 1:1 Instagram
SEGMENT_MAX_SEC    = 300            # chunk long scripts into 5-min segments

# ── Scene Library ─────────────────────────────────────────────────────────────
# Each scene: video loop file, lighting tone, reverb type, ambient sound file
SCENES = {
    "professional/office":       {"file": "office_loop.mp4",        "lighting": "warm",    "reverb": "small_room",  "ambient": "office_hum.wav",     "label": "Office"},
    "professional/news_desk":    {"file": "news_desk_loop.mp4",     "lighting": "studio",  "reverb": "dead",        "ambient": "silence.wav",        "label": "News Desk"},
    "professional/seminar_hall": {"file": "seminar_hall_loop.mp4",  "lighting": "warm",    "reverb": "large_hall",  "ambient": "crowd_murmur.wav",   "label": "Seminar Hall"},
    "professional/stage_dias":   {"file": "stage_dias_loop.mp4",    "lighting": "spot",    "reverb": "large_hall",  "ambient": "crowd_murmur.wav",   "label": "Stage Dias"},
    "professional/conference":   {"file": "conference_loop.mp4",    "lighting": "neutral", "reverb": "medium_room", "ambient": "office_hum.wav",     "label": "Conference Room"},
    "nature/glacier":            {"file": "glacier_loop.mp4",       "lighting": "cool",    "reverb": "outdoors",    "ambient": "wind_cold.wav",      "label": "Glacier"},
    "nature/beach":              {"file": "beach_loop.mp4",         "lighting": "warm",    "reverb": "outdoors",    "ambient": "waves_birds.wav",    "label": "Beach"},
    "nature/forest":             {"file": "forest_loop.mp4",        "lighting": "green",   "reverb": "outdoors",    "ambient": "forest_birds.wav",   "label": "Forest"},
    "nature/mountain":           {"file": "mountain_loop.mp4",      "lighting": "cool",    "reverb": "outdoors",    "ambient": "wind_soft.wav",      "label": "Mountain"},
    "casual/kitchen":            {"file": "kitchen_loop.mp4",       "lighting": "warm",    "reverb": "small_room",  "ambient": "kitchen_ambient.wav","label": "Kitchen"},
    "casual/living_room":        {"file": "living_room_loop.mp4",   "lighting": "warm",    "reverb": "small_room",  "ambient": "silence.wav",        "label": "Living Room"},
    "casual/bedroom":            {"file": "bedroom_loop.mp4",       "lighting": "warm",    "reverb": "small_room",  "ambient": "silence.wav",        "label": "Bedroom"},
    "casual/cafe":               {"file": "cafe_loop.mp4",          "lighting": "warm",    "reverb": "medium_room", "ambient": "cafe_ambient.wav",   "label": "Cafe"},
    "casual/rooftop":            {"file": "rooftop_loop.mp4",       "lighting": "warm",    "reverb": "outdoors",    "ambient": "city_ambient.wav",   "label": "Rooftop"},
    "landmark/red_fort":         {"file": "red_fort_loop.mp4",      "lighting": "warm",    "reverb": "outdoors",    "ambient": "city_ambient.wav",   "label": "Red Fort"},
    "landmark/parliament":       {"file": "parliament_loop.mp4",    "lighting": "neutral", "reverb": "outdoors",    "ambient": "city_ambient.wav",   "label": "Parliament"},
    "landmark/tech_park":        {"file": "tech_park_loop.mp4",     "lighting": "neutral", "reverb": "outdoors",    "ambient": "city_ambient.wav",   "label": "Tech Park"},
    "landmark/market":           {"file": "market_loop.mp4",        "lighting": "warm",    "reverb": "outdoors",    "ambient": "market_crowd.wav",   "label": "Market"},
    "brand/google_stage":        {"file": "google_stage_loop.mp4",  "lighting": "studio",  "reverb": "large_hall",  "ambient": "crowd_murmur.wav",   "label": "Google Stage"},
    "brand/apple_stage":         {"file": "apple_stage_loop.mp4",   "lighting": "studio",  "reverb": "large_hall",  "ambient": "crowd_murmur.wav",   "label": "Apple Stage"},
    "brand/samsung_stage":       {"file": "samsung_stage_loop.mp4", "lighting": "studio",  "reverb": "large_hall",  "ambient": "crowd_murmur.wav",   "label": "Samsung Stage"},
    "brand/ted_stage":           {"file": "ted_stage_loop.mp4",     "lighting": "spot",    "reverb": "large_hall",  "ambient": "crowd_murmur.wav",   "label": "TED Stage"},
    "abstract/dark_studio":      {"file": "dark_studio_loop.mp4",   "lighting": "studio",  "reverb": "dead",        "ambient": "silence.wav",        "label": "Dark Studio"},
    "abstract/gradient_blue":    {"file": "gradient_blue_loop.mp4", "lighting": "cool",    "reverb": "dead",        "ambient": "silence.wav",        "label": "Gradient Blue"},
}

# Brand keyword → auto scene assignment
BRAND_SCENE_MAP = {
    "google": "brand/google_stage",   "apple": "brand/apple_stage",
    "samsung": "brand/samsung_stage", "microsoft": "brand/tech_park",
    "amazon": "brand/tech_park",      "meta": "brand/tech_park",
    "ted": "brand/ted_stage",         "parliament": "landmark/parliament",
    "red fort": "landmark/red_fort",  "lal qila": "landmark/red_fort",
    "beach": "nature/beach",          "glacier": "nature/glacier",
    "market": "landmark/market",      "kitchen": "casual/kitchen",
}

# ── Avatar Personas ───────────────────────────────────────────────────────────
AVATARS = {
    "priya_telugu_f": {
        "name": "Priya", "language": "te", "gender": "female",
        "voice_profile": "te_female_professional",
        "poses": ["standing", "sitting_desk", "half_body"],
        "attires": ["professional", "traditional_saree", "casual"],
        "flux_prompt_base": "photorealistic south indian telugu woman, 30 years old, warm brown skin, dark eyes, professional appearance, sharp features, studio lighting, 8k",
    },
    "arjun_telugu_m": {
        "name": "Arjun", "language": "te", "gender": "male",
        "voice_profile": "te_male_professional",
        "poses": ["standing", "sitting_desk", "half_body"],
        "attires": ["suit", "kurta", "casual"],
        "flux_prompt_base": "photorealistic south indian telugu man, 32 years old, warm brown skin, dark hair, confident look, professional, studio lighting, 8k",
    },
    "kavya_kannada_f": {
        "name": "Kavya", "language": "kn", "gender": "female",
        "voice_profile": "kn_female_professional",
        "poses": ["standing", "half_body"],
        "attires": ["professional", "traditional_saree"],
        "flux_prompt_base": "photorealistic south indian kannada woman, 28 years old, warm brown skin, long dark hair, elegant, professional, studio lighting, 8k",
    },
    "ravi_tamil_m": {
        "name": "Ravi", "language": "ta", "gender": "male",
        "voice_profile": "ta_male_professional",
        "poses": ["standing", "sitting_desk"],
        "attires": ["suit", "casual"],
        "flux_prompt_base": "photorealistic south indian tamil man, 35 years old, dark brown skin, confident expression, professional, studio lighting, 8k",
    },
    "ananya_hindi_f": {
        "name": "Ananya", "language": "hi", "gender": "female",
        "voice_profile": "hi_female_professional",
        "poses": ["standing", "sitting_desk", "half_body"],
        "attires": ["professional", "traditional_saree", "casual"],
        "flux_prompt_base": "photorealistic north indian hindi woman, 29 years old, wheatish skin, expressive eyes, professional, studio lighting, 8k",
    },
}

# Pose prompt additions
POSE_PROMPTS = {
    "standing":     "full body shot, standing upright, arms naturally at sides, visible from head to toe",
    "half_body":    "half body shot, visible from waist up, arms and hands visible, natural posture",
    "sitting_desk": "half body shot, sitting at desk, upper body and arms visible, professional desk setup in front",
}

# Attire prompt additions
ATTIRE_PROMPTS = {
    "professional":       "wearing formal business attire, blazer, professional shirt",
    "suit":               "wearing dark business suit, white shirt, tie",
    "traditional_saree":  "wearing elegant silk saree, traditional south indian style",
    "kurta":              "wearing formal kurta, traditional indian male attire",
    "casual":             "wearing smart casual clothes, relaxed professional look",
}

# ── Voice Profiles ────────────────────────────────────────────────────────────
VOICE_PROFILES = {
    "te_female_professional": {"engine": "indic", "lang": "te", "gender": "female", "speed": 1.0, "pitch": 1.0},
    "te_male_professional":   {"engine": "indic", "lang": "te", "gender": "male",   "speed": 1.0, "pitch": 0.9},
    "kn_female_professional": {"engine": "indic", "lang": "kn", "gender": "female", "speed": 1.0, "pitch": 1.0},
    "ta_male_professional":   {"engine": "indic", "lang": "ta", "gender": "male",   "speed": 1.0, "pitch": 0.9},
    "hi_female_professional": {"engine": "indic", "lang": "hi", "gender": "female", "speed": 1.0, "pitch": 1.0},
    "en_male_professional":   {"engine": "chatterbox", "lang": "en", "gender": "male",   "speed": 1.0, "pitch": 0.9},
    "en_female_professional": {"engine": "chatterbox", "lang": "en", "gender": "female", "speed": 1.0, "pitch": 1.0},
}

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGES = {
    "te": "Telugu", "kn": "Kannada", "ta": "Tamil", "hi": "Hindi",
    "en": "English", "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali",
}

# ── Video Formats ─────────────────────────────────────────────────────────────
VIDEO_FORMATS = {
    "monologue":   {"avatars": 1, "layout": "single",      "max_sec": 600,  "desc": "Single presenter"},
    "debate":      {"avatars": 2, "layout": "split_screen","max_sec": 900,  "desc": "Two avatars debate"},
    "news_anchor": {"avatars": 1, "layout": "desk",        "max_sec": 300,  "desc": "News anchor style"},
    "seminar":     {"avatars": 1, "layout": "full_body",   "max_sec": 1800, "desc": "Full body presenter"},
    "short":       {"avatars": 1, "layout": "vertical",    "max_sec": 90,   "desc": "Reels / Shorts"},
}
