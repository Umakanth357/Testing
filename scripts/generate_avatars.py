#!/usr/bin/env python3
"""
Avatar Studio — Avatar Image Generator
Generates Navya Reddy + Arjun Varma avatar images using Stable Diffusion XL.

Why SDXL instead of FLUX.1-schnell:
  - FLUX full pipeline = ~33GB download, T5-XXL alone = 9.5GB RAM → OOM on g4dn.2xlarge (32GB RAM)
  - SDXL = ~6.5GB VRAM, 7GB download, runs cleanly on T4 16GB
  - SDXL is not gated (no HF terms required)
  - Portrait quality is equivalent for this use case with good prompts

Run once after setup. Images saved to models/avatars/ and never regenerated unless deleted.
Usage: python scripts/generate_avatars.py
"""
import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
AVATARS_DIR = ROOT / "models" / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)

# SDXL base — Apache 2.0, not gated, 6.5GB VRAM, runs on T4
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'}")
if DEVICE == "cuda":
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

# ── Avatar definitions ────────────────────────────────────────────────────────
# Navya Reddy — 22yo, attractive Hyderabad Telugu girl, sweet+confident+strong
# Arjun Varma — 32yo, Vijayawada Telugu professional anchor

NAVYA_BASE = (
    "portrait photo of a beautiful south indian telugu woman, 22 years old, "
    "from hyderabad, warm brown skin, long dark black hair, sharp expressive eyes, "
    "high cheekbones, elegant facial structure, confident warm smile, "
    "photorealistic, professional studio portrait, 85mm lens, "
    "sharp focus, soft studio lighting, 8k uhd, not a real person, unique face"
)

ARJUN_BASE = (
    "portrait photo of a south indian telugu man, 32 years old, "
    "from vijayawada, warm brown skin, short dark hair, strong jawline, "
    "intelligent eyes, composed confident expression, "
    "photorealistic, professional studio portrait, 85mm lens, "
    "sharp focus, soft studio lighting, 8k uhd, not a real person, unique face"
)

NEGATIVE = (
    "blurry, low quality, watermark, text, logo, cartoon, anime, illustration, "
    "painting, sketch, ugly, deformed, bad anatomy, extra fingers, mutation, "
    "poorly drawn face, nsfw, celebrity, famous person, plastic, wax, mannequin, "
    "overexposed, underexposed, grainy, noise"
)

AVATARS = {
    "navya": {
        "base": NAVYA_BASE,
        "variants": [
            {
                "key": "navya_professional",
                "label": "Navya — Professional Blazer",
                "suffix": "wearing a sharp navy blue formal blazer over white shirt, waist up half body shot, professional attire",
                "steps": 40, "guidance": 7.5,
            },
            {
                "key": "navya_traditional",
                "label": "Navya — Traditional Saree",
                "suffix": "wearing an elegant deep red and gold silk saree, traditional south indian jewelry, half body shot",
                "steps": 40, "guidance": 7.5,
            },
            {
                "key": "navya_casual",
                "label": "Navya — Smart Casual",
                "suffix": "wearing smart casual clothes in pastel tones, friendly approachable expression, half body shot",
                "steps": 40, "guidance": 7.5,
            },
        ],
    },
    "arjun": {
        "base": ARJUN_BASE,
        "variants": [
            {
                "key": "arjun_suit",
                "label": "Arjun — Dark Suit",
                "suffix": "wearing a dark charcoal business suit with white shirt, professional anchor look, waist up half body shot",
                "steps": 40, "guidance": 7.5,
            },
            {
                "key": "arjun_kurta",
                "label": "Arjun — Formal Kurta",
                "suffix": "wearing a crisp white formal kurta, traditional indian male attire, half body shot",
                "steps": 40, "guidance": 7.5,
            },
            {
                "key": "arjun_casual",
                "label": "Arjun — Smart Casual",
                "suffix": "wearing smart casual polo shirt in navy, relaxed professional look, half body shot",
                "steps": 40, "guidance": 7.5,
            },
        ],
    },
}


# ── Load pipeline ─────────────────────────────────────────────────────────────

def load_pipeline():
    print(f"\nLoading SDXL pipeline from {SDXL_MODEL}...")
    print("  (First run downloads ~7GB — ~2-3 min on EC2)")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        token=HF_TOKEN,
    )

    # SDXL fits in T4 16GB VRAM without CPU offload, but enable for safety
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  xformers memory efficient attention: ON")
    except Exception:
        print("  xformers not available — using standard attention")

    print("  SDXL pipeline loaded OK")
    return pipe


# ── Generate one image ────────────────────────────────────────────────────────

def generate_image(pipe, variant: dict, base_prompt: str) -> Path:
    out_path = AVATARS_DIR / f"{variant['key']}.png"

    if out_path.exists():
        print(f"  SKIP {variant['label']} — already exists")
        return out_path

    full_prompt = f"{base_prompt}, {variant['suffix']}"
    print(f"  GEN  {variant['label']}...")

    # Deterministic seed per variant (reproducible across runs)
    seed = hash(variant["key"]) % (2 ** 32)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE,
            num_inference_steps=variant.get("steps", 40),
            guidance_scale=variant.get("guidance", 7.5),
            height=1024,
            width=768,        # portrait 3:4 aspect
            generator=generator,
        )

    img = result.images[0]

    # Crop to top 85% for half-body framing
    w, h = img.size
    img_crop = img.crop((0, 0, w, int(h * 0.85)))
    img_crop = img_crop.resize((768, 1024), Image.LANCZOS)
    img_crop.save(str(out_path), "PNG")

    size_kb = out_path.stat().st_size // 1024
    print(f"       Saved → {out_path.name} ({size_kb} KB)")
    return out_path


# ── Save manifest ─────────────────────────────────────────────────────────────

def save_manifest(generated: list[dict]):
    manifest_path = AVATARS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"\nManifest → {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   Avatar Studio — Image Generator   ║")
    print("╚══════════════════════════════════════╝")
    print(f"Model : SDXL (stabilityai/stable-diffusion-xl-base-1.0)")
    print(f"Output: {AVATARS_DIR}")

    all_keys = [v["key"] for char in AVATARS.values() for v in char["variants"]]
    existing = [k for k in all_keys if (AVATARS_DIR / f"{k}.png").exists()]
    if len(existing) == len(all_keys):
        print("\nAll 6 avatar images already exist — nothing to generate.")
        print("Delete images in models/avatars/ to regenerate.")
        return

    pipe = load_pipeline()
    generated = []
    total = sum(len(char["variants"]) for char in AVATARS.values())
    done = 0

    for char_id, char_data in AVATARS.items():
        print(f"\n─── {char_id.upper()} avatars ───")
        for variant in char_data["variants"]:
            done += 1
            print(f"\n[{done}/{total}]", end=" ")
            out_path = generate_image(pipe, variant, char_data["base"])
            generated.append({
                "character": char_id,
                "key": variant["key"],
                "label": variant["label"],
                "path": str(out_path),
                "size_kb": round(out_path.stat().st_size / 1024),
            })

    del pipe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    save_manifest(generated)

    print("\n╔══════════════════════════════════════╗")
    print("║       Generation Complete ✓          ║")
    print("╚══════════════════════════════════════╝")
    for item in generated:
        print(f"  {item['character']:6} | {item['label']:35} | {item['size_kb']} KB")
    print(f"\nTotal: {len(generated)} avatars in {AVATARS_DIR}")


if __name__ == "__main__":
    main()
