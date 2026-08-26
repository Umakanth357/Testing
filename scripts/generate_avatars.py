#!/usr/bin/env python3
"""
Avatar Studio — Avatar Image Generator
Generates Navya Reddy + Arjun Varma avatar images using FLUX.1-schnell + Desi Espresso LoRA.
Run once after setup. Images are saved to models/avatars/ and never regenerated unless deleted.

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
from diffusers import FluxPipeline
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
AVATARS_DIR = ROOT / "models" / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)

FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"   # Apache 2.0 — commercial safe
FLUX_LORA  = "prithivMLmods/Desi-Espresso-Flux"   # South Indian faces LoRA

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN   = os.environ.get("HF_TOKEN")

print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'none'}")

# ── Avatar definitions ────────────────────────────────────────────────────────
# Navya Reddy — young (22), attractive, South Indian Hyderabad Telugu girl
# Sweet + strong + confident. Must look unique and non-existent (no real person).
NAVYA_BASE = (
    "photorealistic south indian telugu woman, 22 years old, attractive hyderabad girl, "
    "warm brown skin, long dark hair, sharp expressive eyes, confident warm smile, "
    "high cheekbones, elegant facial structure, not a real person, unique face, "
    "8k resolution, studio lighting, sharp focus, professional portrait, "
    "Desi Espresso style"
)

ARJUN_BASE = (
    "photorealistic south indian telugu man, 32 years old, vijayawada professional, "
    "warm brown skin, short dark hair, strong jawline, intelligent eyes, "
    "confident composed expression, not a real person, unique face, "
    "8k resolution, studio lighting, sharp focus, professional portrait, "
    "Desi Espresso style"
)

NEGATIVE = (
    "blurry, low quality, watermark, text, logo, cartoon, anime, painting, "
    "drawing, ugly, deformed, bad anatomy, extra fingers, mutation, "
    "poorly drawn face, nsfw, celebrity, real person, famous person"
)

AVATARS = {
    "navya": {
        "base": NAVYA_BASE,
        "variants": [
            {
                "key": "navya_professional",
                "label": "Navya — Professional Blazer",
                "suffix": "wearing a sharp navy blue blazer, formal professional attire, half body shot, waist up",
                "steps": 4, "guidance": 0.0,
            },
            {
                "key": "navya_traditional",
                "label": "Navya — Traditional Saree",
                "suffix": "wearing an elegant silk saree in deep red and gold, traditional south indian style, half body shot",
                "steps": 4, "guidance": 0.0,
            },
            {
                "key": "navya_casual",
                "label": "Navya — Smart Casual",
                "suffix": "wearing smart casual clothes, warm approachable look, half body shot",
                "steps": 4, "guidance": 0.0,
            },
        ],
    },
    "arjun": {
        "base": ARJUN_BASE,
        "variants": [
            {
                "key": "arjun_suit",
                "label": "Arjun — Dark Suit",
                "suffix": "wearing a dark charcoal suit, white shirt, professional business attire, half body shot, waist up",
                "steps": 4, "guidance": 0.0,
            },
            {
                "key": "arjun_kurta",
                "label": "Arjun — Formal Kurta",
                "suffix": "wearing a formal white kurta, traditional indian male attire, half body shot",
                "steps": 4, "guidance": 0.0,
            },
            {
                "key": "arjun_casual",
                "label": "Arjun — Smart Casual",
                "suffix": "wearing smart casual polo shirt, relaxed professional look, half body shot",
                "steps": 4, "guidance": 0.0,
            },
        ],
    },
}

# ── Load pipeline ─────────────────────────────────────────────────────────────

def load_pipeline():
    print("\nLoading FLUX.1-schnell pipeline...")
    print("  (First run downloads ~7GB of model weights — this takes ~5-10 min)")

    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL,
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )

    print("Loading Desi Espresso LoRA (South Indian faces)...")
    try:
        pipe.load_lora_weights(FLUX_LORA, token=HF_TOKEN)
        pipe.fuse_lora(lora_scale=0.8)
        print("  LoRA loaded OK")
    except Exception as e:
        print(f"  LoRA load failed ({e}) — continuing without LoRA (quality may be lower)")

    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


# ── Generate one image ────────────────────────────────────────────────────────

def generate_image(pipe, character: str, variant: dict, base_prompt: str) -> Path:
    out_path = AVATARS_DIR / f"{variant['key']}.png"

    if out_path.exists():
        print(f"  SKIP {variant['label']} — already exists: {out_path.name}")
        return out_path

    full_prompt = f"{base_prompt}, {variant['suffix']}"
    print(f"  GEN  {variant['label']}...")
    print(f"       Prompt: {full_prompt[:120]}...")

    with torch.inference_mode():
        result = pipe(
            prompt=full_prompt,
            num_inference_steps=variant.get("steps", 4),
            guidance_scale=variant.get("guidance", 0.0),
            height=1024,
            width=768,   # portrait aspect for half-body
            generator=torch.Generator(device=DEVICE).manual_seed(
                hash(variant["key"]) % (2**32)   # deterministic seed per variant
            ),
        )

    img = result.images[0]

    # Crop to half-body if image is full body (heuristic: keep top 75%)
    w, h = img.size
    img_crop = img.crop((0, 0, w, int(h * 0.85)))
    img_crop = img_crop.resize((768, 1024), Image.LANCZOS)
    img_crop.save(str(out_path), "PNG", quality=95)

    print(f"       Saved: {out_path}")
    return out_path


# ── Save manifest ─────────────────────────────────────────────────────────────

def save_manifest(generated: list[dict]):
    manifest_path = AVATARS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   Avatar Studio — Image Generator   ║")
    print("╚══════════════════════════════════════╝")
    print(f"Output: {AVATARS_DIR}")

    # Check if all already exist
    all_keys = [v["key"] for char in AVATARS.values() for v in char["variants"]]
    existing = [k for k in all_keys if (AVATARS_DIR / f"{k}.png").exists()]
    if len(existing) == len(all_keys):
        print("\nAll avatar images already exist — nothing to generate.")
        print("Delete images from models/avatars/ to regenerate.")
        return

    pipe = load_pipeline()
    generated = []
    total = sum(len(char["variants"]) for char in AVATARS.values())
    done = 0

    for char_id, char_data in AVATARS.items():
        print(f"\n--- Generating {char_id.upper()} avatars ---")
        for variant in char_data["variants"]:
            done += 1
            print(f"\n[{done}/{total}]", end=" ")
            out_path = generate_image(pipe, char_id, variant, char_data["base"])
            generated.append({
                "character": char_id,
                "key": variant["key"],
                "label": variant["label"],
                "path": str(out_path),
                "size_kb": round(out_path.stat().st_size / 1024),
            })

    # Clean up GPU memory
    del pipe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    save_manifest(generated)

    print("\n╔══════════════════════════════════════╗")
    print("║         Generation Complete ✓        ║")
    print("╚══════════════════════════════════════╝")
    for item in generated:
        print(f"  {item['character']:6} | {item['label']:35} | {item['size_kb']} KB")

    print(f"\nTotal: {len(generated)} images in {AVATARS_DIR}")
    print("Avatars are locked. App will use these faces consistently.")


if __name__ == "__main__":
    main()
