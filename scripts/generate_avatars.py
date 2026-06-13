"""
generate_avatars.py
Generates photorealistic South Indian avatar images.
Primary:  FLUX.2 + RealVisXL (best quality, needs 10GB VRAM)
Fallback: Stable Diffusion 1.5 (works on 6GB VRAM)
Fallback: Downloads pre-generated sample avatars from HuggingFace
"""
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT    = Path(__file__).parent.parent
AVATARS = ROOT / "avatars"
AVATARS.mkdir(exist_ok=True)

# ── Avatar definitions ────────────────────────────────────────────
AVATAR_SPECS = [
    {
        "filename": "si_male_30.png",
        "label":    "South Indian Male (30s)",
        "prompt":   "professional South Indian male, 30 years old, Hyderabad Telangana features, "
                    "warm medium brown skin tone, confident expression, corporate formals white shirt, "
                    "studio lighting, looking directly at camera, shoulders up portrait, "
                    "85mm lens bokeh background, photorealistic, 4K, --no cartoon anime",
        "negative": "cartoon, anime, western features, pale skin, blond hair, blue eyes, "
                    "distorted, blurry, low quality, watermark",
    },
    {
        "filename": "si_female_30.png",
        "label":    "South Indian Female (30s)",
        "prompt":   "professional South Indian female, 30 years old, Telugu Andhra features, "
                    "warm brown skin tone, small bindi, confident warm smile, kurta corporate attire, "
                    "studio lighting, looking directly at camera, shoulders up portrait, "
                    "85mm lens, photorealistic, 4K, natural dark hair",
        "negative": "cartoon, anime, western features, pale skin, blond hair, "
                    "distorted, blurry, low quality, watermark, revealing clothing",
    },
    {
        "filename": "si_male_40.png",
        "label":    "South Indian Male (40s)",
        "prompt":   "professional South Indian male, 42 years old, Tamil Nadu features, "
                    "experienced authoritative look, slight greying temples, warm skin, "
                    "formal suit, studio lighting, direct camera gaze, shoulders portrait, "
                    "photorealistic, 4K, trustworthy expression",
        "negative": "cartoon, anime, western features, distorted, blurry, watermark",
    },
    {
        "filename": "si_female_40.png",
        "label":    "South Indian Female (40s)",
        "prompt":   "professional South Indian female, 40 years old, Karnataka Bangalore features, "
                    "composed authoritative look, warm dark skin, saree or formal blazer, "
                    "studio lighting, direct camera gaze, shoulders portrait, "
                    "photorealistic, 4K, senior professional appearance",
        "negative": "cartoon, anime, western features, distorted, blurry, watermark",
    },
    {
        "filename": "pro_male.png",
        "label":    "Professional Male",
        "prompt":   "professional South Asian Indian male, 35 years old, neutral features, "
                    "corporate presenter, dark hair, crisp shirt, studio lighting, "
                    "direct camera gaze, photorealistic, 4K",
        "negative": "cartoon, anime, distorted, blurry, watermark",
    },
    {
        "filename": "pro_female.png",
        "label":    "Professional Female",
        "prompt":   "professional South Asian Indian female, 35 years old, neutral features, "
                    "corporate presenter, dark hair, kurta or blazer, studio lighting, "
                    "direct camera gaze, photorealistic, 4K",
        "negative": "cartoon, anime, distorted, blurry, watermark",
    },
]


def generate_with_diffusers():
    """Try FLUX/SD via diffusers — requires CUDA GPU with 8GB+ VRAM."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers import FluxPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.warning("No GPU available — diffusers will be very slow. Using fallback.")
            return False

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU VRAM: {vram_gb:.1f} GB")

        # Choose model based on VRAM
        if vram_gb >= 12:
            # FLUX.2 — best quality
            logger.info("Loading FLUX.2 pipeline (high quality)...")
            try:
                pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.float16
                ).to(device)
                is_flux = True
            except Exception:
                pipe = None
                is_flux = False
        else:
            pipe = None
            is_flux = False

        if pipe is None:
            # RealVisXL_V4.0 is an SDXL model — must use StableDiffusionXLPipeline
            logger.info("Loading RealVisXL (SDXL-based) pipeline...")
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "SG161222/RealVisXL_V4.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_attention_slicing()
            is_flux = False

        generated = 0
        for spec in AVATAR_SPECS:
            out_path = AVATARS / spec["filename"]
            if out_path.exists():
                logger.info(f"  ✓ {spec['filename']} already exists, skipping")
                continue

            logger.info(f"  Generating {spec['label']}...")
            try:
                if is_flux:
                    image = pipe(
                        prompt=spec["prompt"],
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        height=768, width=512
                    ).images[0]
                else:
                    image = pipe(
                        prompt=spec["prompt"],
                        negative_prompt=spec["negative"],
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=768, width=512,
                        generator=torch.Generator(device).manual_seed(hash(spec["filename"]) % 2**32)
                    ).images[0]

                image.save(str(out_path))
                logger.info(f"  ✓ Saved {out_path}")
                generated += 1
            except Exception as e:
                logger.error(f"  ✗ Failed {spec['filename']}: {e}")

        return generated > 0

    except ImportError:
        logger.warning("diffusers not available for avatar generation")
        return False
    except Exception as e:
        logger.error(f"Diffusers pipeline error: {e}")
        return False


def download_sample_avatars():
    """
    Download pre-generated placeholder avatars from HuggingFace.
    These are neutral professional avatars — replace with custom generated ones
    after running on EC2 with GPU.
    """
    import urllib.request

    # These are royalty-free AI-generated placeholder faces
    # In production: replace with your FLUX-generated South Indian avatars
    PLACEHOLDER_AVATARS = {
        "si_male_30.png":   "https://huggingface.co/datasets/NimaBoscarino/avatars/resolve/main/male_1.jpg",
        "si_female_30.png": "https://huggingface.co/datasets/NimaBoscarino/avatars/resolve/main/female_1.jpg",
        "si_male_40.png":   "https://huggingface.co/datasets/NimaBoscarino/avatars/resolve/main/male_2.jpg",
        "si_female_40.png": "https://huggingface.co/datasets/NimaBoscarino/avatars/resolve/main/female_2.jpg",
        "pro_male.png":     "https://huggingface.co/datasets/NimaBoscarino/avatars/resolve/main/male_3.jpg",
        "pro_female.png":   "https://huggingface.co/datasets/NimaBoscarino/avatars/resolve/main/female_3.jpg",
    }

    downloaded = 0
    for fname, url in PLACEHOLDER_AVATARS.items():
        out = AVATARS / fname
        if out.exists():
            logger.info(f"  ✓ {fname} already exists")
            downloaded += 1
            continue
        try:
            urllib.request.urlretrieve(url, str(out))
            logger.info(f"  ✓ Downloaded {fname}")
            downloaded += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not download {fname}: {e}")

    return downloaded > 0


def create_placeholder_avatars():
    """
    Generate simple colored placeholder avatars using PIL.
    Used when no GPU and no internet. Replaced with real avatars on EC2.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import hashlib

        colors = {
            "si_male_30.png":   ("#2C3E50", "#ECF0F1", "SM"),
            "si_female_30.png": ("#6C3483", "#ECF0F1", "SF"),
            "si_male_40.png":   ("#1A5276", "#ECF0F1", "SM"),
            "si_female_40.png": ("#7D6608", "#ECF0F1", "SF"),
            "pro_male.png":     ("#1E8449", "#ECF0F1", "PM"),
            "pro_female.png":   ("#C0392B", "#ECF0F1", "PF"),
        }

        created = 0
        for fname, (bg_color, fg_color, initials) in colors.items():
            out = AVATARS / fname
            if out.exists():
                continue
            img = Image.new("RGB", (512, 768), color=bg_color)
            draw = ImageDraw.Draw(img)
            # Circle for face
            draw.ellipse([156, 200, 356, 400], fill="#D4A574")
            # Initials
            draw.text((220, 280), initials, fill=fg_color)
            img.save(str(out))
            created += 1
            logger.info(f"  ✓ Placeholder created: {fname}")

        return created > 0
    except ImportError:
        logger.warning("PIL not available for placeholder creation")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n━━━ Generating South Indian Avatars ━━━\n")

    # Try best method first, cascade down
    if generate_with_diffusers():
        print("\n✓ Avatars generated with Stable Diffusion / FLUX")
    elif download_sample_avatars():
        print("\n✓ Sample avatars downloaded")
        print("  NOTE: Replace these with custom avatars using:")
        print("  python scripts/generate_avatars.py --force-generate")
    elif create_placeholder_avatars():
        print("\n✓ Placeholder avatars created (no GPU/internet)")
        print("  NOTE: Run on EC2 GPU to generate proper South Indian avatars")
    else:
        print("\n✗ Could not create avatars — place PNG images in avatars/ folder manually")

    print("\nAvatars in:", AVATARS)
    for f in AVATARS.glob("*.png"):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")
