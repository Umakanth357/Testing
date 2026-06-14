"""
Avatar Engine — Generate photorealistic South Indian avatars using FLUX.1-schnell.
Apache 2.0 licensed. Commercial safe.

Generates multiple poses and attires per persona at setup time.
Avatars are locked (generated once, reused) for consistency across videos.
"""
import logging
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from config import (
    FLUX_MODEL, FLUX_LORA, FLUX_STEPS, AVATARS, AVATARS_DIR,
    POSE_PROMPTS, ATTIRE_PROMPTS, DEVICE, DTYPE,
)

log = logging.getLogger("avatar_engine")

# FLUX pipeline (loaded once, kept in memory)
_flux_pipe = None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_all_avatars(force: bool = False) -> dict[str, list[Path]]:
    """
    Generate all persona × pose × attire combinations.
    Skips existing images unless force=True.
    Returns {persona_id: [image_path, ...]}
    """
    results = {}
    for persona_id, persona in AVATARS.items():
        paths = []
        for pose in persona["poses"]:
            for attire in persona["attires"]:
                img_path = _avatar_path(persona_id, pose, attire)
                if img_path.exists() and not force:
                    log.info(f"Skipping {img_path.name} (exists)")
                    paths.append(img_path)
                    continue
                path = generate_avatar(persona_id, pose, attire)
                if path:
                    paths.append(path)
        results[persona_id] = paths
    return results


def generate_avatar(persona_id: str, pose: str = "half_body",
                    attire: str = "professional") -> Optional[Path]:
    """
    Generate a single avatar image. Returns path or None on failure.
    """
    persona = AVATARS.get(persona_id)
    if not persona:
        log.error(f"Unknown persona: {persona_id}")
        return None

    out_path = _avatar_path(persona_id, pose, attire)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt  = _build_prompt(persona, pose, attire)
    neg     = _negative_prompt()

    log.info(f"Generating {persona_id}/{pose}/{attire}")

    # Try FLUX first
    img = _generate_flux(prompt, neg)
    if img is None:
        log.warning("FLUX failed — falling back to SD 1.5")
        img = _generate_sd_fallback(prompt, neg)
    if img is None:
        log.error(f"Avatar generation failed for {persona_id}")
        return None

    # Enhance face quality
    img = _enhance_face(img)

    # Ensure minimum resolution
    img = _ensure_resolution(img, min_size=768)

    img.save(str(out_path), quality=95)
    log.info(f"Saved: {out_path}")
    return out_path


def get_avatar_path(persona_id: str, pose: str = "half_body",
                    attire: str = "professional") -> Optional[Path]:
    """Return path to existing avatar image, or None if not generated yet."""
    path = _avatar_path(persona_id, pose, attire)
    return path if path.exists() else None


def list_available_avatars() -> dict:
    """Return all generated avatar images grouped by persona."""
    result = {}
    for persona_id in AVATARS:
        result[persona_id] = []
        for pose in AVATARS[persona_id]["poses"]:
            for attire in AVATARS[persona_id]["attires"]:
                p = _avatar_path(persona_id, pose, attire)
                if p.exists():
                    result[persona_id].append({
                        "pose": pose, "attire": attire, "path": str(p)
                    })
    return result


# ── FLUX.1-schnell ────────────────────────────────────────────────────────────

def _generate_flux(prompt: str, negative_prompt: str) -> Optional[Image.Image]:
    global _flux_pipe
    try:
        if _flux_pipe is None:
            log.info(f"Loading FLUX.1-schnell from {FLUX_MODEL}...")
            from diffusers import FluxPipeline

            _flux_pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL,
                torch_dtype=DTYPE,
            )

            # Load South Indian LoRA
            try:
                _flux_pipe.load_lora_weights(
                    FLUX_LORA,
                    weight_name="desi_espresso_flux.safetensors",
                )
                _flux_pipe.fuse_lora(lora_scale=0.75)
                log.info("Desi Espresso LoRA loaded")
            except Exception as e:
                log.warning(f"LoRA load failed (will use base FLUX): {e}")

            _flux_pipe = _flux_pipe.to(DEVICE)
            if DEVICE == "cuda":
                _flux_pipe.enable_model_cpu_offload()

        with torch.inference_mode():
            result = _flux_pipe(
                prompt=prompt,
                num_inference_steps=FLUX_STEPS,
                guidance_scale=0.0,          # FLUX schnell = 0 guidance
                width=768,
                height=1024,
            )
        return result.images[0]

    except torch.cuda.OutOfMemoryError:
        log.error("FLUX OOM — not enough VRAM. Try g5.2xlarge for A10G.")
        _flux_pipe = None
        return None
    except Exception as e:
        log.error(f"FLUX generation failed: {e}")
        _flux_pipe = None
        return None


# ── SD 1.5 Fallback ───────────────────────────────────────────────────────────

_sd_pipe = None

def _generate_sd_fallback(prompt: str, negative_prompt: str) -> Optional[Image.Image]:
    global _sd_pipe
    try:
        if _sd_pipe is None:
            from diffusers import StableDiffusionPipeline
            log.info("Loading SD 1.5 fallback...")
            _sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=DTYPE,
                safety_checker=None,
            ).to(DEVICE)

        with torch.inference_mode():
            result = _sd_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                width=512, height=768,
            )
        return result.images[0]
    except Exception as e:
        log.error(f"SD fallback failed: {e}")
        _sd_pipe = None
        return None


# ── Face Enhancement ──────────────────────────────────────────────────────────

def _enhance_face(img: Image.Image) -> Image.Image:
    """Run GFPGAN face enhancement to sharpen AI-generated face."""
    try:
        import cv2
        import numpy as np

        from gfpgan import GFPGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # Check for model weights
        gfpgan_weight = Path("models/weights/GFPGANv1.4.pth")
        if not gfpgan_weight.exists():
            log.warning("GFPGAN weights not found — skipping face enhancement")
            return img

        bg_upsampler = RealESRGANer(
            scale=2,
            model_path=str(Path("models/weights/RealESRGAN_x2plus.pth")),
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=2),
            tile=400, tile_pad=10, pre_pad=0, half=DEVICE=="cuda",
        ) if Path("models/weights/RealESRGAN_x2plus.pth").exists() else None

        restorer = GFPGANer(
            model_path=str(gfpgan_weight),
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )

        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _, _, restored = restorer.enhance(
            img_np, has_aligned=False, only_center_face=False, paste_back=True
        )
        return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))

    except Exception as e:
        log.warning(f"Face enhancement skipped: {e}")
        return img


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_prompt(persona: dict, pose: str, attire: str) -> str:
    base    = persona["flux_prompt_base"]
    pose_p  = POSE_PROMPTS.get(pose, POSE_PROMPTS["half_body"])
    attire_p = ATTIRE_PROMPTS.get(attire, ATTIRE_PROMPTS["professional"])
    quality = ("ultra high quality, photorealistic, sharp focus, "
               "professional photography, natural skin texture, "
               "detailed face, expressive eyes, no deformities")
    return f"{base}, {pose_p}, {attire_p}, {quality}"


def _negative_prompt() -> str:
    return (
        "cartoon, anime, illustration, painting, low quality, blurry, "
        "deformed, ugly, extra limbs, bad hands, bad anatomy, "
        "watermark, text, logo, nsfw, extra fingers, mutated hands, "
        "poorly drawn face, mutation, bad proportions"
    )


def _avatar_path(persona_id: str, pose: str, attire: str) -> Path:
    return AVATARS_DIR / persona_id / f"{pose}_{attire}.png"


def _ensure_resolution(img: Image.Image, min_size: int = 768) -> Image.Image:
    w, h = img.size
    if w < min_size or h < min_size:
        scale = min_size / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img
