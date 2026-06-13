"""
download_weights.py
Downloads all required model weights for the Avatar Tool.
Called automatically by setup.sh (Step 5).
Run manually: python scripts/download_weights.py
"""
import os
import sys
import logging
import urllib.request
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT    = Path(__file__).parent.parent
WEIGHTS = ROOT / "models" / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, desc: str):
    """Download a file with progress logging."""
    if dest.exists() and dest.stat().st_size > 100_000:
        logger.info(f"  ✓ {desc} already present ({dest.stat().st_size // 1_000_000} MB), skipping")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  ↓ Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size // 1_000_000
        logger.info(f"  ✓ {desc} downloaded ({size_mb} MB)")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to download {desc}: {e}")
        return False


def download_via_huggingface_hub(repo_id: str, filename: str, dest: Path, desc: str):
    """Download from HuggingFace Hub using huggingface_hub."""
    if dest.exists() and dest.stat().st_size > 100_000:
        logger.info(f"  ✓ {desc} already present, skipping")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"  ↓ Downloading {desc} from HuggingFace ({repo_id})...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent))
        if Path(path) != dest:
            import shutil
            shutil.move(path, str(dest))
        logger.info(f"  ✓ {desc} downloaded")
        return True
    except Exception as e:
        logger.error(f"  ✗ huggingface_hub download failed for {desc}: {e}")
        return False


# ── GFPGAN ────────────────────────────────────────────────────────
def download_gfpgan():
    logger.info("\n── GFPGAN weights ──")
    return download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        WEIGHTS / "GFPGANv1.4.pth",
        "GFPGANv1.4.pth (~332 MB)"
    )


# ── MuseTalk ──────────────────────────────────────────────────────
def download_musetalk():
    logger.info("\n── MuseTalk weights ──")
    musetalk_dir = WEIGHTS / "musetalk"
    musetalk_dir.mkdir(exist_ok=True)

    files = [
        ("TMElyralab/MuseTalk", "models/musetalk/pytorch_model.bin",     musetalk_dir / "pytorch_model.bin",     "MuseTalk UNet weights (~500 MB)"),
        ("TMElyralab/MuseTalk", "models/musetalk/musetalk.json",          musetalk_dir / "musetalk.json",          "MuseTalk config"),
        ("TMElyralab/MuseTalk", "models/dwpose/dw-ll_ucoco_384.pth",      WEIGHTS / "dwpose" / "dw-ll_ucoco_384.pth", "DWPose weights (~250 MB)"),
        ("TMElyralab/MuseTalk", "models/sd-vae-ft-mse/diffusion_pytorch_model.bin",
         WEIGHTS / "sd-vae-ft-mse" / "diffusion_pytorch_model.bin", "SD VAE weights (~335 MB)"),
        ("TMElyralab/MuseTalk", "models/whisper/tiny.pt", WEIGHTS / "whisper" / "tiny.pt", "Whisper tiny (~75 MB)"),
    ]

    ok = True
    for repo_id, filename, dest, desc in files:
        ok &= download_via_huggingface_hub(repo_id, filename, dest, desc)
    return ok


# ── LivePortrait ──────────────────────────────────────────────────
def download_liveportrait():
    logger.info("\n── LivePortrait weights ──")
    lp_dir = WEIGHTS / "liveportrait"
    lp_dir.mkdir(exist_ok=True)

    files = [
        ("KwaiVGI/LivePortrait", "pretrained_weights/spade_generator.safetensors",
         lp_dir / "spade_generator.safetensors", "LivePortrait generator (~200 MB)"),
        ("KwaiVGI/LivePortrait", "pretrained_weights/motion_extractor.safetensors",
         lp_dir / "motion_extractor.safetensors", "LivePortrait motion extractor"),
        ("KwaiVGI/LivePortrait", "pretrained_weights/appearance_feature_extractor.safetensors",
         lp_dir / "appearance_feature_extractor.safetensors", "LivePortrait feature extractor"),
        ("KwaiVGI/LivePortrait", "pretrained_weights/warping_module.safetensors",
         lp_dir / "warping_module.safetensors", "LivePortrait warping module"),
        ("KwaiVGI/LivePortrait", "pretrained_weights/stitching_retargeting_module.safetensors",
         lp_dir / "stitching_retargeting_module.safetensors", "LivePortrait stitching module"),
    ]

    ok = True
    for repo_id, filename, dest, desc in files:
        ok &= download_via_huggingface_hub(repo_id, filename, dest, desc)
    return ok


# ── SadTalker ─────────────────────────────────────────────────────
def download_sadtalker():
    logger.info("\n── SadTalker weights (fallback) ──")
    st_dir = WEIGHTS / "sadtalker"
    st_dir.mkdir(exist_ok=True)

    files = [
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
         st_dir / "SadTalker_V0.0.2_256.safetensors", "SadTalker weights 256px (~540 MB)"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
         st_dir / "alignment_WFLW_4HG.pth", "Face alignment weights"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
         st_dir / "detection_Resnet50_Final.pth", "Face detection weights"),
    ]

    for url, dest, desc in files:
        download_file(url, dest, desc)
    return True


# ── Run all ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  AVATAR TOOL — Model Weight Downloader")
    logger.info("=" * 55)

    results = {}
    results["gfpgan"]      = download_gfpgan()
    results["musetalk"]    = download_musetalk()
    results["liveportrait"] = download_liveportrait()
    results["sadtalker"]   = download_sadtalker()

    logger.info("\n" + "=" * 55)
    passed = sum(results.values())
    total  = len(results)
    logger.info(f"  Downloads: {passed}/{total} successful")
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        logger.info(f"  {icon} {name}")
    logger.info("=" * 55)

    if not results["gfpgan"] or not results["musetalk"] or not results["liveportrait"]:
        logger.error("\nCritical weights missing — video generation will not work.")
        sys.exit(1)
    logger.info("\nAll critical weights downloaded. Run test_all.py to verify.")
