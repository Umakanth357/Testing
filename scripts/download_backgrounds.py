#!/usr/bin/env python3
"""
Avatar Studio — Background Generator
Creates solid-color and gradient placeholder backgrounds for all scene types.
No internet needed — pure PIL generation. Run once after setup.

Usage: python scripts/download_backgrounds.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFilter
import colorsys

BG_DIR = ROOT / "models" / "backgrounds"
BG_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080   # 16:9 Full HD

# ── Scene background definitions ──────────────────────────────────────────────
# Each entry: (filename, type, colors, description)
# type: "solid", "gradient", "vignette"

BACKGROUNDS = [
    # Professional
    ("office_loop.mp4",        "gradient",  ["#1a1a2e", "#16213e", "#0f3460"],  "Deep blue office night"),
    ("news_desk_loop.mp4",     "vignette",  ["#0d0d0d", "#1a1a1a", "#141414"],  "Dark studio news"),
    ("seminar_hall_loop.mp4",  "gradient",  ["#1e3a5f", "#0d2137", "#142840"],  "Conference blue"),
    ("stage_dias_loop.mp4",    "gradient",  ["#2c0a37", "#1a0525", "#3d1254"],  "Stage purple"),
    ("conference_loop.mp4",    "gradient",  ["#1a2f1a", "#0d200d", "#213321"],  "Corporate green"),
    # Nature
    ("glacier_loop.mp4",       "gradient",  ["#a8d8f0", "#c5e8ff", "#7fc0e0"],  "Glacier blue sky"),
    ("beach_loop.mp4",         "gradient",  ["#4a90d9", "#6bb8f0", "#87ceeb"],  "Beach blue sky"),
    ("forest_loop.mp4",        "gradient",  ["#1a3d1a", "#0d2d0d", "#2a5a2a"],  "Forest green"),
    ("mountain_loop.mp4",      "gradient",  ["#8fb4cc", "#b0cde0", "#6a9ab5"],  "Mountain sky"),
    # Casual
    ("kitchen_loop.mp4",       "gradient",  ["#f5e6d3", "#eedfc5", "#e8d4b5"],  "Warm kitchen"),
    ("living_room_loop.mp4",   "gradient",  ["#d4c4a8", "#c8b896", "#bfaa84"],  "Living room warm"),
    ("bedroom_loop.mp4",       "gradient",  ["#e8d4e8", "#dcc8dc", "#d0bcd0"],  "Bedroom soft"),
    ("cafe_loop.mp4",          "gradient",  ["#6b4226", "#4a2d18", "#7d4f2e"],  "Cafe brown"),
    ("rooftop_loop.mp4",       "gradient",  ["#ff7e5f", "#feb47b", "#e8734a"],  "Sunset orange"),
    # Landmark
    ("red_fort_loop.mp4",      "gradient",  ["#c45c1a", "#8b3a0d", "#d4703a"],  "Red fort warm"),
    ("parliament_loop.mp4",    "gradient",  ["#1c3a5e", "#0f2540", "#2a4f7a"],  "Parliament blue"),
    ("tech_park_loop.mp4",     "gradient",  ["#1a2633", "#0d1a26", "#223344"],  "Tech dark"),
    ("market_loop.mp4",        "gradient",  ["#c4922a", "#9b6e1a", "#d4a840"],  "Market gold"),
    # Brand
    ("google_stage_loop.mp4",  "gradient",  ["#0d0d0d", "#1a1a1a", "#0a0a0a"],  "Google dark stage"),
    ("apple_stage_loop.mp4",   "solid",     ["#000000"],                          "Apple black"),
    ("samsung_stage_loop.mp4", "gradient",  ["#001e3c", "#00345a", "#001428"],    "Samsung navy"),
    ("ted_stage_loop.mp4",     "gradient",  ["#cc0000", "#990000", "#dd1111"],    "TED red"),
    # Abstract
    ("dark_studio_loop.mp4",   "vignette",  ["#0a0a0a", "#1a1a1a", "#050505"],   "Dark studio"),
    ("gradient_blue_loop.mp4", "gradient",  ["#0033aa", "#0055cc", "#0044bb"],    "Gradient blue"),
]


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def make_gradient(colors: list[str], w: int = W, h: int = H) -> Image.Image:
    """Create a smooth top-to-bottom gradient between 2-3 colors."""
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    c1 = hex_to_rgb(colors[0])
    c2 = hex_to_rgb(colors[-1])
    for y in range(h):
        t = y / h
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def make_solid(colors: list[str], w: int = W, h: int = H) -> Image.Image:
    return Image.new("RGB", (w, h), hex_to_rgb(colors[0]))


def make_vignette(colors: list[str], w: int = W, h: int = H) -> Image.Image:
    """Dark center-fade vignette."""
    base = make_gradient(colors, w, h)
    vignette = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(vignette)
    cx, cy = w // 2, h // 2
    for r in range(max(w, h), 0, -1):
        alpha = int(255 * (1 - r / max(w, h)) * 0.6)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=alpha)
    vignette = vignette.filter(ImageFilter.GaussianBlur(80))
    overlay = Image.new("RGB", (w, h), (0, 0, 0))
    base = Image.composite(base, overlay, vignette)
    return base


def save_as_video_placeholder(img: Image.Image, path: Path):
    """
    Save as PNG (app falls back to static image if .mp4 not found).
    Also save as .mp4 placeholder name so config lookups don't fail.
    """
    png_path = path.with_suffix(".png")
    img.save(str(png_path), "PNG")
    # Create a symlink from .mp4 → .png for code that does Path(...).exists()
    if not path.exists():
        try:
            path.symlink_to(png_path.name)
        except Exception:
            # Windows doesn't support symlinks easily — just copy
            img.save(str(path.with_suffix(".png")), "PNG")


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║  Avatar Studio — Background Generator ║")
    print("╚══════════════════════════════════════╝")
    print(f"Output: {BG_DIR}")
    print(f"Resolution: {W}x{H}")
    print()

    skip = done = 0
    for filename, bg_type, colors, label in BACKGROUNDS:
        out_path = BG_DIR / filename
        png_path = out_path.with_suffix(".png")

        if png_path.exists():
            print(f"  SKIP {label}")
            skip += 1
            continue

        if bg_type == "solid":
            img = make_solid(colors)
        elif bg_type == "vignette":
            img = make_vignette(colors)
        else:
            img = make_gradient(colors)

        img.save(str(png_path), "PNG")
        print(f"  GEN  {label:<35} → {png_path.name}")
        done += 1

    print(f"\nDone: {done} generated, {skip} skipped.")
    print("Backgrounds saved as PNG files in models/backgrounds/")
    print("\nNote: These are static placeholders. For animated backgrounds,")
    print("replace .png files with actual .mp4 video loops (same filenames).")


if __name__ == "__main__":
    main()
