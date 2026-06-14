"""
Download CC0 background video loops from Pexels/Pixabay.
All videos are Creative Commons Zero (public domain) — safe for commercial use.

For brand stages (Google, Apple, Samsung): uses publicly available footage
cropped as backgrounds. Internal use only — verify before public deployment.
"""
import sys
import logging
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("download_backgrounds")

from config import BACKGROUNDS_DIR, SFX_DIR

# CC0 video backgrounds from Pexels (public domain)
# Format: filename → Pexels video URL
# NOTE: Pexels requires a free API key for programmatic download.
# If URL fails, download manually from pexels.com and place in models/backgrounds/
BACKGROUND_VIDEOS = {
    "office_loop.mp4":        "https://www.pexels.com/video/852390/",
    "news_desk_loop.mp4":     "https://www.pexels.com/video/3753705/",
    "seminar_hall_loop.mp4":  "https://www.pexels.com/video/2278095/",
    "stage_dias_loop.mp4":    "https://www.pexels.com/video/2795405/",
    "conference_loop.mp4":    "https://www.pexels.com/video/3130284/",
    "glacier_loop.mp4":       "https://www.pexels.com/video/857136/",
    "beach_loop.mp4":         "https://www.pexels.com/video/854643/",
    "forest_loop.mp4":        "https://www.pexels.com/video/1448735/",
    "mountain_loop.mp4":      "https://www.pexels.com/video/855117/",
    "kitchen_loop.mp4":       "https://www.pexels.com/video/3256542/",
    "living_room_loop.mp4":   "https://www.pexels.com/video/3769128/",
    "bedroom_loop.mp4":       "https://www.pexels.com/video/5543505/",
    "cafe_loop.mp4":          "https://www.pexels.com/video/3178591/",
    "rooftop_loop.mp4":       "https://www.pexels.com/video/3752930/",
    "red_fort_loop.mp4":      "https://www.pexels.com/video/3763876/",
    "parliament_loop.mp4":    "https://www.pexels.com/video/3763876/",
    "tech_park_loop.mp4":     "https://www.pexels.com/video/3130290/",
    "market_loop.mp4":        "https://www.pexels.com/video/2278095/",
    "dark_studio_loop.mp4":   "https://www.pexels.com/video/2795405/",
    "gradient_blue_loop.mp4": None,  # Generated procedurally below
    # Brand stages — use placeholder, replace with actual footage
    "google_stage_loop.mp4":  None,
    "apple_stage_loop.mp4":   None,
    "samsung_stage_loop.mp4": None,
    "ted_stage_loop.mp4":     None,
}

# CC0 ambient sound files
AMBIENT_SOUNDS = {
    "silence.wav":         None,  # Generated as silence
    "office_hum.wav":      "https://freesound.org/data/previews/263/263178_4486188-lq.mp3",
    "crowd_murmur.wav":    "https://freesound.org/data/previews/212/212371_2398403-lq.mp3",
    "waves_birds.wav":     "https://freesound.org/data/previews/212/212371_2398403-lq.mp3",
    "wind_cold.wav":       "https://freesound.org/data/previews/91/91763_1315829-lq.mp3",
    "wind_soft.wav":       "https://freesound.org/data/previews/91/91763_1315829-lq.mp3",
    "forest_birds.wav":    "https://freesound.org/data/previews/416/416013_2437358-lq.mp3",
    "cafe_ambient.wav":    "https://freesound.org/data/previews/263/263178_4486188-lq.mp3",
    "city_ambient.wav":    "https://freesound.org/data/previews/212/212371_2398403-lq.mp3",
    "market_crowd.wav":    "https://freesound.org/data/previews/212/212371_2398403-lq.mp3",
    "kitchen_ambient.wav": "https://freesound.org/data/previews/263/263178_4486188-lq.mp3",
    "studio_silence.wav":  None,
}


def generate_procedural_background(filename: str, dest: Path) -> bool:
    """Generate a simple gradient or solid color background video using FFmpeg."""
    import subprocess

    color_map = {
        "gradient_blue_loop.mp4": "0x0d1b2a",
        "google_stage_loop.mp4":  "0x1a73e8",
        "apple_stage_loop.mp4":   "0x1d1d1f",
        "samsung_stage_loop.mp4": "0x1428a0",
        "ted_stage_loop.mp4":     "0xeb0028",
    }
    color = color_map.get(filename, "0x1a1a2e")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:size=1920x1080:rate=25",
        "-t", "10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and dest.exists()
    except Exception as e:
        log.error(f"Procedural background failed: {e}")
        return False


def generate_silence(dest: Path, duration_sec: float = 10.0) -> bool:
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration_sec),
        "-c:a", "pcm_s16le",
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def main():
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    SFX_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Setting up background library...")
    log.info("NOTE: Pexels requires manual download. Generating placeholders for missing files.")
    log.info("      Replace placeholder files with real CC0 footage from pexels.com")

    # Generate all backgrounds (as placeholders if URL not directly downloadable)
    for filename, url in BACKGROUND_VIDEOS.items():
        dest = BACKGROUNDS_DIR / filename
        if dest.exists():
            log.info(f"  ✓ {filename} (exists)")
            continue

        if url is None:
            # Generate procedurally
            if generate_procedural_background(filename, dest):
                log.info(f"  ✓ {filename} (generated)")
            else:
                log.warning(f"  ✗ {filename} (generation failed)")
        else:
            # Try direct download (may fail if Pexels requires auth)
            try:
                urllib.request.urlretrieve(url, str(dest))
                log.info(f"  ✓ {filename} (downloaded)")
            except Exception:
                # Generate procedural placeholder
                log.warning(f"  ⚠ {filename} — download failed, generating placeholder")
                generate_procedural_background(filename, dest)

    # Generate ambient sounds
    log.info("\nSetting up ambient sound library...")
    for filename, url in AMBIENT_SOUNDS.items():
        dest = SFX_DIR / filename
        if dest.exists():
            log.info(f"  ✓ {filename} (exists)")
            continue

        if url is None:
            if generate_silence(dest):
                log.info(f"  ✓ {filename} (silence generated)")
        else:
            try:
                urllib.request.urlretrieve(url, str(dest.with_suffix(".mp3")))
                # Convert MP3 → WAV
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(dest.with_suffix(".mp3")),
                    "-ar", "44100", "-ac", "1", str(dest),
                ], capture_output=True, timeout=30)
                dest.with_suffix(".mp3").unlink(missing_ok=True)
                log.info(f"  ✓ {filename} (downloaded)")
            except Exception:
                generate_silence(dest)
                log.warning(f"  ⚠ {filename} — using silence placeholder")

    log.info("\nBackground library setup complete.")
    log.info(f"Location: {BACKGROUNDS_DIR}")
    log.info("Replace placeholder files with real CC0 footage from pexels.com for best quality.")


if __name__ == "__main__":
    main()
