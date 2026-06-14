"""
Generate all default South Indian avatar images.
Run once after setup. Re-run with --force to regenerate.
"""
import sys
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
log = logging.getLogger("generate_avatars")

from config import AVATARS
from pipeline.avatar_engine import generate_all_avatars, list_available_avatars


def main():
    parser = argparse.ArgumentParser(description="Generate avatar images")
    parser.add_argument("--force", action="store_true", help="Regenerate even if files exist")
    parser.add_argument("--persona", type=str, help="Generate only this persona ID")
    args = parser.parse_args()

    if args.persona and args.persona not in AVATARS:
        log.error(f"Unknown persona: {args.persona}. Available: {list(AVATARS.keys())}")
        sys.exit(1)

    log.info("Starting avatar generation...")
    log.info(f"Personas: {len(AVATARS)} | Force: {args.force}")

    results = generate_all_avatars(force=args.force)

    total = sum(len(paths) for paths in results.values())
    log.info(f"\n{'='*50}")
    log.info(f"Avatar generation complete: {total} images")
    for persona_id, paths in results.items():
        name = AVATARS[persona_id]["name"]
        log.info(f"  {name} ({persona_id}): {len(paths)} images")
        for p in paths:
            log.info(f"    → {p.name}")

    # Show all available
    available = list_available_avatars()
    log.info(f"\nAll available avatars:")
    for persona_id, items in available.items():
        log.info(f"  {persona_id}: {len(items)} variants")


if __name__ == "__main__":
    main()
