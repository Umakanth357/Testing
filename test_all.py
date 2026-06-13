#!/usr/bin/env python3
"""
test_all.py — Full component test suite
Run BEFORE your first real video generation to verify everything works.
Usage: python test_all.py
"""
import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

PASS  = "✅"
FAIL  = "❌"
WARN  = "⚠️ "
results = {}

def test(name, fn):
    """Run a test function and record result."""
    print(f"\n  Testing {name}...", end="", flush=True)
    t0 = time.time()
    try:
        ok, msg = fn()
        elapsed = time.time() - t0
        icon = PASS if ok else FAIL
        print(f" {icon} {msg} ({elapsed:.1f}s)")
        results[name] = {"pass": ok, "msg": msg, "time": round(elapsed, 1)}
        return ok
    except Exception as e:
        elapsed = time.time() - t0
        print(f" {FAIL} Exception: {e} ({elapsed:.1f}s)")
        results[name] = {"pass": False, "msg": str(e)[:100], "time": round(elapsed, 1)}
        return False


# ── 1: Python environment ─────────────────────────────────────────
def test_python():
    v = sys.version_info
    return v >= (3, 10), f"Python {v.major}.{v.minor}.{v.micro}"

# ── 2: ffmpeg ─────────────────────────────────────────────────────
def test_ffmpeg():
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    version = r.stdout.split("\n")[0][:50] if r.returncode == 0 else ""
    return r.returncode == 0, version or "ffmpeg not found"

# ── 3: GPU / CUDA ─────────────────────────────────────────────────
def test_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // 1_000_000
            return True, f"{gpu} — {vram} MB VRAM"
        return False, "No CUDA GPU — CPU mode (slow but functional)"
    except ImportError:
        return False, "torch not installed"

# ── 4: Emotion tagger (Ollama or fallback) ────────────────────────
def test_emotion_tagger():
    from emotion_tagger import tag_script
    test = "This is incredible news! However we have a critical issue."
    tagged = tag_script(test)
    if len(tagged) >= 2:
        emotions = [t["emotion"] for t in tagged]
        return True, f"Tagged {len(tagged)} sentences: {emotions}"
    return False, "Emotion tagging returned no results"

# ── 5: Ollama ────────────────────────────────────────────────────
def test_ollama():
    import urllib.request
    try:
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return True, f"Models available: {models[:3]}"
    except Exception as e:
        return False, f"Ollama not reachable: {e}"

# ── 6: Svara TTS ─────────────────────────────────────────────────
def test_svara_tts():
    try:
        import svara
        return True, "svara package importable"
    except ImportError:
        # Try HF API ping
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api-inference.huggingface.co/models/kenpath/svara-tts-v1",
                method="GET"
            )
            urllib.request.urlopen(req, timeout=5)
            return True, "svara not installed but HuggingFace API reachable"
        except Exception:
            return False, "svara not installed and HuggingFace not reachable"

# ── 7: Chatterbox TTS ────────────────────────────────────────────
def test_chatterbox():
    try:
        from chatterbox.tts import ChatterboxTTS
        return True, "chatterbox importable"
    except ImportError:
        return False, "chatterbox not installed — will use Coqui fallback"

# ── 8: Coqui TTS fallback ────────────────────────────────────────
def test_coqui():
    try:
        from TTS.api import TTS
        return True, "Coqui TTS available as EN fallback"
    except ImportError:
        return False, "Coqui TTS not installed"

# ── 9: Whisper ───────────────────────────────────────────────────
def test_whisper():
    try:
        import whisper
        return True, f"whisper {whisper.__version__ if hasattr(whisper, '__version__') else 'available'}"
    except ImportError:
        return False, "whisper not installed"

# ── 10: MuseTalk repo ────────────────────────────────────────────
def test_musetalk():
    path = ROOT / "models" / "MuseTalk"
    weights = ROOT / "models" / "weights" / "musetalk" / "pytorch_model.bin"
    if not path.exists():
        return False, "MuseTalk repo not found — run setup.sh"
    if not weights.exists():
        return False, f"MuseTalk weights missing: {weights}"
    return True, "Repo and weights present"

# ── 11: LivePortrait repo ────────────────────────────────────────
def test_liveportrait():
    path = ROOT / "models" / "LivePortrait"
    if not path.exists():
        return False, "LivePortrait repo not found — run setup.sh"
    # Check key weight files
    weight_dir = ROOT / "models" / "weights" / "liveportrait"
    n_found = len(list(weight_dir.glob("*.safetensors"))) if weight_dir.exists() else 0
    if n_found < 3:
        return False, f"Only {n_found}/5 LivePortrait weights found"
    return True, f"{n_found} weight files found"

# ── 12: SadTalker (fallback) ─────────────────────────────────────
def test_sadtalker():
    path = ROOT / "models" / "SadTalker"
    return path.exists(), "SadTalker repo present" if path.exists() else "SadTalker not found (fallback only)"

# ── 13: GFPGAN weights ───────────────────────────────────────────
def test_gfpgan():
    weight = ROOT / "models" / "weights" / "GFPGANv1.4.pth"
    try:
        from gfpgan import GFPGANer
        pkg = True
    except ImportError:
        pkg = False
    if not weight.exists():
        return False, "GFPGAN weights missing"
    return pkg, f"Weights: {weight.stat().st_size//1_000_000}MB {'| pkg OK' if pkg else '| pkg missing'}"

# ── 14: yt-dlp ───────────────────────────────────────────────────
def test_ytdlp():
    r = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip() or "yt-dlp not found"

# ── 15: Avatars ──────────────────────────────────────────────────
def test_avatars():
    avatars = list((ROOT / "avatars").glob("*.png"))
    if not avatars:
        return False, "No avatar PNG files found — run: python scripts/generate_avatars.py"
    return True, f"{len(avatars)} avatar(s) found: {[f.name for f in avatars[:3]]}"

# ── 16: Full TTS smoke test ──────────────────────────────────────
def test_tts_smoke():
    from tts_engine import TTSEngine
    from emotion_tagger import tag_script
    engine = TTSEngine()
    tagged = tag_script("Hello, this is a test.")
    if not tagged:
        return False, "No tagged segments"
    tmpout = tempfile.mktemp(suffix=".wav")
    result = engine.synthesize_tagged_segments(tagged, "en_female", tmpout)
    if result and Path(tmpout).exists() and Path(tmpout).stat().st_size > 1000:
        size = Path(tmpout).stat().st_size
        os.remove(tmpout)
        return True, f"Audio generated: {size:,} bytes"
    return False, "TTS produced no output — check TTS engine logs"

# ── 17: Disk space ───────────────────────────────────────────────
def test_disk():
    stat = shutil.disk_usage(str(ROOT))
    free_gb = stat.free / 1e9
    total_gb = stat.total / 1e9
    ok = free_gb > 20  # Need at least 20GB free
    return ok, f"{free_gb:.0f} GB free / {total_gb:.0f} GB total"

# ── 18: Content ingester smoke test ──────────────────────────────
def test_ingester():
    from content_ingester import ingest_text
    result = ingest_text(
        "MuseTalk generates realistic lip sync at 30fps using diffusion models.",
        language="en", tone="professional", content_type="tool_review"
    )
    script = result.get("script", "")
    if script and len(script) > 50:
        return True, f"Script generated: {len(script.split())} words"
    return False, "Script generation returned empty — check Ollama"


# ── Run all tests ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AVATAR TOOL — FULL COMPONENT TEST SUITE")
    print("=" * 60)

    print("\n🔧 Environment & Infrastructure")
    test("Python 3.10+",        test_python)
    test("ffmpeg",              test_ffmpeg)
    test("GPU / CUDA",          test_gpu)
    test("Disk space",          test_disk)

    print("\n🧠 AI Models")
    test("Ollama (local LLM)",  test_ollama)
    test("Emotion tagger",      test_emotion_tagger)
    test("Whisper (STT)",       test_whisper)
    test("yt-dlp (YouTube)",    test_ytdlp)

    print("\n🎙️ TTS Engines")
    test("Svara TTS (Telugu)",  test_svara_tts)
    test("Chatterbox (EN)",     test_chatterbox)
    test("Coqui (EN fallback)", test_coqui)
    test("TTS smoke test",      test_tts_smoke)

    print("\n🎬 Video Models")
    test("MuseTalk (lip sync)",  test_musetalk)
    test("LivePortrait (anim)",  test_liveportrait)
    test("SadTalker (fallback)", test_sadtalker)
    test("GFPGAN (enhance)",     test_gfpgan)

    print("\n🎭 Assets")
    test("Avatar images",        test_avatars)

    print("\n🔗 Integration")
    test("Content ingester",     test_ingester)

    # ── Summary ───────────────────────────────────────────────────
    total  = len(results)
    passed = sum(1 for r in results.values() if r["pass"])
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed", end="")
    if failed > 0:
        critical_fails = [n for n, r in results.items() if not r["pass"] and
                         n in ["Python 3.10+", "ffmpeg", "Emotion tagger", "TTS smoke test", "Avatar images"]]
        if critical_fails:
            print(f"\n  ❌ CRITICAL FAILURES: {critical_fails}")
            print("  → Fix these before running the tool")
        else:
            print(f"\n  ⚠️  {failed} non-critical issues (tool will still work with fallbacks)")
    else:
        print("\n  ✅ All tests passed — tool is ready!")
    print("=" * 60)

    # Save results
    results_path = ROOT / "logs" / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved: {results_path}\n")

    sys.exit(0 if failed == 0 else 1)
