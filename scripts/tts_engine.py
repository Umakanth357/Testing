"""
tts_engine.py
Generates audio from tagged script segments.
Primary EN  : Chatterbox TTS (MIT, emotion exaggeration dial)
Primary TE/IN: Svara TTS v1 (Apache 2.0, 19 Indic languages)
Fallback EN : Coqui TTS (Mozilla Public License)
Fallback TE : AI4Bharat Indic-TTS (MIT)
"""
import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "models" / "weights"

# ── Voice profile definitions ─────────────────────────────────────
VOICE_PROFILES = {
    # Telugu
    "te_female": {"lang": "te", "gender": "female", "engine": "svara", "voice_id": "te_female"},
    "te_male":   {"lang": "te", "gender": "male",   "engine": "svara", "voice_id": "te_male"},
    # Kannada
    "kn_female": {"lang": "kn", "gender": "female", "engine": "svara", "voice_id": "kn_female"},
    "kn_male":   {"lang": "kn", "gender": "male",   "engine": "svara", "voice_id": "kn_male"},
    # Tamil
    "ta_female": {"lang": "ta", "gender": "female", "engine": "svara", "voice_id": "ta_female"},
    "ta_male":   {"lang": "ta", "gender": "male",   "engine": "svara", "voice_id": "ta_male"},
    # Hindi
    "hi_female": {"lang": "hi", "gender": "female", "engine": "svara", "voice_id": "hi_female"},
    "hi_male":   {"lang": "hi", "gender": "male",   "engine": "svara", "voice_id": "hi_male"},
    # English
    "en_female": {"lang": "en", "gender": "female", "engine": "chatterbox", "voice_id": "en_female"},
    "en_male":   {"lang": "en", "gender": "male",   "engine": "chatterbox", "voice_id": "en_male"},
}


class SvaraTTS:
    """Svara TTS — 19 Indian languages, emotion tags, MIT license."""

    def __init__(self):
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            # Try pip-installed svara package
            from svara import SvaraTTS as SvaraModel
            model_path = WEIGHTS_DIR / "svara-tts-v1.Q4_K_M.gguf"
            if model_path.exists():
                self._model = SvaraModel(str(model_path))
            else:
                self._model = SvaraModel()  # downloads automatically
            self._available = True
            logger.info("Svara TTS loaded successfully")
        except ImportError:
            # Try HuggingFace inference endpoint
            try:
                import urllib.request
                import json
                # Quick ping test
                req = urllib.request.Request(
                    "https://api-inference.huggingface.co/models/kenpath/svara-tts-v1",
                    headers={"Content-Type": "application/json"},
                    method="GET"
                )
                urllib.request.urlopen(req, timeout=5)
                self._available = True
                self._model = "hf_api"
                logger.info("Svara TTS using HuggingFace API")
            except Exception:
                logger.warning("Svara TTS unavailable — will use AI4Bharat fallback")

    def is_available(self) -> bool:
        return self._available

    def synthesize(self, text: str, voice_id: str, emotion_tag: str, output_path: str) -> bool:
        if not self._available:
            return False
        try:
            if self._model == "hf_api":
                return self._hf_synthesize(text, voice_id, emotion_tag, output_path)

            tagged_text = f"{text} {emotion_tag}" if emotion_tag else text
            self._model.synthesize(
                text=tagged_text,
                voice_id=voice_id,
                output_path=output_path
            )
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
        except Exception as e:
            logger.error(f"Svara synthesis error: {e}")
            return False

    def _hf_synthesize(self, text: str, voice_id: str, emotion_tag: str, output_path: str) -> bool:
        """Fallback: HuggingFace Inference API for Svara."""
        try:
            import urllib.request, json
            tagged_text = f"{text} {emotion_tag}" if emotion_tag else text
            payload = json.dumps({
                "inputs": tagged_text,
                "parameters": {"voice_id": voice_id}
            }).encode()
            req = urllib.request.Request(
                "https://api-inference.huggingface.co/models/kenpath/svara-tts-v1",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                audio_bytes = resp.read()
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            return len(audio_bytes) > 1000
        except Exception as e:
            logger.error(f"Svara HF API error: {e}")
            return False


class ChatterboxTTS:
    """Chatterbox TTS — English, emotion exaggeration dial 0-1, MIT."""

    def __init__(self):
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = CBModel.from_pretrained(device=device)
            self._available = True
            logger.info(f"Chatterbox TTS loaded on {device}")
        except Exception as e:
            logger.warning(f"Chatterbox TTS unavailable: {e}")

    def is_available(self) -> bool:
        return self._available

    def synthesize(self, text: str, exaggeration: float, output_path: str) -> bool:
        if not self._available:
            return False
        try:
            import torchaudio
            wav = self._model.generate(text, exaggeration=exaggeration)
            torchaudio.save(output_path, wav, self._model.sr)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
        except Exception as e:
            logger.error(f"Chatterbox synthesis error: {e}")
            return False


class CoquiTTS:
    """Coqui TTS — fallback for English when Chatterbox unavailable."""

    def __init__(self):
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from TTS.api import TTS
            self._model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            self._available = True
            logger.info("Coqui TTS loaded as fallback")
        except Exception as e:
            logger.warning(f"Coqui TTS unavailable: {e}")

    def is_available(self) -> bool:
        return self._available

    def synthesize(self, text: str, output_path: str) -> bool:
        if not self._available:
            return False
        try:
            self._model.tts_to_file(text=text, file_path=output_path)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
        except Exception as e:
            logger.error(f"Coqui TTS error: {e}")
            return False


class TTSEngine:
    """
    Unified TTS engine with automatic fallback chain.
    Usage:
        engine = TTSEngine()
        audio_path = engine.synthesize_tagged_segments(tagged_segments, "te_female", "/output/audio.wav")
    """

    def __init__(self):
        logger.info("Initialising TTS engines...")
        self.svara     = SvaraTTS()
        self.chatterbox = ChatterboxTTS()
        self.coqui     = CoquiTTS()
        self._log_availability()

    def _log_availability(self):
        avail = []
        if self.svara.is_available():     avail.append("Svara (Indic)")
        if self.chatterbox.is_available(): avail.append("Chatterbox (EN)")
        if self.coqui.is_available():     avail.append("Coqui (EN fallback)")
        if not avail:
            logger.error("NO TTS ENGINES AVAILABLE — check installation")
        else:
            logger.info(f"TTS engines available: {', '.join(avail)}")

    def synthesize_tagged_segments(
        self,
        tagged_segments: List[Dict],
        voice_profile: str,
        output_path: str
    ) -> Optional[str]:
        """
        Synthesizes each tagged segment separately, then concatenates.
        Returns path to final WAV file or None on failure.
        """
        profile = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["en_female"])
        lang    = profile["lang"]
        engine  = profile["engine"]
        voice_id = profile["voice_id"]

        segment_files = []
        tmpdir = tempfile.mkdtemp(prefix="tts_segments_")

        for i, seg in enumerate(tagged_segments):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.wav")
            text     = seg["sentence"]
            emotion  = seg.get("svara_tag", "")
            exagg    = seg.get("chatterbox_exaggeration", 0.4)

            success = False

            # ── Indic languages → Svara ──────────────────────
            if lang != "en":
                if self.svara.is_available():
                    success = self.svara.synthesize(text, voice_id, emotion, seg_path)
                if not success:
                    # AI4Bharat fallback for Indic
                    success = self._ai4bharat_fallback(text, lang, seg_path)
                if not success:
                    # Last resort: gTTS (requires internet)
                    success = self._gtts_fallback(text, lang, seg_path)

            # ── English → Chatterbox ─────────────────────────
            else:
                if self.chatterbox.is_available():
                    success = self.chatterbox.synthesize(text, exagg, seg_path)
                if not success and self.coqui.is_available():
                    success = self.coqui.synthesize(text, seg_path)
                if not success:
                    success = self._gtts_fallback(text, "en", seg_path)

            if success and os.path.exists(seg_path):
                segment_files.append(seg_path)
            else:
                logger.warning(f"Failed to synthesize segment {i}: {text[:50]}")

        if not segment_files:
            logger.error("No audio segments generated")
            return None

        # ── Concatenate segments with ffmpeg ──────────────────
        return self._concat_segments(segment_files, output_path, tmpdir)

    def _concat_segments(self, segment_files: List[str], output_path: str, tmpdir: str) -> Optional[str]:
        """Concatenate WAV segments using ffmpeg with natural breath between them."""
        try:
            # Create silence file (50ms breath pause between sentences)
            silence_path = os.path.join(tmpdir, "silence.wav")
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
                "-t", "0.05", "-q:a", "9", silence_path, "-y", "-loglevel", "error"
            ], check=True)

            # Build file list with pauses
            list_path = os.path.join(tmpdir, "concat_list.txt")
            with open(list_path, "w") as f:
                for i, seg in enumerate(segment_files):
                    f.write(f"file '{seg}'\n")
                    if i < len(segment_files) - 1:
                        f.write(f"file '{silence_path}'\n")

            # Concatenate
            subprocess.run([
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_path,
                "-ar", "22050", "-ac", "1", output_path, "-y", "-loglevel", "error"
            ], check=True)

            # Add subtle room reverb (makes voice sound more natural / less studio-clean)
            reverb_path = output_path.replace(".wav", "_reverb.wav")
            subprocess.run([
                "ffmpeg", "-i", output_path,
                "-af", "aecho=0.8:0.9:40:0.03",
                reverb_path, "-y", "-loglevel", "error"
            ], check=True)

            if os.path.exists(reverb_path) and os.path.getsize(reverb_path) > 1000:
                os.replace(reverb_path, output_path)

            logger.info(f"Audio synthesized: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio concat failed: {e}")
            return segment_files[0] if segment_files else None

    def _ai4bharat_fallback(self, text: str, lang: str, output_path: str) -> bool:
        """AI4Bharat Indic-TTS via HuggingFace API — free, MIT."""
        try:
            import urllib.request, json
            payload = json.dumps({
                "inputs": text,
                "parameters": {"language": lang}
            }).encode()
            req = urllib.request.Request(
                f"https://api-inference.huggingface.co/models/ai4bharat/indic-tts-{lang}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                audio = resp.read()
            if len(audio) > 1000:
                with open(output_path, "wb") as f:
                    f.write(audio)
                return True
        except Exception as e:
            logger.debug(f"AI4Bharat fallback failed: {e}")
        return False

    def _gtts_fallback(self, text: str, lang: str, output_path: str) -> bool:
        """gTTS — last resort, requires internet."""
        try:
            from gtts import gTTS
            tmp_mp3 = output_path.replace(".wav", "_gtts.mp3")
            gTTS(text=text, lang=lang, slow=False).save(tmp_mp3)
            subprocess.run([
                "ffmpeg", "-i", tmp_mp3, "-ar", "22050", "-ac", "1",
                output_path, "-y", "-loglevel", "error"
            ], check=True)
            return os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"gTTS fallback failed: {e}")
        return False


if __name__ == "__main__":
    # Quick self-test with mock tagged segments
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from emotion_tagger import tag_script

    test_script = "Welcome everyone! Today is an incredible day. Please pay attention to this critical point."
    print("Testing TTS engine...\n")
    tagged = tag_script(test_script)
    engine = TTSEngine()
    out = Path("/tmp/tts_test.wav")
    result = engine.synthesize_tagged_segments(tagged, "en_female", str(out))
    if result and Path(result).exists():
        size = Path(result).stat().st_size
        print(f"✓ TTS test passed — output: {result} ({size:,} bytes)")
    else:
        print("✗ TTS test failed")
