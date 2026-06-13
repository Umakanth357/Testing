"""
app.py — Avatar Tool Main Application
Run: python app.py
Access: http://localhost:7860
"""
import os
import sys
import uuid
import json
import logging
import threading
import time
from pathlib import Path
from datetime import datetime

# Add scripts to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

# Load .env file if present (GROQ_API_KEY, HF_TOKEN etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import gradio as gr
from emotion_tagger import tag_script, build_tagged_script
from tts_engine import TTSEngine, VOICE_PROFILES
from video_pipeline import VideoPipeline
from content_ingester import ingest_youtube, ingest_text

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "app.log")
    ]
)
logger = logging.getLogger("avatar_tool")

OUTPUTS = ROOT / "outputs"
AVATARS = ROOT / "avatars"
OUTPUTS.mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

# ── Initialise engines (done once at startup) ────────────────────
logger.info("Initialising engines...")
tts_engine = TTSEngine()
vid_pipeline = VideoPipeline()
logger.info("Engines ready.")


def get_avatar_list():
    """Return list of available avatar images."""
    default_avatars = {
        "South Indian Male (30s)":   "si_male_30.png",
        "South Indian Female (30s)": "si_female_30.png",
        "South Indian Male (40s)":   "si_male_40.png",
        "South Indian Female (40s)": "si_female_40.png",
        "Professional Male":         "pro_male.png",
        "Professional Female":       "pro_female.png",
    }
    available = {}
    for label, fname in default_avatars.items():
        path = AVATARS / fname
        if path.exists():
            available[label] = str(path)
    if not available:
        # No avatars generated yet — return placeholder message
        available["⚠ Run setup.sh to generate avatars"] = ""
    return available


def generate_video_job(
    source_type,       # "script" | "youtube" | "text"
    script_text,       # used when source_type == "script"
    youtube_url,       # used when source_type == "youtube"
    raw_text,          # used when source_type == "text"
    content_type,      # KT / tool_review / news / movie_review / event_summary / notification
    voice_profile,     # e.g. "te_female"
    avatar_choice,     # display name from dropdown
    background,        # "office" / "studio" / "gradient"
    language,          # "en" / "te" / "kn" / "ta" / "hi"
    tone,              # "professional" / "friendly" / "excited" / "serious" / "reviewer"
    target_duration,   # int minutes
    add_captions,      # bool
    progress=gr.Progress()
) -> tuple:
    """
    Main generation function. Returns (status_text, script_preview, video_path, script_for_download)
    """
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] New job: {source_type} | {voice_profile} | {content_type}")

    try:
        # ── Step 1: Get / generate script ─────────────────────
        progress(0.05, desc="📝 Preparing script...")

        if source_type == "youtube":
            if not youtube_url or not youtube_url.startswith("http"):
                return "❌ Please enter a valid YouTube URL", "", None, ""
            progress(0.10, desc="🎥 Fetching YouTube content...")
            result = ingest_youtube(
                url=youtube_url,
                language=language,
                tone=tone,
                target_duration_mins=int(target_duration),
                content_type=content_type
            )
            if "error" in result:
                return f"❌ {result['error']}", "", None, ""
            final_script = result["script"]

        elif source_type == "text":
            if not raw_text or len(raw_text.strip()) < 20:
                return "❌ Please enter at least 20 characters of content", "", None, ""
            progress(0.10, desc="📄 Processing text content...")
            result = ingest_text(
                text=raw_text,
                language=language,
                tone=tone,
                target_duration_mins=int(target_duration),
                content_type=content_type
            )
            final_script = result["script"]

        else:  # direct script
            if not script_text or len(script_text.strip()) < 20:
                return "❌ Please enter a script (minimum 20 characters)", "", None, ""
            final_script = script_text.strip()

        if not final_script:
            return "❌ Script generation failed. Please check logs.", "", None, ""

        word_count = len(final_script.split())
        logger.info(f"[{job_id}] Script ready: {word_count} words")

        # ── Step 2: Emotion tagging ────────────────────────────
        progress(0.20, desc="🎭 Detecting emotions in script...")
        tagged_segments = tag_script(final_script)
        emotion_summary = {}
        for seg in tagged_segments:
            e = seg["emotion"]
            emotion_summary[e] = emotion_summary.get(e, 0) + 1
        logger.info(f"[{job_id}] Emotions detected: {emotion_summary}")

        # ── Step 3: TTS audio generation ──────────────────────
        progress(0.35, desc=f"🎙️ Generating {voice_profile} voice...")
        audio_path = str(OUTPUTS / f"{job_id}_audio.wav")
        audio_result = tts_engine.synthesize_tagged_segments(
            tagged_segments=tagged_segments,
            voice_profile=voice_profile,
            output_path=audio_path
        )
        if not audio_result or not Path(audio_path).exists():
            return "❌ Audio generation failed. Check TTS engine logs.", final_script, None, final_script

        audio_size = Path(audio_path).stat().st_size
        logger.info(f"[{job_id}] Audio generated: {audio_size:,} bytes")

        # ── Step 4: Get avatar image ───────────────────────────
        progress(0.45, desc="👤 Loading avatar...")
        avatar_map = get_avatar_list()
        avatar_img = avatar_map.get(avatar_choice, "")
        if not avatar_img or not Path(avatar_img).exists():
            # Try first available avatar
            for k, v in avatar_map.items():
                if v and Path(v).exists():
                    avatar_img = v
                    break
        if not avatar_img:
            return "❌ No avatar images found. Run setup.sh to generate avatars.", final_script, None, final_script

        # ── Step 5: Video generation ───────────────────────────
        progress(0.55, desc="🎬 Animating avatar (LivePortrait + MuseTalk)...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUTS / f"video_{timestamp}_{job_id}.mp4")

        video_result = vid_pipeline.generate(
            avatar_image=avatar_img,
            audio_path=audio_path,
            output_path=output_path,
            background=background.lower(),
            add_captions=add_captions,
            logo_path=str(ROOT / "web" / "assets" / "logo.png") if (ROOT / "web" / "assets" / "logo.png").exists() else None,
            job_id=job_id
        )

        progress(0.90, desc="✨ Enhancing with GFPGAN...")
        time.sleep(0.5)  # Let progress update visually

        if not video_result or not Path(video_result).exists():
            return (
                "⚠️ Video generation had issues but audio was created. Check logs.",
                final_script,
                audio_path,  # Return audio at least
                final_script
            )

        video_size_mb = Path(video_result).stat().st_size / 1_000_000
        progress(1.0, desc="✅ Done!")

        # Clean up temp audio
        try:
            os.remove(audio_path)
        except Exception:
            pass

        status = (
            f"✅ Video generated successfully!\n"
            f"📊 Script: {word_count} words | "
            f"🎭 Emotions detected: {len(emotion_summary)} types | "
            f"📁 Size: {video_size_mb:.1f} MB\n"
            f"🎯 Job ID: {job_id}"
        )

        logger.info(f"[{job_id}] ✓ Complete: {video_result} ({video_size_mb:.1f} MB)")
        return status, final_script, video_result, final_script

    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}\n\nJob ID: {job_id}", "", None, ""


# ── Gradio UI ─────────────────────────────────────────────────────

def build_ui():
    avatar_choices = list(get_avatar_list().keys())

    with gr.Blocks(
        title="🎭 Avatar Video Generator",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="teal",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        ),
        css="""
        .header { text-align: center; padding: 20px 0 10px; }
        .section-title { font-weight: 600; color: #6366f1; margin-bottom: 4px; }
        .status-box { font-family: monospace; font-size: 13px; }
        footer { display: none !important; }
        """
    ) as app:

        gr.HTML("""
        <div class="header">
            <h1>🎭 Avatar Video Generator</h1>
            <p style="color:#666">Paste a script, drop a YouTube URL, or type any content → Get a realistic South Indian avatar video</p>
        </div>
        """)

        with gr.Tabs():

            # ══ TAB 1: GENERATE ════════════════════════════════════
            with gr.TabItem("🎬 Generate Video"):
                with gr.Row():
                    # ── LEFT: Input ──────────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Content Input")
                        source_type = gr.Radio(
                            ["script", "youtube", "text"],
                            value="script",
                            label="Source type",
                            info="Direct script / YouTube URL / Paste any text"
                        )

                        with gr.Group(visible=True) as script_group:
                            script_input = gr.Textbox(
                                label="Your Script",
                                placeholder="Type or paste your narration script here...\nThe AI will detect emotions automatically from the words you write.",
                                lines=8,
                                max_lines=20
                            )

                        with gr.Group(visible=False) as youtube_group:
                            yt_url = gr.Textbox(
                                label="YouTube URL",
                                placeholder="https://www.youtube.com/watch?v=...",
                            )
                            gr.HTML("<small style='color:#888'>The tool will transcribe, summarise, and rewrite in your voice. Content is transformed — not copied.</small>")

                        with gr.Group(visible=False) as text_group:
                            raw_text = gr.Textbox(
                                label="Paste article / notes / review",
                                placeholder="Paste any content — news article, tool review, movie review, event notes...",
                                lines=8
                            )

                        content_type = gr.Dropdown(
                            choices=["KT", "tool_review", "news", "movie_review", "event_summary", "notification"],
                            value="KT",
                            label="Content type",
                            info="Shapes the script structure and tone"
                        )
                        tone = gr.Dropdown(
                            choices=["professional", "friendly", "excited", "serious", "reviewer"],
                            value="professional",
                            label="Tone / delivery style"
                        )
                        target_duration = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Target duration (minutes)"
                        )

                    # ── RIGHT: Avatar & voice settings ──────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎙️ Voice & Avatar")
                        language = gr.Dropdown(
                            choices=["en", "te", "kn", "ta", "hi"],
                            value="en",
                            label="Language",
                            info="en=English | te=Telugu | kn=Kannada | ta=Tamil | hi=Hindi"
                        )
                        voice_profile = gr.Dropdown(
                            choices=list(VOICE_PROFILES.keys()),
                            value="en_female",
                            label="Voice profile",
                            info="Select gender + language combination"
                        )
                        avatar_choice = gr.Dropdown(
                            choices=avatar_choices,
                            value=avatar_choices[0] if avatar_choices else None,
                            label="Avatar",
                            info="South Indian avatars generated at setup"
                        )
                        background = gr.Dropdown(
                            choices=["Office", "Studio", "Gradient"],
                            value="Office",
                            label="Background"
                        )
                        add_captions = gr.Checkbox(
                            value=True,
                            label="Auto-generate captions (Whisper)",
                            info="Recommended — adds subtitles automatically"
                        )

                        gr.HTML("<hr style='margin:10px 0'>")
                        gr.Markdown("### 🎭 Emotion detection")
                        gr.HTML("""
                        <div style='background:#f0f4ff;border-radius:8px;padding:12px;font-size:13px;color:#444'>
                        <b>Automatic per-sentence:</b><br>
                        Each sentence gets its own emotion tag.<br>
                        Voice exaggeration, pace, pitch, and avatar expression all change per sentence.<br><br>
                        <b>Example:</b><br>
                        "Today is incredible!" → <span style='color:#7c3aed'>excited (fast, +2st pitch)</span><br>
                        "This is a critical issue." → <span style='color:#dc2626'>urgent (slow, tense)</span><br>
                        "Thank you all!" → <span style='color:#059669'>warm (medium, friendly)</span>
                        </div>
                        """)

                # ── Generate button ──────────────────────────────────
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate Avatar Video",
                        variant="primary",
                        size="lg",
                        scale=2
                    )

                # ── Outputs ──────────────────────────────────────────
                with gr.Row():
                    with gr.Column(scale=1):
                        status_out = gr.Textbox(
                            label="Status", lines=4, interactive=False,
                            elem_classes=["status-box"]
                        )
                        script_preview = gr.Textbox(
                            label="Generated script (review before generating again)",
                            lines=6, interactive=True,
                            info="You can edit this script and run again with 'Direct script' mode"
                        )
                    with gr.Column(scale=1):
                        video_out = gr.Video(label="Generated Video", height=400)
                        script_download = gr.File(label="Download script as .txt")

                # ── Visibility toggles ───────────────────────────────
                def toggle_source(src):
                    return (
                        gr.update(visible=(src == "script")),
                        gr.update(visible=(src == "youtube")),
                        gr.update(visible=(src == "text")),
                    )

                source_type.change(
                    toggle_source,
                    inputs=[source_type],
                    outputs=[script_group, youtube_group, text_group]
                )

                # Auto-update voice profile when language changes
                def suggest_voice(lang):
                    return f"{lang}_female"
                language.change(suggest_voice, inputs=[language], outputs=[voice_profile])

                # ── Save script to temp file for download ────────────
                def save_script_file(script_text):
                    if not script_text:
                        return None
                    path = str(OUTPUTS / f"script_{str(uuid.uuid4())[:8]}.txt")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(script_text)
                    return path

                # ── Wire generate button ─────────────────────────────
                def generate_and_save(
                    source_type, script_input, yt_url, raw_text,
                    content_type, voice_profile, avatar_choice, background,
                    language, tone, target_duration, add_captions,
                    progress=gr.Progress()
                ):
                    status, script, video, script_text = generate_video_job(
                        source_type, script_input, yt_url, raw_text,
                        content_type, voice_profile, avatar_choice, background,
                        language, tone, target_duration, add_captions,
                        progress=progress
                    )
                    script_file = save_script_file(script_text)
                    return status, script, video, script_file

                generate_btn.click(
                    fn=generate_and_save,
                    inputs=[
                        source_type, script_input, yt_url, raw_text,
                        content_type, voice_profile, avatar_choice, background,
                        language, tone, target_duration, add_captions
                    ],
                    outputs=[status_out, script_preview, video_out, script_download],
                    show_progress=True
                )

            # ══ TAB 2: VIDEO LIBRARY ═══════════════════════════════
            with gr.TabItem("📚 Video Library"):
                refresh_btn = gr.Button("🔄 Refresh Library", size="sm")
                library_display = gr.HTML(value=_render_library())

                refresh_btn.click(
                    fn=lambda: _render_library(),
                    outputs=[library_display]
                )

            # ══ TAB 3: HEALTH CHECK ════════════════════════════════
            with gr.TabItem("🔧 System Status"):
                health_btn = gr.Button("Run Health Check", variant="secondary")
                health_out = gr.JSON(label="Component status", value={})

                health_btn.click(fn=run_health_check, outputs=[health_out])

                gr.HTML("""
                <div style='margin-top:20px;padding:16px;background:#f9fafb;border-radius:8px;font-size:13px'>
                <b>What this tool needs to work:</b><br><br>
                <b>FREE (no cost):</b><br>
                ✓ Ollama + Gemma3 4B — emotion tagging (fully local)<br>
                ✓ Svara TTS — Telugu/Indic voice (Apache 2.0)<br>
                ✓ Chatterbox TTS — English voice (MIT)<br>
                ✓ MuseTalk — lip sync (MIT)<br>
                ✓ LivePortrait — expressions (MIT)<br>
                ✓ GFPGAN — face upscale (Apache 2.0)<br>
                ✓ Whisper — transcription + captions (MIT)<br>
                ✓ yt-dlp — YouTube audio (Unlicense)<br><br>
                <b>Paid (EC2 server only):</b><br>
                ⚡ AWS EC2 g4dn.2xlarge — ~$0.22/hr spot (~$50/month if run 8hrs/day)<br>
                </div>
                """)

    return app


def _render_library() -> str:
    """Render video library as HTML table."""
    videos = sorted(OUTPUTS.glob("video_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not videos:
        return "<p style='color:#888;padding:20px'>No videos generated yet. Use the Generate tab to create your first video.</p>"

    rows = ""
    for v in videos[:20]:  # Show latest 20
        size_mb = v.stat().st_size / 1_000_000
        mtime = datetime.fromtimestamp(v.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        rows += f"""
        <tr>
            <td style='padding:8px'>{v.stem}</td>
            <td style='padding:8px'>{mtime}</td>
            <td style='padding:8px'>{size_mb:.1f} MB</td>
            <td style='padding:8px'><a href='/outputs/{v.name}' download>⬇ Download</a></td>
        </tr>"""

    return f"""
    <table style='width:100%;border-collapse:collapse;font-size:13px'>
        <thead><tr style='background:#f3f4f6'>
            <th style='padding:8px;text-align:left'>File</th>
            <th style='padding:8px;text-align:left'>Created</th>
            <th style='padding:8px;text-align:left'>Size</th>
            <th style='padding:8px;text-align:left'>Action</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def run_health_check() -> dict:
    """Check all components and return status dict."""
    status = {}

    # Ollama
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        status["ollama"] = "✅ Running"
    except Exception:
        status["ollama"] = "❌ Not running — run: ollama serve"

    # MuseTalk
    status["musetalk"] = "✅ Found" if (ROOT / "models" / "MuseTalk").exists() else "❌ Not found — run setup.sh"

    # LivePortrait
    status["liveportrait"] = "✅ Found" if (ROOT / "models" / "LivePortrait").exists() else "❌ Not found"

    # GFPGAN weights
    status["gfpgan_weights"] = "✅ Found" if (ROOT / "models" / "weights" / "GFPGANv1.4.pth").exists() else "⚠️ Missing — will skip upscale"

    # Svara TTS
    try:
        import svara
        status["svara_tts"] = "✅ Available"
    except ImportError:
        status["svara_tts"] = "⚠️ Not installed — Telugu TTS will use HF API"

    # Chatterbox
    try:
        from chatterbox.tts import ChatterboxTTS
        status["chatterbox_tts"] = "✅ Available"
    except ImportError:
        status["chatterbox_tts"] = "⚠️ Not installed — English TTS will use Coqui"

    # Whisper
    try:
        import whisper
        status["whisper"] = "✅ Available"
    except ImportError:
        status["whisper"] = "❌ Not installed"

    # ffmpeg
    try:
        import subprocess
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        status["ffmpeg"] = "✅ Available"
    except Exception:
        status["ffmpeg"] = "❌ ffmpeg not found"

    # Avatars
    avatar_count = len(list(AVATARS.glob("*.png")))
    status["avatars"] = f"✅ {avatar_count} avatars available" if avatar_count > 0 else "⚠️ 0 avatars — run: python scripts/generate_avatars.py"

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // 1_000_000
            status["gpu"] = f"✅ {gpu} ({vram} MB VRAM)"
        else:
            status["gpu"] = "⚠️ No GPU — CPU mode (slow but works)"
    except Exception:
        status["gpu"] = "⚠️ torch not available"

    return status


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=str(ROOT / "web" / "assets" / "favicon.ico") if (ROOT / "web" / "assets" / "favicon.ico").exists() else None,
    )
