"""
app.py — Avatar Studio v3.0 — Full Production UI

Tab 1: Content     → URL (Instagram/YouTube) | Audio upload | YouTube cookies.txt | Topic text
Tab 2: Script      → LLM generation | Emotion-tagged | Review + Approve gate
Tab 3: Avatar      → Persona | Pose | Attire | Scene | Debate speaker
Tab 4: Generate    → Full pipeline | Progress | Download all formats

Pipeline: SadTalker → MuseTalk → GFPGAN → FFmpeg compose → multi-format output

Security:
  server_name="127.0.0.1"  (Nginx proxies port 80 only — never open 7860)
  share=False
  max_threads=2
"""
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
log = logging.getLogger("app")

ROOT    = Path(__file__).parent.resolve()
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline.content_engine  import process_source
from pipeline.script_engine   import generate_script, generate_debate, check_script_quality, strip_emotion_tags
from pipeline.tts_engine      import synthesize, synthesize_segmented, detect_emotion
from pipeline.musetalk_engine import generate_lipsync, get_pipeline_status
from pipeline.video_engine    import (
    compose_video, export_vertical, export_square,
    generate_thumbnail, generate_ass_subtitles,
)
from pipeline.avatar_engine   import load_avatar_image

# ── Config ────────────────────────────────────────────────────────────────────
from config import AVATARS, SCENES

# ── Pipeline status banner ────────────────────────────────────────────────────
_STATUS = get_pipeline_status()
_STATUS_LINES = []
_STATUS_LINES.append("✅ SadTalker: head motion + eye blink + expressions"
                     if _STATUS["sadtalker"] else
                     "⚠️ SadTalker: not installed (run setup.sh) — no head motion")
_STATUS_LINES.append("✅ MuseTalk: lip sync"
                     if _STATUS["musetalk"] else
                     "⚠️ MuseTalk: not installed — static avatar fallback")
_STATUS_LINES.append("✅ GFPGAN: face quality enhancement"
                     if _STATUS["gfpgan"] else
                     "⚠️ GFPGAN: not installed")
STATUS_BANNER = "\n".join(_STATUS_LINES)

AVATAR_CHOICES = {
    k: f"{v['name']} — {v['language'].upper()} {v['gender'].title()}"
    for k, v in AVATARS.items()
}

# ── Cookies path ──────────────────────────────────────────────────────────────
COOKIES_PATH = ROOT / "models" / "yt_cookies.txt"


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONTENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_content(url, audio_file, topic_text, language, cookies_file):
    """Extract and transcribe content from any source."""
    try:
        # Save cookies if uploaded
        if cookies_file is not None:
            COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(cookies_file.name, str(COOKIES_PATH))
            log.info(f"Cookies saved → {COOKIES_PATH}")

        # Inject cookies path into env for yt-dlp
        if COOKIES_PATH.exists():
            os.environ["YTDLP_COOKIES"] = str(COOKIES_PATH)

        result = process_source(
            url=url or "",
            audio_path=audio_file.name if audio_file else "",
            topic_text=topic_text or "",
            language=language,
        )

        if result.get("error"):
            return gr.update(value=f"⚠️ {result['error']}", visible=True), "", gr.update()

        transcript = result["transcript"]
        platform   = result["platform"]
        title      = result.get("title", "")
        duration   = result.get("duration", 0)

        status = f"✅ Content extracted | Source: {platform}"
        if title:
            status += f" | {title}"
        if duration:
            status += f" | {int(duration)}s"
        status += f" | {len(transcript.split())} words"

        return (
            gr.update(value=status,  visible=True),
            transcript,
            gr.update(interactive=True),
        )

    except Exception as e:
        log.error(f"Content extraction: {e}")
        return gr.update(value=f"❌ {e}", visible=True), "", gr.update()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCRIPT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def gen_script(transcript, persona_id, duration, content_type, use_debate, persona_b_id):
    """Generate Telugu script with emotion tags."""
    if not transcript.strip():
        return "⚠️ No content — go back to Tab 1 and extract content first.", "", gr.update()

    try:
        if use_debate and persona_b_id:
            result = generate_debate(
                transcript=transcript,
                persona_a_id=persona_id,
                persona_b_id=persona_b_id,
                duration=duration,
            )
        else:
            result = generate_script(
                transcript=transcript,
                persona_id=persona_id,
                duration=duration,
                content_type=content_type,
                add_emotion_tags=True,
            )

        quality = check_script_quality(result["script"])
        script_display = strip_emotion_tags(result["script"])
        word_count = result.get("word_count", 0)
        est_min    = result.get("estimated_duration_sec", 0) // 60
        est_sec    = result.get("estimated_duration_sec", 0) % 60

        info_line = (
            f"✅ Script ready | ~{word_count} words | "
            f"~{est_min}m{est_sec:02d}s | "
            f"{'⚠️ ' + ' · '.join(quality['issues']) if quality['issues'] else 'Quality OK'}"
        )

        return (
            info_line,
            script_display,
            gr.update(interactive=True),   # Approve button
        )

    except Exception as e:
        log.error(f"Script generation: {e}")
        return f"❌ {e}", "", gr.update()


def approve_script(script_display):
    """Mark script as approved and move to Tab 3."""
    if not script_display.strip():
        return "⚠️ Script is empty — generate one first.", gr.update()
    return "✅ Script approved — proceed to Tab 3", gr.update(interactive=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_video(
    transcript, script_display, persona_id, pose, attire, scene,
    debate_mode, persona_b_id,
    use_subtitles, export_reels, export_square_fmt, use_gfpgan, use_parler,
    language, progress=gr.Progress(track_tqdm=True),
):
    """Full production pipeline: TTS → lip sync → compose → multi-format."""
    if not script_display.strip():
        return None, None, None, "❌ No script — complete Tabs 1-3 first."

    job_id  = str(uuid.uuid4())[:8]
    job_dir = OUTPUTS / job_id
    job_dir.mkdir(parents=True)

    try:
        # ── 1. TTS ──────────────────────────────────────────────────────────
        progress(0.05, desc="🎙️ Synthesising voice...")
        audio_path = job_dir / "audio.wav"

        # Reconstruct script with emotion tags for segmented synthesis
        ok = synthesize_segmented(
            text=script_display,
            voice_profile=AVATARS[persona_id]["voice_profile"],
            out_path=str(audio_path),
            use_parler=use_parler,
        )
        if not ok:
            return None, None, None, "❌ TTS failed. Check logs."

        # ── 2. Load avatar image ────────────────────────────────────────────
        progress(0.15, desc="🖼️ Loading avatar...")
        avatar_cfg   = AVATARS[persona_id]
        avatar_image = load_avatar_image(persona_id, pose, attire)

        # Save avatar image to job dir for reference
        import shutil
        saved_avatar = job_dir / "avatar.png"
        shutil.copy2(str(avatar_image), str(saved_avatar))

        # ── 3. Lip sync (SadTalker + MuseTalk + GFPGAN) ────────────────────
        progress(0.20, desc="💋 Animating face (SadTalker + MuseTalk)...")
        avatar_video = job_dir / "avatar_lipsync.mp4"

        dominant_emotion = detect_emotion(script_display[:500])
        generate_lipsync(
            source_image=saved_avatar,
            audio_path=audio_path,
            out_path=avatar_video,
            use_gfpgan=use_gfpgan,
            emotion=dominant_emotion,
        )

        if not avatar_video.exists():
            return None, None, None, "❌ Lip sync failed. Check models."

        # ── 4. Subtitles ────────────────────────────────────────────────────
        subtitle_path = None
        if use_subtitles:
            progress(0.60, desc="📝 Generating subtitles...")
            subtitle_path = job_dir / "subtitles.ass"
            subtitle_path = generate_ass_subtitles(audio_path, subtitle_path, language)

        # ── 5. Compose 16:9 ────────────────────────────────────────────────
        progress(0.70, desc="🎬 Composing broadcast video...")
        final_video = job_dir / f"avatar_studio_{job_id}.mp4"

        compose_video(
            avatar_video=avatar_video,
            audio_path=audio_path,
            out_path=final_video,
            scene=scene,
            persona_name=avatar_cfg["name"],
            persona_title=avatar_cfg.get("title", "Telugu Anchor"),
            subtitle_path=subtitle_path,
            lower_third=True,
            color_grade=True,
            breathing=True,
        )

        if not final_video.exists():
            return None, None, None, "❌ Video composition failed."

        # ── 6. Reels export (9:16) ──────────────────────────────────────────
        reels_video = None
        if export_reels:
            progress(0.85, desc="📱 Exporting Reels (9:16)...")
            reels_path = job_dir / f"avatar_studio_{job_id}_reels.mp4"
            if export_vertical(final_video, reels_path):
                reels_video = str(reels_path)

        # ── 7. Square export (1:1) ──────────────────────────────────────────
        square_video = None
        if export_square_fmt:
            progress(0.90, desc="⬜ Exporting square (1:1)...")
            square_path = job_dir / f"avatar_studio_{job_id}_square.mp4"
            if export_square(final_video, square_path):
                square_video = str(square_path)

        # ── 8. Thumbnail ────────────────────────────────────────────────────
        progress(0.95, desc="🖼️ Generating thumbnail...")
        thumb_path = job_dir / "thumbnail.jpg"
        generate_thumbnail(
            avatar_image=saved_avatar,
            title_text=script_display[:80],
            out_path=thumb_path,
        )

        # ── 9. Save script ──────────────────────────────────────────────────
        (job_dir / f"script_{job_id}.txt").write_text(script_display, encoding="utf-8")

        status = (
            f"✅ Generation complete | Job: {job_id}\n"
            f"📁 Files saved to: outputs/{job_id}/\n"
            f"📹 16:9 (YouTube): avatar_studio_{job_id}.mp4\n"
        )
        if reels_video:
            status += f"📱 9:16 (Reels):  avatar_studio_{job_id}_reels.mp4\n"
        if square_video:
            status += f"⬜ 1:1 (Instagram): avatar_studio_{job_id}_square.mp4\n"

        progress(1.0, desc="✅ Done!")
        return str(final_video), reels_video, square_video, status

    except Exception as e:
        log.exception(f"Generation failed: {e}")
        return None, None, None, f"❌ Generation failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
#status-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: monospace;
    font-size: 13px;
    border: 1px solid #2a3a5e;
    white-space: pre-line;
}
.status-ok  { color: #4ade80; }
.status-warn{ color: #fbbf24; }
.tab-header { font-weight: 700; color: #e8920a; }
"""

with gr.Blocks(
    title="Avatar Studio v3.0",
    theme=gr.themes.Base(primary_hue="orange"),
    css=CUSTOM_CSS,
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown("# 🎬 Avatar Studio v3.0 — Telugu AI Anchor Platform")
    gr.Markdown(
        f"**Pipeline:** SadTalker (head+blink) → MuseTalk (lip sync) → GFPGAN (face quality) → FFmpeg (broadcast compose)\n\n"
        f"```\n{STATUS_BANNER}\n```"
    )

    # ── State ───────────────────────────────────────────────────────────────
    state_transcript = gr.State("")
    state_script     = gr.State("")

    # ────────────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── TAB 1: CONTENT ───────────────────────────────────────────────────
        with gr.Tab("📥 1 · Content"):
            gr.Markdown("### Extract content from any source")

            with gr.Row():
                with gr.Column(scale=2):
                    inp_url = gr.Textbox(
                        label="Instagram Reel or YouTube URL",
                        placeholder="https://www.instagram.com/reel/XXXXXXXXX/  or  https://youtu.be/XXXX",
                    )
                    inp_audio = gr.File(
                        label="Upload Audio (MP3/WAV) — use this for YouTube videos downloaded on your PC",
                        file_types=[".mp3", ".wav", ".m4a", ".ogg"],
                    )
                    inp_topic = gr.Textbox(
                        label="Or type a topic / script directly",
                        placeholder="Bigg Boss Telugu Season 8 Day 10 — eviction results and drama...",
                        lines=4,
                    )

                with gr.Column(scale=1):
                    inp_cookies = gr.File(
                        label="YouTube Cookies (cookies.txt) — required for YouTube downloads on EC2",
                        file_types=[".txt"],
                    )
                    gr.Markdown(
                        "**How to get cookies.txt:**\n"
                        "1. Install: *Get cookies.txt LOCALLY* (Chrome extension)\n"
                        "2. Log in to youtube.com\n"
                        "3. Click extension → Export cookies → upload here\n\n"
                        "⚠️ Cookies expire after ~30 days — re-upload if YouTube blocks"
                    )
                    inp_language = gr.Dropdown(
                        label="Content Language",
                        choices=["te", "hi", "ta", "kn", "en"],
                        value="te",
                    )

            btn_extract = gr.Button("🔍 Extract Content", variant="primary")
            extract_status = gr.Textbox(label="Status", interactive=False, visible=False)
            out_transcript  = gr.Textbox(label="Extracted Content Preview", lines=6, interactive=False)

            btn_extract.click(
                extract_content,
                inputs=[inp_url, inp_audio, inp_topic, inp_language, inp_cookies],
                outputs=[extract_status, out_transcript, btn_extract],
            )

        # ── TAB 2: SCRIPT ────────────────────────────────────────────────────
        with gr.Tab("✍️ 2 · Script"):
            gr.Markdown("### Generate & review Telugu anchor script")

            with gr.Row():
                sel_persona   = gr.Dropdown(label="Anchor", choices=list(AVATAR_CHOICES.keys()),
                                            value=list(AVATAR_CHOICES.keys())[0])
                sel_duration  = gr.Dropdown(label="Duration",
                                            choices=["30s","60s","2min","3min","5min","8min","10min"],
                                            value="3min")
                sel_content   = gr.Dropdown(label="Content Type",
                                            choices=["news","analysis","entertainment","educational"],
                                            value="news")

            with gr.Row():
                chk_debate  = gr.Checkbox(label="Debate Format (two anchors)", value=False)
                sel_persona_b = gr.Dropdown(label="Debate Speaker B",
                                             choices=list(AVATAR_CHOICES.keys()),
                                             visible=False)

            chk_debate.change(lambda v: gr.update(visible=v), chk_debate, sel_persona_b)

            btn_gen_script = gr.Button("🤖 Generate Script (llama3.1:8b)", variant="primary")
            script_status  = gr.Textbox(label="Status", interactive=False)
            out_script     = gr.Textbox(
                label="Generated Script (edit if needed — emotion tags are stripped for display)",
                lines=15, interactive=True,
            )

            with gr.Row():
                btn_approve = gr.Button("✅ Approve Script", variant="secondary", interactive=False)
                approve_status = gr.Textbox(label="", interactive=False, scale=3)

            btn_gen_script.click(
                gen_script,
                inputs=[out_transcript, sel_persona, sel_duration, sel_content,
                        chk_debate, sel_persona_b],
                outputs=[script_status, out_script, btn_approve],
            )
            btn_approve.click(
                approve_script,
                inputs=[out_script],
                outputs=[approve_status, btn_approve],
            )

        # ── TAB 3: AVATAR & SCENE ────────────────────────────────────────────
        with gr.Tab("🎭 3 · Avatar & Scene"):
            gr.Markdown("### Configure your avatar and scene")

            with gr.Row():
                with gr.Column():
                    av_persona = gr.Dropdown(
                        label="Avatar",
                        choices=list(AVATAR_CHOICES.items()),
                        value=list(AVATAR_CHOICES.keys())[0],
                    )
                    av_pose = gr.Radio(
                        label="Pose",
                        choices=["half_body", "standing"],
                        value="half_body",
                    )
                    av_attire = gr.Radio(
                        label="Attire",
                        choices=["professional", "traditional", "casual"],
                        value="professional",
                    )

                with gr.Column():
                    av_scene = gr.Dropdown(
                        label="Background Scene",
                        choices=list(SCENES.keys()) if isinstance(list(SCENES.values())[0], str)
                                else [k for k in SCENES],
                        value=list(SCENES.keys())[0],
                    )
                    gr.Markdown(
                        "**Scene tip:** 'studio' works for all content. "
                        "'parliament' for political news. 'entertainment' for showbiz."
                    )

        # ── TAB 4: GENERATE ──────────────────────────────────────────────────
        with gr.Tab("🚀 4 · Generate"):
            gr.Markdown("### Generate your broadcast-quality video")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Output options:**")
                    opt_subtitles = gr.Checkbox(label="Burn subtitles (Telugu Unicode)", value=True)
                    opt_reels     = gr.Checkbox(label="Export 9:16 Reels/Shorts version", value=True)
                    opt_square    = gr.Checkbox(label="Export 1:1 Instagram feed version", value=False)
                    opt_gfpgan    = gr.Checkbox(label="GFPGAN face enhancement (slower)", value=True)
                    opt_parler    = gr.Checkbox(
                        label="Use Indic Parler-TTS (best quality — needs HF gated access)",
                        value=False,
                    )

            btn_generate = gr.Button(
                "🎬 Generate Video  (1–30 min depending on duration + options)",
                variant="primary", size="lg",
            )

            gen_status   = gr.Textbox(label="Progress", interactive=False, lines=6)

            with gr.Row():
                out_main   = gr.Video(label="📹 16:9 (YouTube/LinkedIn)")
                out_reels  = gr.Video(label="📱 9:16 (Reels/Shorts)", visible=True)

            out_square = gr.Video(label="⬜ 1:1 (Instagram feed)", visible=False)

            opt_reels.change(lambda v: gr.update(visible=v), opt_reels, out_reels)
            opt_square.change(lambda v: gr.update(visible=v), opt_square, out_square)

            btn_generate.click(
                generate_video,
                inputs=[
                    out_transcript,    # transcript from Tab 1
                    out_script,        # script from Tab 2
                    av_persona,        # from Tab 3
                    av_pose, av_attire, av_scene,
                    chk_debate, sel_persona_b,
                    opt_subtitles, opt_reels, opt_square,
                    opt_gfpgan, opt_parler,
                    inp_language,
                ],
                outputs=[out_main, out_reels, out_square, gen_status],
            )

        # ── Help tab ─────────────────────────────────────────────────────────
        with gr.Tab("ℹ️ Help"):
            gr.Markdown("""
## Generation Times (T4 GPU)

| Duration | Full pipeline (SadTalker+MuseTalk+GFPGAN) | Without GFPGAN |
|---|---|---|
| 1 min | ~6 min | ~4 min |
| 3 min | ~14 min | ~9 min |
| 5 min | ~22 min | ~14 min |
| 10 min | ~40 min | ~25 min |

---

## YouTube Download on EC2

EC2 IPs are blocked by YouTube. Options:

**Option A — Cookies (recommended):**
1. Install "Get cookies.txt LOCALLY" Chrome extension
2. Log in to youtube.com
3. Export cookies → upload in Tab 1

**Option B — Download audio locally:**
```bash
pip install yt-dlp
yt-dlp -x --audio-format mp3 -o "video.mp3" "YOUTUBE_URL"
```
Then upload the MP3 in Tab 1.

**Option C — Instagram Reels:**
Instagram works perfectly on EC2. Use Reels as your source.

---

## Voice Quality

| Engine | When | Quality |
|---|---|---|
| Indic Parler-TTS | Enabled + HF access approved | ⭐⭐⭐⭐⭐ |
| edge-tts (ShrutiNeural) | Default | ⭐⭐⭐⭐ |
| gTTS | Fallback | ⭐⭐ |

Indic Parler-TTS: request access at huggingface.co/ai4bharat/indic-parler-tts

---

## Security

- Port 7860 never opened in AWS Security Group (Nginx proxies port 80)
- server_name=127.0.0.1 in app.py (never 0.0.0.0)
- share=False in Gradio
- cookies.txt stored locally in models/yt_cookies.txt (never committed to git)
            """)

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    log.info(f"Pipeline: SadTalker={_STATUS['sadtalker']} MuseTalk={_STATUS['musetalk']} GFPGAN={_STATUS['gfpgan']}")
    demo.launch(
        server_name="127.0.0.1",  # NEVER change to 0.0.0.0 — Nginx proxies port 80
        server_port=7860,
        share=False,
        max_threads=2,
        show_error=True,
    )
