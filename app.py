"""
Avatar Studio — Production App (v2.0)

Full pipeline:
  Content (Instagram/YouTube/Audio/Text)
  → Script (llama3.1:8b, Telugu anchor style)
  → TTS (edge-tts Microsoft Neural, primary)
  → Lip Sync (MuseTalk, fallback: static)
  → Compose (FFmpeg, background + lower-third + subtitles)
  → Export (16:9 · 9:16 Reels · 1:1 Instagram)

Security:
  - server_name=127.0.0.1 (Nginx proxies — never 0.0.0.0)
  - share=False
  - Port 7860 NEVER opened in Security Group
"""
import json
import logging
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from config import (
    OUTPUTS_DIR, AVATARS_DIR, SCENES, AVATARS, LANGUAGES,
    VIDEO_FORMATS, VOICE_PROFILES, BACKGROUNDS_DIR,
    OLLAMA_MODEL,
)
from pipeline.content_engine  import process_source
from pipeline.script_engine   import process_content as generate_script_pipeline
from pipeline.tts_engine      import synthesize, add_room_acoustics
from pipeline.musetalk_engine import generate_lipsync, is_available as musetalk_available
from pipeline.video_engine    import compose_video, compose_debate, generate_thumbnail
from pipeline.avatar_engine   import get_avatar_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "logs" / "app.log"),
    ],
)
log = logging.getLogger("app")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

# Generation lock — single GPU handles one job at a time
_gen_lock = threading.Lock()

# ── UI option lists ───────────────────────────────────────────────────────────
SCENE_CHOICES    = [(v["label"], k) for k, v in SCENES.items()]
AVATAR_CHOICES   = [(f"{v['name']} ({LANGUAGES.get(v['language'], v['language'])})", k)
                    for k, v in AVATARS.items()]
LANG_CHOICES     = [(v, k) for k, v in LANGUAGES.items()]
FORMAT_CHOICES   = [(v["desc"], k) for k, v in VIDEO_FORMATS.items()] + [("Auto Detect", "auto")]
CHAR_CHOICES     = [("🎤 Navya Reddy (Telugu F, Hyderabad)", "navya"),
                    ("🎙️ Arjun Varma (Telugu M, Vijayawada)", "arjun")]
POSE_CHOICES     = [("Half Body — recommended", "half_body"),
                    ("Standing Full Body", "standing"),
                    ("Sitting at Desk", "sitting_desk")]
ATTIRE_CHOICES   = [("Professional", "professional"), ("Suit", "suit"),
                    ("Traditional (Saree/Kurta)", "traditional_saree"),
                    ("Casual", "casual")]
CAT_CHOICES      = [
    ("Auto Detect", "general"), ("Bigg Boss / Reality TV", "bigg_boss"),
    ("Movie Review", "movie_review"), ("Tech Review", "tech_review"),
    ("News / Current Affairs", "general"), ("Festival / Special", "festival"),
]


# ── Step 1: Extract content ───────────────────────────────────────────────────

def step_extract(
    url: str, audio_file, topic_text: str, language: str
) -> tuple[str, str]:
    """
    Extract content from Instagram Reel / YouTube / audio upload / topic text.
    Returns (transcript, status_message)
    """
    audio_path = audio_file if audio_file else ""

    result = process_source(
        url=url or "",
        audio_path=str(audio_path) if audio_path else "",
        topic_text=topic_text or "",
        language=language,
    )

    if result.get("error"):
        return "", f"❌ {result['error']}"

    transcript = result["transcript"]
    platform   = result["platform"]
    title      = result.get("title", "")
    info = f"✅ Content extracted | Source: {platform} | {len(transcript)} chars"
    if title:
        info += f" | Title: {title[:60]}"

    return transcript, info


# ── Step 2: Generate script ───────────────────────────────────────────────────

def step_generate_script(
    transcript: str,
    language: str,
    format_type: str,
    duration: int,
    topic: str,
    character_id: str,
    category: str,
    source_url: str,
) -> tuple[str, str, str]:
    """
    Generate script from transcript using llama3.1:8b.
    Returns (script, scene_key, status)
    """
    if not transcript.strip():
        return "", "professional/office", "❌ No transcript. Extract content first (Step 1)."

    try:
        result = generate_script_pipeline(
            source=transcript,
            language=language,
            format_type=format_type if format_type != "auto" else "auto",
            duration_sec=duration,
            topic=topic or "",
            character_id=character_id,
            topic_category=category,
            save_to_memory=True,
        )

        if result.get("error"):
            return "", "professional/office", f"❌ Script error: {result['error']}"

        script   = result["script"]
        fmt      = result["format"]
        scene    = result["metadata"].get("detected_scene", "professional/office")
        entities = result["metadata"].get("entities", [])

        status = (
            f"✅ Script ready | Model: {OLLAMA_MODEL} | Format: {fmt} | "
            f"Scene: {SCENES.get(scene, {}).get('label', scene)} | "
            f"{len(script)} chars"
        )
        if entities:
            status += f" | Keywords: {', '.join(entities[:4])}"

        return script, scene, status

    except Exception as e:
        log.exception("Script generation error")
        return "", "professional/office", f"❌ Error: {e}"


# ── Full generate pipeline ────────────────────────────────────────────────────

def generate_video(
    approved_script: str,
    language: str,
    character_id: str,
    persona_id: str,
    pose: str,
    attire: str,
    scene_key: str,
    format_type: str,
    display_name: str,
    display_title: str,
    show_subtitles: bool,
    export_vertical: bool,
    export_square: bool,
    persona_b_id: str,
    attire_b: str,
    use_parler: bool,
    enhance_faces: bool,
    progress=gr.Progress(),
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Full generation pipeline. Returns (status, video_path, vertical_path, thumbnail_path).
    """
    if not approved_script.strip():
        return "❌ No script. Complete Steps 1 & 2 and click Approve.", None, None, None

    if not _gen_lock.acquire(blocking=False):
        return "⏳ Another video is currently generating. Please wait...", None, None, None

    job_id  = str(uuid.uuid4())[:8]
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info(f"[{job_id}] Generation start | format={format_type} char={character_id}")

        # ── Step 1: Avatar ────────────────────────────────────────────────────
        progress(0.05, desc="Loading avatar...")
        avatar_img = get_avatar_path(persona_id, pose, attire)
        if not avatar_img:
            return (
                f"❌ Avatar not found: {persona_id}/{pose}/{attire}. "
                "Run python scripts/generate_avatars.py",
                None, None, None,
            )

        # ── Step 2: Split script for debate ───────────────────────────────────
        script_a = approved_script
        script_b = ""
        if format_type == "debate":
            lines_a, lines_b = [], []
            for line in approved_script.splitlines():
                s = line.strip()
                if s.upper().startswith(("NAVYA:", "SPEAKER_A:")):
                    lines_a.append(re.sub(r'^(NAVYA|SPEAKER_A):\s*', '', s, flags=re.I))
                elif s.upper().startswith(("ARJUN:", "SPEAKER_B:")):
                    lines_b.append(re.sub(r'^(ARJUN|SPEAKER_B):\s*', '', s, flags=re.I))
            script_a = " ".join(lines_a) if lines_a else approved_script
            script_b = " ".join(lines_b)

        # ── Step 3: TTS ───────────────────────────────────────────────────────
        progress(0.10, desc="Generating voice (edge-tts)...")
        voice_profile = AVATARS[persona_id]["voice_profile"]
        audio_raw     = job_dir / "voice_raw.wav"
        audio_final   = job_dir / "voice_final.wav"

        ok = synthesize(script_a, voice_profile, audio_raw, use_parler=use_parler)
        if not ok:
            return "❌ TTS failed. Check logs/app.log.", None, None, None

        # Room acoustics
        scene_cfg   = SCENES.get(scene_key, {})
        reverb_type = scene_cfg.get("reverb", "small_room")
        add_room_acoustics(audio_raw, reverb_type, audio_final)
        if not audio_final.exists():
            import shutil; shutil.copy2(str(audio_raw), str(audio_final))

        # ── Step 4: Lip Sync (MuseTalk) ───────────────────────────────────────
        mt_status = "MuseTalk" if musetalk_available() else "static (MuseTalk not installed)"
        progress(0.25, desc=f"Lip sync ({mt_status})...")

        lipsync_a = job_dir / "lipsync_a.mp4"
        ok = generate_lipsync(
            avatar_path=Path(avatar_img),
            audio_path=audio_final,
            out_path=lipsync_a,
            enhance=enhance_faces,
        )
        if not ok:
            return "❌ Lip sync failed. Check logs/app.log.", None, None, None

        # ── Step 5: Debate — second speaker ───────────────────────────────────
        lipsync_b = None
        audio_b   = None
        if format_type == "debate" and script_b:
            progress(0.55, desc="Generating debate Speaker B...")
            avatar_b = get_avatar_path(persona_b_id, pose, attire_b)
            if not avatar_b:
                avatar_b = avatar_img

            voice_b = AVATARS.get(persona_b_id, AVATARS[persona_id])["voice_profile"]
            audio_b_raw = job_dir / "voice_b_raw.wav"
            audio_b     = job_dir / "voice_b.wav"
            lipsync_b   = job_dir / "lipsync_b.mp4"

            synthesize(script_b, voice_b, audio_b_raw, use_parler=use_parler)
            add_room_acoustics(audio_b_raw, reverb_type, audio_b)
            if not audio_b.exists():
                import shutil; shutil.copy2(str(audio_b_raw), str(audio_b))

            generate_lipsync(Path(avatar_b), audio_b, lipsync_b, enhance=enhance_faces)

        # ── Step 6: Compose ───────────────────────────────────────────────────
        progress(0.75, desc="Composing final video (FFmpeg)...")
        final_video = job_dir / f"avatar_studio_{job_id}.mp4"
        name_out    = display_name or AVATARS[persona_id]["name"]
        title_out   = display_title or "AI Avatar"

        if format_type == "debate" and lipsync_b and audio_b:
            name_b = AVATARS.get(persona_b_id, {}).get("name", "Speaker B")
            ok = compose_debate(
                lipsync_a=lipsync_a,
                lipsync_b=lipsync_b,
                audio_a=audio_final,
                audio_b=audio_b,
                name_a=name_out,
                name_b=name_b,
                scene_key=scene_key,
                out_path=final_video,
            )
        else:
            ok = compose_video(
                lipsync_video=lipsync_a,
                audio_path=audio_final,
                scene_key=scene_key,
                out_path=final_video,
                lower_third_name=name_out,
                lower_third_title=title_out,
                script_text=script_a,
                language=language,
                show_subtitles=show_subtitles,
                export_vertical=export_vertical,
                export_square=export_square,
            )

        if not ok or not final_video.exists():
            return "❌ Compose step failed. Check logs/app.log.", None, None, None

        # ── Step 7: Thumbnail ─────────────────────────────────────────────────
        progress(0.92, desc="Generating thumbnail...")
        thumb_path = job_dir / "thumbnail.jpg"
        generate_thumbnail(Path(avatar_img), name_out, thumb_path)

        # ── Save script text ──────────────────────────────────────────────────
        script_file = job_dir / f"script_{job_id}.txt"
        script_file.write_text(approved_script, encoding="utf-8")

        # Vertical path (created by compose_video if export_vertical=True)
        vert_path = final_video.with_stem(final_video.stem + "_reels")
        vert_out  = str(vert_path) if vert_path.exists() else None

        size_mb = final_video.stat().st_size // (1024 * 1024)
        progress(1.0, desc="Done!")
        log.info(f"[{job_id}] Complete: {final_video} ({size_mb}MB)")

        status = (
            f"✅ Video ready! | Job: {job_id} | Size: {size_mb}MB | "
            f"Lip sync: {mt_status} | Formats: 16:9"
            + (" + 9:16" if vert_out else "")
        )
        return status, str(final_video), vert_out, str(thumb_path) if thumb_path.exists() else None

    except Exception as e:
        log.exception(f"[{job_id}] Generation crashed")
        return f"❌ Unexpected error: {e}\n\nSee logs/app.log", None, None, None
    finally:
        _gen_lock.release()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

import re

def build_ui() -> gr.Blocks:
    musetalk_status = "✅ MuseTalk installed" if musetalk_available() else "⚠️ MuseTalk not installed (static fallback active)"

    with gr.Blocks(
        title="Avatar Studio",
        theme=gr.themes.Base(
            primary_hue="blue", neutral_hue="slate",
            font=["Inter", "Noto Sans", "sans-serif"],
        ),
        css="""
        body { font-family: 'Inter', sans-serif; }
        .tab-nav button { font-size: 14px; font-weight: 600; padding: 10px 16px; }
        .bigbtn { background: #1565c0 !important; color: white !important;
                  font-size: 17px !important; height: 52px !important;
                  font-weight: 700 !important; }
        .status { font-family: monospace; font-size: 13px; }
        .step-header { font-size: 16px; font-weight: 700; color: #1565c0;
                       margin-bottom: 8px; }
        .info-box { background: #f0f4ff; border-left: 3px solid #1565c0;
                    padding: 12px; border-radius: 4px; font-size: 13px; }
        """,
    ) as app:

        gr.Markdown(f"""
# 🎬 Avatar Studio v2.0
**Telugu AI Video Generator** — Instagram Reels, YouTube, Audio, Text → Professional video
{musetalk_status} | LLM: {OLLAMA_MODEL} | TTS: edge-tts (Microsoft Neural)
        """)

        # Shared state
        state_transcript = gr.State("")
        state_script     = gr.State("")
        state_scene      = gr.State("professional/office")
        state_format     = gr.State("monologue")

        with gr.Tabs():

            # ─── Tab 1: Content Source ────────────────────────────────────────
            with gr.TabItem("1 · Content"):
                gr.Markdown('<div class="step-header">Step 1 — Get your content</div>')
                gr.Markdown("""
<div class="info-box">
📱 <b>Instagram Reels</b> work on EC2 — paste reel URL directly.<br>
🎵 <b>YouTube</b> on EC2 is blocked — upload the MP3/audio file instead.<br>
✍️ <b>Topic text</b> always works — type what the video should cover.
</div>
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        url_input = gr.Textbox(
                            label="🔗 URL (Instagram Reel / YouTube)",
                            placeholder="https://www.instagram.com/reel/... or https://youtube.com/watch?v=...",
                            lines=2,
                        )
                        audio_upload = gr.Audio(
                            label="🎵 Upload Audio (MP3/WAV — works for YouTube too)",
                            type="filepath",
                        )
                        topic_input = gr.Textbox(
                            label="✍️ Topic / Script Text (paste transcript or describe the topic)",
                            placeholder="e.g. Google I/O 2025 announcements in Telugu — key highlights...",
                            lines=4,
                        )
                    with gr.Column(scale=1):
                        lang_input     = gr.Dropdown(LANG_CHOICES, value="te", label="Language")
                        format_input   = gr.Dropdown(FORMAT_CHOICES, value="auto", label="Video Format")
                        category_input = gr.Dropdown(CAT_CHOICES, value="general", label="Topic Category")
                        duration_input = gr.Slider(30, 600, value=180, step=30,
                                                    label="Target Duration (seconds)")
                        topic_label    = gr.Textbox(label="Topic label (optional)",
                                                    placeholder="e.g. Google I/O 2025")

                extract_btn = gr.Button("📥 Extract Content", variant="secondary")
                extract_status = gr.Textbox(label="Extraction Status", interactive=False,
                                            elem_classes="status", lines=2)
                transcript_box = gr.Textbox(label="Extracted Transcript (editable)",
                                            lines=8, interactive=True)

                extract_btn.click(
                    fn=step_extract,
                    inputs=[url_input, audio_upload, topic_input, lang_input],
                    outputs=[transcript_box, extract_status],
                )

            # ─── Tab 2: Script ────────────────────────────────────────────────
            with gr.TabItem("2 · Script"):
                gr.Markdown('<div class="step-header">Step 2 — Generate & review script</div>')

                with gr.Row():
                    with gr.Column(scale=1):
                        char_input     = gr.Dropdown(CHAR_CHOICES, value="navya",
                                                      label="Character")
                        gr.Markdown("""**Script options:**""")
                        with gr.Accordion("Debate settings", open=False):
                            debate_navya = gr.Textbox(label="Navya's position",
                                                       value="in favour")
                            debate_arjun = gr.Textbox(label="Arjun's position",
                                                       value="against")

                    with gr.Column(scale=2):
                        gen_script_btn = gr.Button("🧠 Generate Script", variant="primary")
                        script_status = gr.Textbox(label="Status", interactive=False,
                                                   elem_classes="status", lines=2)

                script_box = gr.Textbox(
                    label="Generated Script (edit freely before approving)",
                    lines=18, interactive=True,
                    placeholder="Script appears here after clicking Generate Script...",
                )

                with gr.Row():
                    scene_detected = gr.Textbox(label="Auto-detected scene", interactive=False)
                    approve_btn = gr.Button("✅ Approve Script → Ready to Generate",
                                           variant="primary", elem_classes="bigbtn")
                approve_status = gr.Textbox(label="", interactive=False)

                gen_script_btn.click(
                    fn=step_generate_script,
                    inputs=[transcript_box, lang_input, format_input, duration_input,
                            topic_label, char_input, category_input, url_input],
                    outputs=[script_box, state_scene, script_status],
                ).then(
                    fn=lambda scene: SCENES.get(scene, {}).get("label", scene),
                    inputs=[state_scene],
                    outputs=[scene_detected],
                )

                approve_btn.click(
                    fn=lambda s, fmt: (s, fmt or "monologue", "✅ Script approved. Go to Generate tab."),
                    inputs=[script_box, format_input],
                    outputs=[state_script, state_format, approve_status],
                )

            # ─── Tab 3: Avatar & Scene ────────────────────────────────────────
            with gr.TabItem("3 · Avatar & Scene"):
                gr.Markdown('<div class="step-header">Step 3 — Choose your presenter and background</div>')

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Primary Avatar")
                        persona_input  = gr.Dropdown(AVATAR_CHOICES, value="navya_telugu_f",
                                                      label="Avatar Persona")
                        pose_input     = gr.Dropdown(POSE_CHOICES, value="half_body", label="Pose")
                        attire_input   = gr.Dropdown(ATTIRE_CHOICES, value="professional", label="Attire")
                        name_input     = gr.Textbox(label="Display Name (lower third)",
                                                     placeholder="e.g. Navya Reddy")
                        title_input    = gr.Textbox(label="Display Title",
                                                     placeholder="e.g. AI News Anchor")

                    with gr.Column():
                        gr.Markdown("#### Background Scene")
                        scene_input    = gr.Dropdown(SCENE_CHOICES, value="professional/office",
                                                      label="Background")
                        gr.Markdown("#### Debate — Second Speaker (if debate format)")
                        persona_b      = gr.Dropdown(AVATAR_CHOICES, value="arjun_telugu_m",
                                                      label="Debate Speaker B")
                        attire_b       = gr.Dropdown(ATTIRE_CHOICES, value="suit",
                                                      label="Speaker B Attire")

                # Sync detected scene to dropdown
                state_scene.change(
                    fn=lambda s: gr.update(value=s),
                    inputs=[state_scene],
                    outputs=[scene_input],
                )

            # ─── Tab 4: Generate ──────────────────────────────────────────────
            with gr.TabItem("4 · Generate"):
                gr.Markdown('<div class="step-header">Step 4 — Generate video</div>')
                gr.Markdown("""
<div class="info-box">
Make sure you have <b>approved the script</b> in Tab 2 before generating.
</div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Output options:**")
                        show_subs_cb  = gr.Checkbox(value=True,
                                                     label="Burn subtitles (Telugu Unicode)")
                        export_vert_cb = gr.Checkbox(value=True,
                                                      label="Export 9:16 (Reels/Shorts)")
                        export_sq_cb  = gr.Checkbox(value=False,
                                                     label="Export 1:1 (Instagram feed)")
                        gr.Markdown("**Voice options:**")
                        use_parler_cb = gr.Checkbox(value=False,
                                                     label="Use Indic Parler-TTS (best quality, needs HF approval)")
                        enhance_cb    = gr.Checkbox(value=True,
                                                     label="GFPGAN face enhancement")

                    with gr.Column(scale=2):
                        gen_btn = gr.Button("🚀 Generate Video", variant="primary",
                                            elem_classes="bigbtn")
                        gen_status = gr.Textbox(label="Generation Status", lines=3,
                                                interactive=False, elem_classes="status")

                with gr.Row():
                    video_out    = gr.Video(label="🎬 Generated Video (16:9)", height=400)
                    video_vert   = gr.Video(label="📱 Reels / Shorts (9:16)", height=400)

                with gr.Row():
                    thumb_out    = gr.Image(label="🖼️ Thumbnail", height=200)

                gr.Markdown("""
**Generation time on T4:**

| Duration | MuseTalk installed | Static fallback |
|---|---|---|
| 60s short | ~3 min | ~1 min |
| 3 min KT video | ~6 min | ~2 min |
| 8 min review | ~12 min | ~4 min |

*MuseTalk runs at ~1× real-time on T4. Static = no lip sync.*
                """)

                gen_btn.click(
                    fn=generate_video,
                    inputs=[
                        state_script, lang_input, char_input,
                        persona_input, pose_input, attire_input,
                        scene_input, state_format,
                        name_input, title_input,
                        show_subs_cb, export_vert_cb, export_sq_cb,
                        persona_b, attire_b,
                        use_parler_cb, enhance_cb,
                    ],
                    outputs=[gen_status, video_out, video_vert, thumb_out],
                )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Avatar Studio v2.0...")
    log.info(f"MuseTalk: {'available' if musetalk_available() else 'not installed (static fallback)'}")
    log.info(f"LLM: {OLLAMA_MODEL}")
    log.info(f"Outputs: {OUTPUTS_DIR}")

    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",   # Nginx proxies — NEVER change to 0.0.0.0
        server_port=7860,
        share=False,               # NEVER set to True
        show_error=True,
        max_threads=2,
    )
