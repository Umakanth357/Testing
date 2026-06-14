"""
Avatar Studio — Main Gradio Application

Tabs:
  1. Content    — YouTube URL or text input, language, duration
  2. Avatar     — Persona, pose, attire selection
  3. Scene      — Background, format, extras
  4. Review     — Script approval gate before generation
  5. Generate   — Run pipeline, download output
"""
import logging
import sys
import uuid
import threading
from pathlib import Path
from datetime import datetime
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
)
from pipeline.script_engine  import process_content
from pipeline.tts_engine     import synthesize, add_room_acoustics
from pipeline.animation_engine import animate, generate_walk_entrance
from pipeline.lipsync_engine import apply_lipsync
from pipeline.compose_engine import (
    compose_video, compose_debate, stitch_scene_segments,
)
from pipeline.avatar_engine  import get_avatar_path, list_available_avatars

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

# Generation lock — one job at a time on single GPU
_gen_lock = threading.Lock()

# ── Scene options for Gradio dropdown ─────────────────────────────────────────
SCENE_CHOICES = [(v["label"], k) for k, v in SCENES.items()]
AVATAR_CHOICES = [(f"{v['name']} ({LANGUAGES.get(v['language'], v['language'])})", k)
                  for k, v in AVATARS.items()]
LANGUAGE_CHOICES = [(v, k) for k, v in LANGUAGES.items()]
FORMAT_CHOICES   = [(v["desc"], k) for k, v in VIDEO_FORMATS.items()]
POSE_CHOICES     = [("Half Body (Default)", "half_body"),
                    ("Standing Full Body", "standing"),
                    ("Sitting at Desk", "sitting_desk")]
ATTIRE_CHOICES   = [("Professional", "professional"), ("Suit", "suit"),
                    ("Traditional (Saree/Kurta)", "traditional_saree"),
                    ("Casual", "casual"), ("Kurta", "kurta")]


# ── Tab 1: Script Preview (after content analysis) ────────────────────────────

def analyse_content(source: str, language: str, format_type: str,
                    duration: int, topic: str) -> tuple:
    """Step 1: Analyse content and generate script for review."""
    if not source.strip():
        return gr.update(value="⚠️ Please enter a YouTube URL or paste a script."), "", "", "monologue", []

    log.info(f"Analysing content | lang={language} format={format_type}")

    result = process_content(
        source=source.strip(),
        language=language,
        format_type=format_type,
        duration_sec=duration,
        topic=topic,
    )

    if result.get("error"):
        return gr.update(value=f"❌ Error: {result['error']}"), "", "", "monologue", []

    script   = result["script"]
    fmt      = result["format"]
    agenda   = result["agenda"]
    entities = result["metadata"].get("entities", [])
    scene    = result["metadata"].get("detected_scene", "professional/office")

    info = (f"✅ Script ready | Format: {fmt} | "
            f"Detected scene: {SCENES.get(scene, {}).get('label', scene)} | "
            f"Agenda items: {len(agenda)} | Entities: {', '.join(entities) or 'none'}")

    return (
        gr.update(value=info),
        script,
        scene,
        fmt,
        agenda,
    )


# ── Tab 5: Generate Video ─────────────────────────────────────────────────────

def generate_video(
    # Content
    approved_script: str,
    language: str,
    # Avatar
    persona_id: str,
    pose: str,
    attire: str,
    # Scene
    scene_key: str,
    format_type: str,
    # Options
    avatar_name: str,
    avatar_title: str,
    show_agenda: bool,
    agenda_json: str,
    export_vertical: bool,
    # Debate
    persona_b_id: str,
    attire_b: str,
    # Progress
    progress=gr.Progress(),
) -> tuple:
    """Full generation pipeline. Returns (status_text, video_path, script_path)."""

    if not approved_script.strip():
        return "❌ No script to generate from. Go to Content tab first.", None, None

    if not _gen_lock.acquire(blocking=False):
        return "⏳ Another video is generating. Please wait...", None, None

    job_id  = str(uuid.uuid4())[:8]
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        log.info(f"[{job_id}] Starting generation | format={format_type} pose={pose}")
        agenda = []
        try:
            agenda = eval(agenda_json) if agenda_json.strip() else []
        except Exception:
            pass

        # ── Step 1: Get avatar image ──────────────────────────────────────────
        progress(0.05, desc="Loading avatar...")
        avatar_img = get_avatar_path(persona_id, pose, attire)
        if not avatar_img:
            return (f"❌ Avatar not found: {persona_id}/{pose}/{attire}. "
                    f"Run scripts/generate_avatars.py first."), None, None

        # ── Step 2: TTS ───────────────────────────────────────────────────────
        progress(0.10, desc="Generating voice...")
        profile  = AVATARS[persona_id]["voice_profile"]
        voice    = VOICE_PROFILES.get(profile, {})
        ref_wav  = None  # TODO: wire up voice reference WAV
        audio_raw = job_dir / "voice_raw.wav"

        # For debate: split script
        if format_type == "debate" and "SPEAKER_A:" in approved_script.upper():
            from pipeline.script_engine import split_debate_script
            lines_a, lines_b = split_debate_script(approved_script)
            script_a = " ".join(lines_a)
            script_b = " ".join(lines_b)
        else:
            script_a = approved_script
            script_b = ""

        ok = synthesize(script_a, profile, audio_raw, ref_wav)
        if not ok:
            return "❌ TTS failed. Check logs.", None, None

        # ── Step 3: Room acoustics ────────────────────────────────────────────
        progress(0.20, desc="Applying audio acoustics...")
        scene_cfg   = SCENES.get(scene_key, {})
        reverb_type = scene_cfg.get("reverb", "small_room")
        audio_final = job_dir / "voice_final.wav"
        add_room_acoustics(audio_raw, reverb_type, audio_final)

        # ── Step 4: Animation ─────────────────────────────────────────────────
        progress(0.30, desc="Animating avatar (half body)...")
        anim_video = job_dir / "animated.mp4"
        ok = animate(avatar_img, audio_final, anim_video, pose=pose)
        if not ok:
            return "❌ Animation failed. Check EchoMimicV2 installation.", None, None

        # ── Step 5: Lip sync ──────────────────────────────────────────────────
        progress(0.55, desc="Applying lip sync (LatentSync)...")
        lipsync_video = job_dir / "lipsync.mp4"
        ok = apply_lipsync(anim_video, audio_final, lipsync_video)
        if not ok:
            return "❌ Lip sync failed. Check LatentSync installation.", None, None

        # ── Step 6: Compose ───────────────────────────────────────────────────
        progress(0.70, desc="Composing final video...")
        final_video = job_dir / f"avatar_video_{job_id}.mp4"

        if format_type == "debate" and script_b:
            # Debate: generate second avatar track
            progress(0.72, desc="Generating debate speaker B...")
            avatar_b_img = get_avatar_path(persona_b_id, pose, attire_b)
            if not avatar_b_img:
                avatar_b_img = avatar_img  # fallback to same avatar

            profile_b = AVATARS.get(persona_b_id, AVATARS[persona_id])["voice_profile"]
            audio_b   = job_dir / "voice_b.wav"
            synthesize(script_b, profile_b, audio_b)

            anim_b    = job_dir / "animated_b.mp4"
            lipsync_b = job_dir / "lipsync_b.mp4"
            animate(avatar_b_img, audio_b, anim_b, pose=pose)
            apply_lipsync(anim_b, audio_b, lipsync_b)

            name_b = AVATARS.get(persona_b_id, {}).get("name", "Speaker B")
            ok = compose_debate(
                lipsync_video, lipsync_b,
                avatar_name or AVATARS[persona_id]["name"], name_b,
                audio_final, audio_b,
                scene_key, final_video,
                agenda_items=agenda,
            )
        else:
            ok = compose_video(
                animated_video=lipsync_video,
                audio_path=audio_final,
                scene_key=scene_key,
                out_path=final_video,
                pose=pose,
                lower_third_name=avatar_name or AVATARS[persona_id]["name"],
                lower_third_title=avatar_title,
                agenda_items=agenda,
                show_agenda=show_agenda,
                export_vertical=export_vertical,
            )

        if not ok or not final_video.exists():
            return "❌ Compose step failed. Check logs/app.log.", None, None

        # ── Save script ───────────────────────────────────────────────────────
        script_file = job_dir / f"script_{job_id}.txt"
        script_file.write_text(approved_script, encoding="utf-8")

        size_mb = final_video.stat().st_size // (1024 * 1024)
        progress(1.0, desc="Done!")
        log.info(f"[{job_id}] Generation complete: {final_video} ({size_mb}MB)")

        return (
            f"✅ Video ready! Job: {job_id} | Size: {size_mb}MB | "
            f"Location: {final_video}",
            str(final_video),
            str(script_file),
        )

    except Exception as e:
        log.exception(f"[{job_id}] Generation crashed: {e}")
        return f"❌ Unexpected error: {e}\n\nCheck logs/app.log for details.", None, None
    finally:
        _gen_lock.release()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Avatar Studio",
        theme=gr.themes.Base(
            primary_hue="blue", neutral_hue="slate",
            font=["Inter", "sans-serif"],
        ),
        css="""
        .tab-nav button { font-size: 15px; font-weight: 600; }
        .generate-btn { background: #1565c0 !important; color: white !important;
                        font-size: 18px !important; height: 56px !important; }
        .status-box { font-family: monospace; }
        """,
    ) as app:

        gr.Markdown("""
        # 🎬 Avatar Studio
        **Flexible Human AI Video Generator** | Photorealistic South Indian Avatars | Multi-language | Half-body minimum
        """)

        # ── Shared state ──────────────────────────────────────────────────────
        state_script = gr.State("")
        state_scene  = gr.State("professional/office")
        state_format = gr.State("monologue")
        state_agenda = gr.State([])

        with gr.Tabs():

            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("1 · Content"):
                gr.Markdown("### Input your content source")
                with gr.Row():
                    with gr.Column(scale=2):
                        source_input = gr.Textbox(
                            label="YouTube URL  or  Paste Script / Topic",
                            placeholder="https://youtube.com/watch?v=... or paste text...",
                            lines=4,
                        )
                        topic_input = gr.Textbox(
                            label="Topic / Context (optional — helps script quality)",
                            placeholder="e.g. Google I/O 2025 AI announcements",
                        )
                    with gr.Column(scale=1):
                        lang_input    = gr.Dropdown(LANGUAGE_CHOICES, value="te", label="Language")
                        format_input  = gr.Dropdown(FORMAT_CHOICES + [("Auto Detect", "auto")],
                                                    value="auto", label="Video Format")
                        duration_input = gr.Slider(30, 600, value=180, step=30,
                                                   label="Target Duration (seconds)")

                analyse_btn = gr.Button("🔍 Analyse & Generate Script", variant="primary")
                status_1    = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")

            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("2 · Review Script"):
                gr.Markdown("### Review and edit the script before generating video")
                gr.Markdown("> ⚠️ This is your approval gate. Edit freely. Click **Approve** when ready.")
                script_box = gr.Textbox(label="Generated Script", lines=20,
                                        placeholder="Script will appear here after Step 1...")
                with gr.Row():
                    agenda_display = gr.JSON(label="Detected Agenda Items")
                    detected_info  = gr.Textbox(label="Detected Scene & Format",
                                                interactive=False)
                approve_btn = gr.Button("✅ Approve Script → Proceed to Generate", variant="primary")
                approved_status = gr.Textbox(label="", interactive=False)

            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("3 · Avatar"):
                gr.Markdown("### Choose your presenter")
                with gr.Row():
                    with gr.Column():
                        persona_input  = gr.Dropdown(AVATAR_CHOICES, value="priya_telugu_f",
                                                     label="Avatar Persona")
                        pose_input     = gr.Dropdown(POSE_CHOICES, value="half_body",
                                                     label="Body Pose")
                        attire_input   = gr.Dropdown(ATTIRE_CHOICES, value="professional",
                                                     label="Attire")
                        avatar_name_input  = gr.Textbox(label="Display Name (for lower third)",
                                                         placeholder="e.g. Priya Sharma")
                        avatar_title_input = gr.Textbox(label="Title / Role",
                                                         placeholder="e.g. AI Reporter")
                    with gr.Column():
                        gr.Markdown("#### Debate — Second Speaker (if debate format)")
                        persona_b_input = gr.Dropdown(AVATAR_CHOICES, value="arjun_telugu_m",
                                                      label="Debate Speaker B")
                        attire_b_input  = gr.Dropdown(ATTIRE_CHOICES, value="suit",
                                                      label="Speaker B Attire")
                        gr.Markdown("""
                        **Pose guide:**
                        - Half Body → torso + arms + hands visible (recommended)
                        - Standing → full body, best for seminar/stage
                        - Sitting at Desk → news anchor, KT session
                        """)

            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("4 · Scene"):
                gr.Markdown("### Choose background and scene settings")
                with gr.Row():
                    with gr.Column():
                        scene_input   = gr.Dropdown(SCENE_CHOICES,
                                                    value="professional/office",
                                                    label="Background Scene")
                        gr.Markdown("""
                        **Scene auto-switching:** When your script mentions Google, Apple,
                        Samsung, Red Fort etc., the background switches automatically per topic.
                        """)
                        show_agenda_cb = gr.Checkbox(value=True,
                                                     label="Show agenda card on screen")
                        export_vert_cb = gr.Checkbox(value=False,
                                                     label="Also export 9:16 vertical (Reels/Shorts)")
                    with gr.Column():
                        gr.Markdown("""
                        **Background categories:**
                        - **Professional:** Office, News Desk, Seminar Hall, Stage Dias
                        - **Nature:** Glacier, Beach, Forest, Mountain
                        - **Casual:** Kitchen, Bedroom, Cafe, Rooftop
                        - **Landmark:** Red Fort, Parliament, Market
                        - **Brand Stages:** Google, Apple, Samsung, TED
                        - **Custom:** Upload your own (coming Phase 2)
                        """)

            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("5 · Generate"):
                gr.Markdown("### Generate your video")
                gr.Markdown("> Ensure you have approved the script in Tab 2 before generating.")

                gen_btn = gr.Button("🚀 Generate Video", variant="primary",
                                    elem_classes="generate-btn")

                with gr.Row():
                    status_out  = gr.Textbox(label="Generation Status", lines=3,
                                             interactive=False, elem_classes="status-box")

                with gr.Row():
                    video_out  = gr.Video(label="Generated Video", height=480)
                    script_out = gr.File(label="Download Script (.txt)")

                gr.Markdown("""
                **Generation time estimates (T4 GPU):**
                | Duration | Time |
                |---|---|
                | 30 sec short | ~5 min |
                | 3 min KT | ~20 min |
                | 8 min review | ~40 min |
                | 15 min debate | ~80 min |
                """)

        # ── Wire events ───────────────────────────────────────────────────────

        analyse_btn.click(
            fn=analyse_content,
            inputs=[source_input, lang_input, format_input, duration_input, topic_input],
            outputs=[status_1, script_box, state_scene, state_format, state_agenda],
        ).then(
            fn=lambda scene, fmt, agenda: (
                SCENES.get(scene, {}).get("label", scene) + " | Format: " + fmt,
                agenda,
                scene,
            ),
            inputs=[state_scene, state_format, state_agenda],
            outputs=[detected_info, agenda_display, scene_input],
        )

        approve_btn.click(
            fn=lambda s: (s, "✅ Script approved. Go to Generate tab."),
            inputs=[script_box],
            outputs=[state_script, approved_status],
        )

        gen_btn.click(
            fn=generate_video,
            inputs=[
                state_script, lang_input,
                persona_input, pose_input, attire_input,
                scene_input, state_format,
                avatar_name_input, avatar_title_input,
                show_agenda_cb, state_agenda,
                export_vert_cb,
                persona_b_input, attire_b_input,
            ],
            outputs=[status_out, video_out, script_out],
        )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Avatar Studio...")
    log.info(f"Outputs: {OUTPUTS_DIR}")

    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",   # Nginx proxies — do NOT bind to 0.0.0.0
        server_port=7860,
        share=False,
        show_error=True,
        max_threads=2,             # GPU can only handle 1 job; 2 for UI responsiveness
    )
