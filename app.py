import os
import csv
import html
import json
import queue
import threading
import time

import av
import numpy as np
import streamlit as st
import websocket
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


# ============================================================
# Settings
# ============================================================

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
DEFAULT_TERMS_FILE = "technical_terms.csv"

DEFAULT_RESET_SECONDS = 3.0
MAX_ORIGINAL_CHARS = 160
MAX_TRANSLATION_CHARS = 260
MAX_HISTORY_ITEMS = 5
MAX_DEBUG_MESSAGES = 10


# ============================================================
# Glossary
# ============================================================

def load_soniox_context_terms(terms_file):
    terms = []
    translation_terms = []

    try:
        with open(terms_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                jp = row.get("jp", "").strip()
                en = row.get("en", "").strip()
                reading = row.get("reading", "").strip()
                common_wrong = row.get("common_wrong", "").strip()

                if jp:
                    terms.append(jp)

                if reading:
                    terms.append(reading)

                if common_wrong:
                    terms.extend([
                        item.strip()
                        for item in common_wrong.split(";")
                        if item.strip()
                    ])

                if jp and en:
                    translation_terms.append({
                        "source": jp,
                        "target": en,
                    })

        terms = list(dict.fromkeys(terms))

        unique_translation_terms = []
        seen = set()

        for item in translation_terms:
            key = (item["source"], item["target"])
            if key not in seen:
                unique_translation_terms.append(item)
                seen.add(key)

        return terms[:300], unique_translation_terms[:300]

    except Exception:
        return [], []


def light_caption_cleanup(text):
    if not text:
        return ""

    cleaned = text.strip()

    replacements = {
        "servo-motor": "servo motor",
        "servomotor": "servo motor",
        "brake force": "braking force",
        "braking power": "braking force",
        "sudden braking": "emergency braking",
        "sudden brake": "emergency braking",
        "restraints": "constraint condition",
        "restraint": "constraint condition",
        "modifier": "jig",
        "fixture": "jig",
        "quality management": "quality control",
        "bad product": "defective product",
    }

    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)

    cleaned = cleaned.replace("a obstacle", "an obstacle")
    cleaned = cleaned.replace("an inertia compensation", "inertia compensation")
    cleaned = cleaned.replace("a inertia compensation", "inertia compensation")

    return cleaned.strip()


def trim_caption_soft(text, max_chars):
    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    recent = text[-max_chars:]

    separators = [". ", "? ", "! ", "。", "、", ", ", " "]
    best_index = -1

    for sep in separators:
        index = recent.find(sep)
        if index > best_index:
            best_index = index + len(sep)

    if best_index > 0 and best_index < len(recent) - 5:
        recent = recent[best_index:]

    return recent.strip()


# ============================================================
# Audio processor
# ============================================================

class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=48000,
        )

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            resampled_frames = self.resampler.resample(frame)

            for resampled_frame in resampled_frames:
                audio = resampled_frame.to_ndarray().reshape(-1)

                if audio.size == 0:
                    continue

                pcm16 = audio.astype(np.int16)
                self.audio_queue.put(pcm16.tobytes())

        except Exception:
            pass

        return frame


# ============================================================
# Soniox worker
# ============================================================

def soniox_live_worker(
    audio_queue,
    result_queue,
    stop_event,
    control_queue,
    api_key,
    terms_file,
    domain_mode,
    caption_reset_seconds,
):
    ws = None

    try:
        context_terms, translation_terms = load_soniox_context_terms(terms_file)

        if domain_mode == "auto":
            domain_text = (
                "Japanese automotive engineering, control engineering, CAD, "
                "manufacturing, classroom interpretation, technical terms"
            )
        else:
            domain_text = f"Japanese {domain_mode} technical class interpretation"

        config = {
            "api_key": api_key,
            "model": "stt-rt-v5",
            "audio_format": "s16le",
            "sample_rate": 48000,
            "num_channels": 1,
            "language_hints": ["ja"],
            "enable_language_identification": False,
            "enable_endpoint_detection": True,
            "max_endpoint_delay_ms": 800,
            "context": {
                "general": [
                    {
                        "key": "domain",
                        "value": domain_text,
                    },
                    {
                        "key": "task",
                        "value": (
                            "Translate Japanese technical classroom speech "
                            "into clear English subtitles for an interpreter."
                        ),
                    },
                    {
                        "key": "style",
                        "value": (
                            "Use short, readable English captions. "
                            "Preserve technical terms accurately."
                        ),
                    },
                ],
                "terms": context_terms,
                "translation_terms": translation_terms,
            },
            "translation": {
                "type": "one_way",
                "target_language": "en",
            },
        }

        ws = websocket.create_connection(SONIOX_WS_URL, timeout=10)
        ws.send(json.dumps(config))

        result_queue.put({
            "type": "debug",
            "message": "Connected to Soniox.",
        })

        final_original = ""
        final_translation = ""
        last_token_time = time.time()
        current_reset_seconds = float(caption_reset_seconds)

        def send_audio():
            while not stop_event.is_set():
                try:
                    audio_bytes = audio_queue.get(timeout=0.1)

                    if audio_bytes:
                        ws.send_binary(audio_bytes)

                except queue.Empty:
                    continue

                except Exception as e:
                    if not stop_event.is_set():
                        result_queue.put({
                            "type": "error",
                            "message": f"Audio send error: {e}",
                        })
                    break

            try:
                ws.send_binary(b"")
            except Exception:
                pass

        sender_thread = threading.Thread(target=send_audio, daemon=True)
        sender_thread.start()

        while not stop_event.is_set():
            while control_queue is not None and not control_queue.empty():
                try:
                    command = control_queue.get_nowait()

                    if command == "clear":
                        final_original = ""
                        final_translation = ""
                        result_queue.put({"type": "cleared"})

                    elif isinstance(command, dict):
                        if command.get("type") == "set_reset_seconds":
                            current_reset_seconds = float(
                                command.get("value", current_reset_seconds)
                            )
                            result_queue.put({
                                "type": "debug",
                                "message": f"Reset seconds changed to {current_reset_seconds}",
                            })

                except queue.Empty:
                    break

            try:
                msg = ws.recv()

            except websocket.WebSocketTimeoutException:
                continue

            except Exception as e:
                if not stop_event.is_set():
                    result_queue.put({
                        "type": "error",
                        "message": f"WebSocket receive error: {e}",
                    })
                break

            if not msg:
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("error_code"):
                result_queue.put({
                    "type": "error",
                    "message": data.get("error_message", "Unknown Soniox error"),
                })
                break

            if data.get("finished"):
                break

            tokens = data.get("tokens", [])

            has_real_token = any(
                token.get("text", "") and token.get("text", "") != "<end>"
                for token in tokens
            )

            if has_real_token:
                now = time.time()

                if now - last_token_time > current_reset_seconds:
                    final_original = ""
                    final_translation = ""

                    # Timer reset: clear current subtitle page only.
                    # History should stay.
                    result_queue.put({"type": "page_reset"})

                last_token_time = now

            non_final_original = ""
            non_final_translation = ""
            endpoint_detected = False

            for token in tokens:
                text = token.get("text", "")

                if not text:
                    continue

                if text == "<end>":
                    endpoint_detected = True
                    continue

                status = token.get("translation_status")
                is_final = token.get("is_final", False)
                is_translation_token = status in ["translation", "translated"]

                if is_translation_token:
                    if is_final:
                        final_translation += text
                    else:
                        non_final_translation += text
                else:
                    if is_final:
                        final_original += text
                    else:
                        non_final_original += text

            current_original = (final_original + non_final_original).strip()
            current_translation = (final_translation + non_final_translation).strip()
            current_translation = light_caption_cleanup(current_translation)

            if current_original or current_translation:
                result_queue.put({
                    "type": "tokens",
                    "original": current_original,
                    "translation": current_translation,
                    "endpoint": endpoint_detected,
                })

    except Exception as e:
        if not stop_event.is_set():
            result_queue.put({
                "type": "error",
                "message": str(e),
            })

    finally:
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass

        result_queue.put({
            "type": "stopped",
        })


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Technical Interpreter Captioner",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }

        h1 {
            font-size: 42px !important;
            line-height: 1.1 !important;
        }

        h2, h3 {
            margin-top: 0.8rem !important;
        }

        div[data-testid="stAlert"] {
            padding: 0.75rem 1rem;
        }

        @media screen and (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            h1 {
                font-size: 30px !important;
                line-height: 1.05 !important;
            }

            h2 {
                font-size: 25px !important;
            }

            h3 {
                font-size: 22px !important;
            }

            p {
                font-size: 14px !important;
            }

            div[data-testid="stVerticalBlock"] {
                gap: 0.55rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Technical Interpreter Captioner")

st.caption(
    "Japanese → English live captions using Soniox real-time translation "
    "and a technical glossary."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Settings")

    domain_mode = st.selectbox(
        "Technical domain",
        ["auto", "automotive", "control", "cad", "manufacturing"],
        index=0,
    )

    subtitle_display = st.radio(
        "Caption display",
        ["Latest only", "History"],
        index=0,
    )

    font_size = st.slider(
        "English caption font size",
        min_value=18,
        max_value=44,
        value=26,
        step=2,
    )

    jp_font_size = st.slider(
        "Japanese original font size",
        min_value=14,
        max_value=34,
        value=19,
        step=1,
    )

    reset_seconds = st.slider(
        "Reset caption after pause",
        min_value=1.5,
        max_value=8.0,
        value=DEFAULT_RESET_SECONDS,
        step=0.5,
    )

    show_debug = st.checkbox(
        "Show debug panel",
        value=False,
    )

    st.caption(
        "Font size changes immediately. Reset seconds also updates while running."
    )

    st.divider()

    st.write("Glossary")

    terms_file = DEFAULT_TERMS_FILE

    uploaded_glossary = st.file_uploader(
        "Upload custom glossary CSV",
        type=["csv"],
    )

    if uploaded_glossary is not None:
        os.makedirs("glossaries", exist_ok=True)

        glossary_path = os.path.join("glossaries", uploaded_glossary.name)

        with open(glossary_path, "wb") as f:
            f.write(uploaded_glossary.getbuffer())

        terms_file = glossary_path
        st.success(f"Using: {uploaded_glossary.name}")
    else:
        st.info("Using default technical_terms.csv")

    context_terms, translation_terms = load_soniox_context_terms(terms_file)

    st.caption(
        f"Loaded {len(context_terms)} context terms and "
        f"{len(translation_terms)} translation terms."
    )


# ============================================================
# Session state
# ============================================================

defaults = {
    "app_active": False,
    "pending_start_translation": False,

    # Important for phone mic release:
    # Changing this destroys and recreates the WebRTC component.
    "mic_instance_id": 0,

    "live_original": "",
    "live_translation": "",
    "caption_history": [],
    "soniox_running": False,
    "soniox_error": "",
    "soniox_result_queue": queue.Queue(),
    "soniox_control_queue": queue.Queue(),
    "soniox_stop_event": threading.Event(),
    "soniox_thread": None,
    "debug_messages": [],
    "last_update_time": "",
    "last_reset_seconds": DEFAULT_RESET_SECONDS,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Update reset seconds while running.
if float(reset_seconds) != float(st.session_state.last_reset_seconds):
    st.session_state.last_reset_seconds = float(reset_seconds)

    if st.session_state.soniox_running:
        st.session_state.soniox_control_queue.put({
            "type": "set_reset_seconds",
            "value": float(reset_seconds),
        })


# ============================================================
# API key
# ============================================================

api_key = st.secrets.get("SONIOX_API_KEY", os.getenv("SONIOX_API_KEY"))

if not api_key:
    st.error(
        "SONIOX_API_KEY is not set.\n\n"
        "For Streamlit Cloud, add this in Secrets:\n\n"
        'SONIOX_API_KEY = "your_api_key_here"'
    )
    st.stop()


# ============================================================
# Microphone / WebRTC
# ============================================================

rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                    "stun:stun2.l.google.com:19302",
                    "stun:stun3.l.google.com:19302",
                    "stun:stun4.l.google.com:19302",
                ]
            }
        ]
    }
)

st.subheader("Microphone")

st.info(
    "Press Start Translation to start the microphone and translation. "
    "If microphone access is denied, allow microphone permission in browser site settings."
)

webrtc_ctx = webrtc_streamer(
    key=f"soniox-live-caption-mic-{st.session_state.mic_instance_id}",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": False,
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
    },
    audio_processor_factory=AudioProcessor,
    async_processing=True,
    desired_playing_state=st.session_state.app_active,
)


# ============================================================
# Controls
# ============================================================

toggle_label = (
    "Stop Translation"
    if st.session_state.app_active
    else "Start Translation"
)

toggle_clicked = st.button(
    toggle_label,
    type="primary",
    use_container_width=True,
)

clear_clicked = st.button(
    "Clear Captions",
    use_container_width=True,
)

if toggle_clicked:
    if st.session_state.app_active:
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.soniox_running = False
        st.session_state.soniox_stop_event.set()

        # Force WebRTC component to be destroyed and recreated.
        # This helps phone browsers release the microphone.
        st.session_state.mic_instance_id += 1

        st.rerun()

    else:
        st.session_state.app_active = True
        st.session_state.pending_start_translation = True
        st.session_state.soniox_error = ""

        st.rerun()

if clear_clicked:
    st.session_state.live_original = ""
    st.session_state.live_translation = ""
    st.session_state.caption_history = []
    st.session_state.last_update_time = ""
    st.session_state.soniox_error = ""

    if st.session_state.soniox_running:
        st.session_state.soniox_control_queue.put("clear")


# ============================================================
# Auto-start Soniox after WebRTC mic is ready
# ============================================================

if (
    st.session_state.pending_start_translation
    and st.session_state.app_active
    and not st.session_state.soniox_running
    and webrtc_ctx.audio_processor
):
    st.session_state.soniox_stop_event = threading.Event()
    st.session_state.soniox_result_queue = queue.Queue()
    st.session_state.soniox_control_queue = queue.Queue()
    st.session_state.soniox_error = ""
    st.session_state.debug_messages = []
    st.session_state.live_original = ""
    st.session_state.live_translation = ""
    st.session_state.caption_history = []
    st.session_state.last_update_time = ""

    processor = webrtc_ctx.audio_processor

    st.session_state.soniox_running = True
    st.session_state.pending_start_translation = False

    st.session_state.soniox_thread = threading.Thread(
        target=soniox_live_worker,
        args=(
            processor.audio_queue,
            st.session_state.soniox_result_queue,
            st.session_state.soniox_stop_event,
            st.session_state.soniox_control_queue,
            api_key,
            terms_file,
            domain_mode,
            float(reset_seconds),
        ),
        daemon=True,
    )

    st.session_state.soniox_thread.start()


# ============================================================
# Pull Soniox results into UI state
# ============================================================

while not st.session_state.soniox_result_queue.empty():
    item = st.session_state.soniox_result_queue.get()
    item_type = item.get("type")

    if item_type == "tokens":
        original = item.get("original", "")
        translation = item.get("translation", "")

        if original:
            st.session_state.live_original = original

        if translation:
            st.session_state.live_translation = translation

            if (
                not st.session_state.caption_history
                or st.session_state.caption_history[-1] != translation
            ):
                st.session_state.caption_history.append(translation)
                st.session_state.caption_history = (
                    st.session_state.caption_history[-MAX_HISTORY_ITEMS:]
                )

        st.session_state.last_update_time = time.strftime("%H:%M:%S")

    elif item_type == "page_reset":
        # Timer reset: clear current page only.
        # Do NOT clear caption_history.
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.last_update_time = ""

    elif item_type == "cleared":
        # Manual clear: clear everything.
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.caption_history = []
        st.session_state.last_update_time = ""

    elif item_type == "debug":
        message = item.get("message", "")
        if message:
            st.session_state.debug_messages.append(message)
            st.session_state.debug_messages = (
                st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
            )

    elif item_type == "error":
        st.session_state.soniox_error = item.get("message", "")
        st.session_state.soniox_running = False
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.mic_instance_id += 1

    elif item_type == "stopped":
        st.session_state.soniox_running = False


# ============================================================
# Status
# ============================================================

if st.session_state.soniox_running:
    st.success("Live translation running.")
elif st.session_state.app_active:
    st.info("Starting microphone...")
else:
    st.info("Live translation stopped.")

if st.session_state.soniox_error:
    st.error(st.session_state.soniox_error)


# ============================================================
# Debug panel
# ============================================================

if show_debug:
    with st.expander("Debug", expanded=True):
        st.write("Last update:")
        st.code(
            st.session_state.last_update_time
            if st.session_state.last_update_time
            else "No token update yet"
        )

        st.write("Japanese:")
        st.code(
            st.session_state.live_original
            if st.session_state.live_original
            else "Empty"
        )

        st.write("English:")
        st.code(
            st.session_state.live_translation
            if st.session_state.live_translation
            else "Empty"
        )

        st.write("History:")
        st.write(st.session_state.caption_history)

        st.write("Mic instance:")
        st.code(str(st.session_state.mic_instance_id))

        st.write("Error:")
        st.code(
            st.session_state.soniox_error
            if st.session_state.soniox_error
            else "No error"
        )

        if st.session_state.debug_messages:
            st.write("Messages:")
            for message in st.session_state.debug_messages:
                st.write("- " + str(message))


# ============================================================
# Caption display
# ============================================================

st.subheader("Live Captions")

if subtitle_display == "History":
    caption_text = "\n\n".join(st.session_state.caption_history[-MAX_HISTORY_ITEMS:])
else:
    caption_text = st.session_state.live_translation

display_japanese = trim_caption_soft(
    st.session_state.live_original,
    max_chars=MAX_ORIGINAL_CHARS,
)

english_max_chars = (
    MAX_TRANSLATION_CHARS * 2
    if subtitle_display == "History"
    else MAX_TRANSLATION_CHARS
)

display_english = trim_caption_soft(
    caption_text,
    max_chars=english_max_chars,
)

safe_original = html.escape(display_japanese)
safe_caption_text = html.escape(display_english)

caption_html = f"""
<style>
.caption-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 8px;
}}

.caption-label {{
    font-size: 14px;
    opacity: 0.75;
    margin-bottom: 5px;
    font-weight: 700;
}}

.jp-caption-box {{
    font-size: {jp_font_size}px;
    line-height: 1.35;
    padding: 12px;
    border-radius: 14px;
    background-color: #F3F4F6;
    color: #111827;
    min-height: 70px;
    max-height: 110px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #D1D5DB;
    box-sizing: border-box;
}}

.en-caption-box {{
    font-size: {font_size}px;
    line-height: 1.25;
    font-weight: 700;
    padding: 16px;
    border-radius: 18px;
    background-color: #111827;
    color: white;
    min-height: 125px;
    max-height: 230px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #374151;
    box-sizing: border-box;
}}

@media screen and (max-width: 768px) {{
    .caption-wrapper {{
        gap: 8px;
    }}

    .caption-label {{
        font-size: 12px;
        margin-bottom: 4px;
    }}

    .jp-caption-box {{
        font-size: {jp_font_size}px;
        line-height: 1.35;
        padding: 9px;
        min-height: 55px;
        max-height: 90px;
    }}

    .en-caption-box {{
        font-size: {font_size}px;
        line-height: 1.25;
        padding: 12px;
        min-height: 115px;
        max-height: 210px;
    }}
}}
</style>

<div class="caption-wrapper">
    <div>
        <div class="caption-label">Japanese Original</div>
        <div class="jp-caption-box">{safe_original}</div>
    </div>

    <div>
        <div class="caption-label">English Caption</div>
        <div class="en-caption-box">{safe_caption_text}</div>
    </div>
</div>
"""

st.html(caption_html)


# ============================================================
# Live refresh
# ============================================================

if st.session_state.app_active or st.session_state.soniox_running:
    time.sleep(0.7)
    st.rerun()