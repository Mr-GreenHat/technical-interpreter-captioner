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

SOURCE_LANGUAGE = "Japanese"
TARGET_LANGUAGE = "English"

MAX_HISTORY_ITEMS = 5
MAX_RAW_MESSAGES = 5
MAX_DEBUG_MESSAGES = 20


# ============================================================
# Glossary / technical terms
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
                    wrong_items = [
                        item.strip()
                        for item in common_wrong.split(";")
                        if item.strip()
                    ]
                    terms.extend(wrong_items)

                if jp and en:
                    translation_terms.append({
                        "source": jp,
                        "target": en,
                    })

        terms = list(dict.fromkeys(terms))

        translation_terms_unique = []
        seen_pairs = set()

        for item in translation_terms:
            key = (item["source"], item["target"])
            if key not in seen_pairs:
                translation_terms_unique.append(item)
                seen_pairs.add(key)

        return terms[:300], translation_terms_unique[:300]

    except FileNotFoundError:
        return [], []

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


# ============================================================
# WebRTC audio processor
# ============================================================

class AudioProcessor:
    """
    Receives browser microphone audio continuously.
    Resamples WebRTC audio to mono 48 kHz signed 16-bit PCM for Soniox.
    """

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
                audio = resampled_frame.to_ndarray()
                audio = audio.reshape(-1)

                if audio.size == 0:
                    continue

                pcm16 = audio.astype(np.int16)
                self.audio_queue.put(pcm16.tobytes())

        except Exception:
            pass

        return frame


# ============================================================
# Soniox live worker
# ============================================================

def soniox_live_worker(
    audio_queue,
    result_queue,
    stop_event,
    control_queue,
    api_key,
    terms_file,
    domain_mode,
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

        result_queue.put({
            "type": "debug",
            "message": "Connecting to Soniox WebSocket...",
        })

        ws = websocket.create_connection(
            SONIOX_WS_URL,
            timeout=10,
        )

        result_queue.put({
            "type": "debug",
            "message": "Connected to Soniox WebSocket.",
        })

        ws.send(json.dumps(config))

        result_queue.put({
            "type": "debug",
            "message": "Sent Soniox config.",
        })

        final_original = ""
        final_translation = ""

        def send_audio():
            while not stop_event.is_set():
                try:
                    audio_bytes = audio_queue.get(timeout=0.1)

                    if audio_bytes:
                        ws.send_binary(audio_bytes)

                        result_queue.put({
                            "type": "audio",
                            "bytes": len(audio_bytes),
                        })

                except queue.Empty:
                    continue

                except Exception as e:
                    result_queue.put({
                        "type": "error",
                        "message": f"Audio send error: {e}",
                    })
                    break

            try:
                ws.send_binary(b"")
            except Exception:
                pass

        sender_thread = threading.Thread(
            target=send_audio,
            daemon=True,
        )
        sender_thread.start()

        while not stop_event.is_set():
            try:
                msg = ws.recv()

            except websocket.WebSocketTimeoutException:
                continue

            except Exception as e:
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
                result_queue.put({
                    "type": "debug",
                    "message": "Received non-JSON message from Soniox.",
                })
                continue

            result_queue.put({
                "type": "raw",
                "message": data,
            })

            if data.get("error_code"):
                result_queue.put({
                    "type": "error",
                    "message": data.get("error_message", "Unknown Soniox error"),
                })
                break

            if data.get("finished"):
                result_queue.put({
                    "type": "status",
                    "message": "Soniox stream finished.",
                })
                break

            # Handle UI control commands while live translation is running.
            while control_queue is not None and not control_queue.empty():
                try:
                    command = control_queue.get_nowait()

                    if command == "clear":
                        final_original = ""
                        final_translation = ""

                        result_queue.put({
                            "type": "cleared",
                        })

                except queue.Empty:
                    break

            tokens = data.get("tokens", [])

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
            "type": "status",
            "message": "Stopped.",
        })


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Technical Interpreter Captioner",
    layout="wide",
)

st.title("Technical Interpreter Captioner")

st.caption(
    "Japanese → English live captions using Soniox real-time translation "
    "and a technical glossary."
)


# ============================================================
# Sidebar settings
# ============================================================

with st.sidebar:
    st.header("Settings")

    domain_mode = st.selectbox(
        "Technical domain",
        [
            "auto",
            "automotive",
            "control",
            "cad",
            "manufacturing",
        ],
        index=0,
    )

    subtitle_display = st.radio(
        "Caption display",
        [
            "Latest only",
            "History",
        ],
        index=0,
    )

    show_debug = st.checkbox(
        "Show debug panel",
        value=False,
    )

    font_size = st.slider(
        "English caption font size",
        min_value=20,
        max_value=56,
        value=30,
        step=2,
    )

    jp_font_size = st.slider(
        "Japanese original font size",
        min_value=16,
        max_value=40,
        value=22,
        step=2,
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

        glossary_path = os.path.join(
            "glossaries",
            uploaded_glossary.name,
        )

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

if "live_original" not in st.session_state:
    st.session_state.live_original = ""

if "live_translation" not in st.session_state:
    st.session_state.live_translation = ""

if "caption_history" not in st.session_state:
    st.session_state.caption_history = []

if "soniox_running" not in st.session_state:
    st.session_state.soniox_running = False

if "soniox_error" not in st.session_state:
    st.session_state.soniox_error = ""

if "soniox_result_queue" not in st.session_state:
    st.session_state.soniox_result_queue = queue.Queue()

if "soniox_control_queue" not in st.session_state:
    st.session_state.soniox_control_queue = queue.Queue()

if "soniox_stop_event" not in st.session_state:
    st.session_state.soniox_stop_event = threading.Event()

if "soniox_thread" not in st.session_state:
    st.session_state.soniox_thread = None

if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []

if "audio_bytes_sent" not in st.session_state:
    st.session_state.audio_bytes_sent = 0

if "soniox_raw_messages" not in st.session_state:
    st.session_state.soniox_raw_messages = []

if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = ""


# ============================================================
# API key
# ============================================================

api_key = st.secrets.get("SONIOX_API_KEY", os.getenv("SONIOX_API_KEY"))

if not api_key:
    st.error(
        "SONIOX_API_KEY is not set.\n\n"
        "For Streamlit Cloud, add this in Secrets:\n\n"
        'SONIOX_API_KEY = "your_api_key_here"\n\n'
        "For local PowerShell, run:\n\n"
        'setx SONIOX_API_KEY "your_api_key_here"\n\n'
        "Then close PowerShell and open it again."
    )
    st.stop()


# ============================================================
# WebRTC microphone
# ============================================================

rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)

st.subheader("Microphone")

webrtc_ctx = webrtc_streamer(
    key="soniox-live-caption-mic",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": False,
        "audio": True,
    },
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)


# ============================================================
# Controls
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    start_clicked = st.button(
        "Start Translation",
        type="primary",
        use_container_width=True,
    )

with col2:
    stop_clicked = st.button(
        "Stop",
        use_container_width=True,
    )

with col3:
    clear_clicked = st.button(
        "Clear Captions",
        use_container_width=True,
    )

if clear_clicked:
    st.session_state.live_original = ""
    st.session_state.live_translation = ""
    st.session_state.caption_history = []
    st.session_state.soniox_error = ""
    st.session_state.debug_messages = []
    st.session_state.audio_bytes_sent = 0
    st.session_state.soniox_raw_messages = []
    st.session_state.last_update_time = ""

    if st.session_state.soniox_running:
        st.session_state.soniox_control_queue.put("clear")

if stop_clicked:
    st.session_state.soniox_running = False
    st.session_state.soniox_stop_event.set()

if start_clicked:
    if not webrtc_ctx.audio_processor:
        st.warning("Start the microphone first, then click Start Translation.")

    else:
        st.session_state.soniox_stop_event = threading.Event()
        st.session_state.soniox_result_queue = queue.Queue()
        st.session_state.soniox_control_queue = queue.Queue()
        st.session_state.soniox_error = ""
        st.session_state.debug_messages = []
        st.session_state.audio_bytes_sent = 0
        st.session_state.soniox_raw_messages = []
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.caption_history = []
        st.session_state.last_update_time = ""
        st.session_state.soniox_running = True

        processor = webrtc_ctx.audio_processor

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
            ),
            daemon=True,
        )

        st.session_state.soniox_thread.start()


# ============================================================
# Pull Soniox results into UI state
# ============================================================

result_queue = st.session_state.soniox_result_queue

while not result_queue.empty():
    item = result_queue.get()

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
                st.session_state.caption_history = st.session_state.caption_history[-MAX_HISTORY_ITEMS:]

        st.session_state.last_update_time = time.strftime("%H:%M:%S")

    elif item_type == "cleared":
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.caption_history = []
        st.session_state.last_update_time = ""

    elif item_type == "audio":
        st.session_state.audio_bytes_sent += item.get("bytes", 0)

    elif item_type == "raw":
        raw_message = item.get("message", {})

        st.session_state.soniox_raw_messages.append(raw_message)
        st.session_state.soniox_raw_messages = st.session_state.soniox_raw_messages[-MAX_RAW_MESSAGES:]

    elif item_type == "debug":
        message = item.get("message", "")
        if message:
            st.session_state.debug_messages.append(message)
            st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]

    elif item_type == "error":
        st.session_state.soniox_error = item.get("message", "")
        st.session_state.soniox_running = False

    elif item_type == "status":
        message = item.get("message", "")
        if message:
            st.session_state.debug_messages.append(message)
            st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]


# ============================================================
# Status
# ============================================================

if st.session_state.soniox_running:
    st.success("Live translation running.")
else:
    st.info("Live translation stopped.")

if st.session_state.soniox_error:
    st.error(st.session_state.soniox_error)


# ============================================================
# Debug panel
# ============================================================

if show_debug:
    with st.expander("Debug / Error Info", expanded=True):
        st.write("**Audio bytes sent to Soniox:**")
        st.code(str(st.session_state.audio_bytes_sent))

        st.write("**Last UI update time:**")
        st.code(
            st.session_state.last_update_time
            if st.session_state.last_update_time
            else "No token update yet"
        )

        st.write("**Current Japanese original:**")
        st.code(
            st.session_state.live_original
            if st.session_state.live_original
            else "Empty"
        )

        st.write("**Current English translation:**")
        st.code(
            st.session_state.live_translation
            if st.session_state.live_translation
            else "Empty"
        )

        st.write("**Soniox error:**")
        st.code(
            st.session_state.soniox_error
            if st.session_state.soniox_error
            else "No error"
        )

        st.write("**Recent debug messages:**")
        if st.session_state.debug_messages:
            for message in st.session_state.debug_messages:
                st.write("- " + str(message))
        else:
            st.write("No debug messages yet.")

        st.write("**Recent raw Soniox messages:**")
        if st.session_state.soniox_raw_messages:
            for msg in st.session_state.soniox_raw_messages:
                st.json(msg)
        else:
            st.write("No raw Soniox messages yet.")


# ============================================================
# Caption display
# ============================================================

st.subheader("Live Captions")

if subtitle_display == "History":
    caption_text = "\n".join(st.session_state.caption_history)
else:
    caption_text = st.session_state.live_translation

safe_caption_text = html.escape(caption_text)
safe_original = html.escape(st.session_state.live_original)

st.markdown(
    f"""
    <style>
        .caption-wrapper {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 8px;
        }}

        .caption-label {{
            font-size: 14px;
            opacity: 0.75;
            margin-bottom: 6px;
            font-weight: 700;
        }}

        .jp-caption-box {{
            font-size: {jp_font_size}px;
            line-height: 1.45;
            padding: 16px;
            border-radius: 14px;
            background-color: #F3F4F6;
            color: #111827;
            min-height: 70px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            border: 1px solid #D1D5DB;
        }}

        .en-caption-box {{
            font-size: {font_size}px;
            line-height: 1.3;
            font-weight: 700;
            padding: 22px;
            border-radius: 18px;
            background-color: #111827;
            color: white;
            min-height: 150px;
            max-height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
            border: 1px solid #374151;
        }}

        @media screen and (max-width: 768px) {{
            .caption-wrapper {{
                gap: 10px;
            }}

            .caption-label {{
                font-size: 13px;
                margin-bottom: 4px;
            }}

            .jp-caption-box {{
                font-size: 18px;
                line-height: 1.4;
                padding: 12px;
                min-height: 55px;
                max-height: 105px;
            }}

            .en-caption-box {{
                font-size: 24px;
                line-height: 1.28;
                padding: 16px;
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
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Force live UI refresh
# ============================================================

if st.session_state.soniox_running:
    time.sleep(0.3)
    st.rerun()