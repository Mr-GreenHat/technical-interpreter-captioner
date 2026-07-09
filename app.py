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

from google import genai
from google.genai import types


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

LLM_MODEL_DEFAULT = "gemini-3.1-flash-lite"
LLM_MODEL_BACKUP = "gemini-2.5-flash-lite"

DEFAULT_LLM_HINT_INTERVAL = 30.0
MIN_LLM_CONTEXT_CHARS = 160
MAX_LLM_CONTEXT_CHUNKS = 6


# ============================================================
# Secrets
# ============================================================

def safe_get_secret_or_env(key):
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None

    if not value:
        value = os.getenv(key)

    return value


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


def load_glossary_entries(terms_file):
    entries = []

    try:
        with open(terms_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                jp = row.get("jp", "").strip()
                en = row.get("en", "").strip()
                reading = row.get("reading", "").strip()
                common_wrong = row.get("common_wrong", "").strip()
                notes = row.get("notes", "").strip()
                domain = row.get("domain", "").strip()

                if not jp or not en:
                    continue

                entries.append({
                    "domain": domain,
                    "jp": jp,
                    "en": en,
                    "reading": reading,
                    "common_wrong": common_wrong,
                    "notes": notes,
                })

    except Exception:
        pass

    return entries


def extract_key_terms_for_llm(original_text, translation_text, terms_file, max_terms=8):
    original_text = original_text or ""
    translation_text = translation_text or ""
    translation_lower = translation_text.lower()

    entries = load_glossary_entries(terms_file)
    matched_terms = []

    for row in entries:
        jp = row["jp"]
        en = row["en"]
        reading = row.get("reading", "")
        common_wrong = row.get("common_wrong", "")
        notes = row.get("notes", "")

        candidates = [jp, en, reading]

        if common_wrong:
            candidates.extend([
                item.strip()
                for item in common_wrong.split(";")
                if item.strip()
            ])

        found = False

        for candidate in candidates:
            if not candidate:
                continue

            if candidate in original_text:
                found = True
                break

            if candidate.lower() in translation_lower:
                found = True
                break

        if found:
            matched_terms.append({
                "jp": jp,
                "en": en,
                "notes": notes,
            })

    unique_terms = []
    seen = set()

    for item in matched_terms:
        key = (item["jp"], item["en"])

        if key not in seen:
            unique_terms.append(item)
            seen.add(key)

    return unique_terms[:max_terms]


# ============================================================
# Cleanup and correction
# ============================================================

def apply_llm_corrections(text, corrections):
    if not text:
        return ""

    cleaned = text

    for item in corrections or []:
        wrong = str(item.get("wrong", "")).strip()
        correct = str(item.get("correct", "")).strip()

        if not wrong or not correct:
            continue

        # Avoid dangerous one-character replacements.
        if len(wrong) < 2:
            continue

        cleaned = cleaned.replace(wrong, correct)

    return cleaned.strip()


def light_caption_cleanup(text):
    if not text:
        return ""

    cleaned = text.strip()

    replacements = {
        # ====================================================
        # TTC correction
        # ====================================================
        "ABC is large enough": "TTC is large enough",
        "the ABC is large enough": "the TTC is large enough",
        "If the ABC is large enough": "If the TTC is large enough",
        "If ABC is large enough": "If TTC is large enough",
        "ABC value": "TTC value",
        "the ABC": "the TTC",
        "ABC": "TTC",
        "Time to Collision": "TTC",
        "time to collision": "TTC",

        # ====================================================
        # Strong correction for 慣性補償
        # ====================================================
        "sensory compensation control": "inertia compensation control",
        "sensitivity compensation control": "inertia compensation control",
        "sensibility compensation control": "inertia compensation control",
        "sensory compensation": "inertia compensation",
        "sensitivity compensation": "inertia compensation",
        "sensibility compensation": "inertia compensation",

        "completion assurance control": "inertia compensation control",
        "completion compensation control": "inertia compensation control",
        "complete assurance control": "inertia compensation control",
        "complete compensation control": "inertia compensation control",
        "control for the completion assurance": "inertia compensation control",
        "the control for the completion assurance": "inertia compensation control",
        "completion assurance": "inertia compensation",
        "completion compensation": "inertia compensation",
        "complete assurance": "inertia compensation",
        "complete compensation": "inertia compensation",

        "Today is sensory compensation": "Today, I will explain inertia compensation",
        "Today is inertia compensation": "Today, I will explain inertia compensation",
        "About control": "control",

        # ====================================================
        # General technical cleanup
        # ====================================================
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

    lower_replacements = {
        "abc is large enough": "TTC is large enough",
        "the abc is large enough": "the TTC is large enough",
        "if the abc is large enough": "If the TTC is large enough",
        "time to collision": "TTC",

        "sensory compensation control": "inertia compensation control",
        "sensitivity compensation control": "inertia compensation control",
        "sensibility compensation control": "inertia compensation control",
        "sensory compensation": "inertia compensation",
        "sensitivity compensation": "inertia compensation",
        "sensibility compensation": "inertia compensation",
        "completion assurance control": "inertia compensation control",
        "completion compensation control": "inertia compensation control",
        "completion assurance": "inertia compensation",
        "completion compensation": "inertia compensation",
    }

    for wrong, correct in lower_replacements.items():
        cleaned = cleaned.replace(wrong, correct)

    cleaned = cleaned.replace("a obstacle", "an obstacle")
    cleaned = cleaned.replace("an inertia compensation", "inertia compensation")
    cleaned = cleaned.replace("a inertia compensation", "inertia compensation")

    return cleaned.strip()


def light_original_cleanup(text):
    if not text:
        return ""

    cleaned = text.strip()

    replacements = {
        # TTC correction
        "ABC": "TTC",
        "エービーシー": "TTC",
        "エービーシーが": "TTCが",
        "ABCが": "TTCが",

        # 慣性補償 correction
        "感性補償": "慣性補償",
        "感性保証": "慣性補償",
        "感性保障": "慣性補償",
        "完成保証": "慣性補償",
        "完成補償": "慣性補償",
        "完成保障": "慣性補償",
        "慣性保障": "慣性補償",

        "感性補償制御": "慣性補償制御",
        "感性保証制御": "慣性補償制御",
        "完成保証制御": "慣性補償制御",
        "完成補償制御": "慣性補償制御",
        "慣性保障制御": "慣性補償制御",
    }

    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)

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
# LLM context helpers
# ============================================================

def make_context_chunk(original_text, translation_text):
    original_text = (original_text or "").strip()
    translation_text = (translation_text or "").strip()

    if not original_text and not translation_text:
        return ""

    return (
        f"Japanese: {original_text}\n"
        f"English: {translation_text}"
    ).strip()


def build_llm_context(context_chunks, current_original, current_translation):
    chunks = list(context_chunks or [])

    current_chunk = make_context_chunk(
        current_original,
        current_translation,
    )

    if current_chunk:
        chunks.append(current_chunk)

    chunks = chunks[-MAX_LLM_CONTEXT_CHUNKS:]

    return "\n\n---\n\n".join(chunks).strip()


# ============================================================
# LLM Interpreter Support
# ============================================================

def parse_llm_json(text):
    if not text:
        return {
            "main_idea": "",
            "say_it_simply": "",
            "key_terms": [],
            "corrections": [],
        }

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        data = json.loads(cleaned)

        return {
            "main_idea": str(data.get("main_idea", "")).strip(),
            "say_it_simply": str(data.get("say_it_simply", "")).strip(),
            "key_terms": data.get("key_terms", []),
            "corrections": data.get("corrections", []),
        }

    except Exception:
        return {
            "main_idea": cleaned[:220],
            "say_it_simply": "",
            "key_terms": [],
            "corrections": [],
        }


def llm_hint_worker(
    result_queue,
    api_key,
    model_name,
    context_text,
    current_translation,
    key_terms,
):
    try:
        client = genai.Client(api_key=api_key)

        glossary_text = ""

        if key_terms:
            glossary_lines = []

            for item in key_terms:
                jp = item.get("jp", "")
                en = item.get("en", "")
                notes = item.get("notes", "")

                if notes:
                    glossary_lines.append(f"- {jp} = {en} ({notes})")
                else:
                    glossary_lines.append(f"- {jp} = {en}")

            glossary_text = "\n".join(glossary_lines)

        prompt = f"""
You are an interpreter assistant.

Your job is NOT to translate everything again.
Your job is to help the interpreter understand the lecture flow quickly
AND repair obvious STT/translation mistakes in technical terms.

Use the recent context below. The latest part is at the bottom.

Rules:
- Output JSON only.
- Do not add new facts.
- Keep it short.
- Focus on the speaker's current main point, not every detail.
- Use previous context to understand what the speaker is talking about.
- Make the simple sentence useful for a human interpreter.
- Preserve technical terms from the glossary.
- If the transcript is unclear, give the safest interpretation.
- If STT or translation uses a wrong technical term, add it to corrections.
- If the caption says ABC but the context means TTC / Time To Collision, correct ABC to TTC.
- Prefer corrected technical terms in key_terms.
- Example correction:
  {{"wrong": "ABC", "correct": "TTC", "reason": "TTC means Time To Collision in AEB context"}}

Recent lecture context:
{context_text}

Current English translation:
{current_translation}

Technical glossary terms detected:
{glossary_text}

Return JSON in this exact format:
{{
  "main_idea": "one short sentence explaining the current main point",
  "say_it_simply": "one natural sentence the interpreter can say",
  "key_terms": [
    {{"term": "Japanese or English term", "meaning": "short meaning"}}
  ],
  "corrections": [
    {{
      "wrong": "wrong recognized word or phrase",
      "correct": "correct word or phrase",
      "reason": "short reason"
    }}
  ]
}}
""".strip()

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=340,
            ),
        )

        parsed = parse_llm_json(response.text)

        result_queue.put({
            "type": "llm_hint",
            "main_idea": parsed.get("main_idea", ""),
            "say_it_simply": parsed.get("say_it_simply", ""),
            "key_terms": parsed.get("key_terms", []),
            "corrections": parsed.get("corrections", []),
        })

    except Exception as e:
        result_queue.put({
            "type": "llm_error",
            "message": str(e),
        })


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
                "Japanese automotive engineering, CAD, product design, vehicle systems, "
                "braking systems, vehicle control, TTC, Time To Collision, AEB, "
                "inertia compensation, classroom interpretation, technical terms"
            )

        elif domain_mode == "automotive":
            domain_text = (
                "Japanese automotive engineering class, vehicle systems, braking systems, "
                "drivetrain, suspension, steering, ADAS, AEB, TTC, Time To Collision, "
                "vehicle control, inertia compensation"
            )

        elif domain_mode == "cad":
            domain_text = (
                "Japanese CAD class, sketch constraints, dimensions, extrusion, chamfering, "
                "fillet, technical drawing, projection drawing, product modeling"
            )

        elif domain_mode == "product design":
            domain_text = (
                "Japanese product design class, design process, CAD modeling, dimensions, "
                "materials, usability, product development, prototyping"
            )

        else:
            domain_text = "Japanese technical classroom interpretation"

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
                        "key": "important_term",
                        "value": (
                            "TTC means Time To Collision. "
                            "If speech sounds like ABC in AEB context, it is probably TTC. "
                            "慣性補償 means inertia compensation. "
                            "Do not translate 慣性補償 as sensory compensation, completion assurance, "
                            "or completion compensation."
                        ),
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
            current_original = light_original_cleanup(current_original)

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

st.title("Technical Interpreter Captioner")

st.caption(
    "Japanese → English live captions using Soniox real-time translation, "
    "technical glossary, and optional LLM interpreter support."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Settings")

    domain_mode = st.selectbox(
        "Technical domain",
        ["auto", "automotive", "cad", "product design"],
        index=0,
    )

    subtitle_display = st.radio(
        "Caption display",
        ["Latest only", "History"],
        index=0,
    )

    font_size = st.slider(
        "English caption font size",
        min_value=16,
        max_value=38,
        value=22,
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

    st.divider()

    st.write("LLM Interpreter Support")

    use_llm_hints = st.checkbox(
        "Use LLM support",
        value=False,
    )

    llm_model_name = st.selectbox(
        "LLM model",
        [
            LLM_MODEL_DEFAULT,
            LLM_MODEL_BACKUP,
        ],
        index=0,
    )

    llm_hint_interval = st.slider(
        "LLM update interval",
        min_value=15.0,
        max_value=90.0,
        value=DEFAULT_LLM_HINT_INTERVAL,
        step=5.0,
    )

    st.caption(
        "Soniox handles live translation. The LLM repairs obvious technical terms and creates a simple interpreter sentence."
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

    "llm_result_queue": queue.Queue(),
    "llm_thread": None,
    "llm_running": False,
    "llm_error": "",
    "llm_main_idea": "",
    "llm_say_it_simply": "",
    "llm_key_terms": [],
    "llm_corrections": [],
    "llm_last_call_time": 0.0,
    "llm_last_source_text": "",
    "llm_context_chunks": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


if float(reset_seconds) != float(st.session_state.last_reset_seconds):
    st.session_state.last_reset_seconds = float(reset_seconds)

    if st.session_state.soniox_running:
        st.session_state.soniox_control_queue.put({
            "type": "set_reset_seconds",
            "value": float(reset_seconds),
        })


# ============================================================
# API keys
# ============================================================

api_key = safe_get_secret_or_env("SONIOX_API_KEY")

if not api_key:
    st.error(
        "SONIOX_API_KEY is not set.\n\n"
        "For Streamlit Cloud, add this in Secrets:\n\n"
        'SONIOX_API_KEY = "your_soniox_api_key_here"'
    )
    st.stop()

gemini_api_key = safe_get_secret_or_env("GEMINI_API_KEY")

if use_llm_hints and not gemini_api_key:
    st.warning(
        "GEMINI_API_KEY is not set. LLM support is disabled until you add it."
    )


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

    st.session_state.llm_context_chunks = []
    st.session_state.llm_main_idea = ""
    st.session_state.llm_say_it_simply = ""
    st.session_state.llm_key_terms = []
    st.session_state.llm_corrections = []
    st.session_state.llm_error = ""
    st.session_state.llm_last_source_text = ""

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

    st.session_state.llm_context_chunks = []
    st.session_state.llm_main_idea = ""
    st.session_state.llm_say_it_simply = ""
    st.session_state.llm_key_terms = []
    st.session_state.llm_corrections = []
    st.session_state.llm_error = ""
    st.session_state.llm_last_source_text = ""
    st.session_state.llm_last_call_time = 0.0

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
        completed_chunk = make_context_chunk(
            st.session_state.live_original,
            st.session_state.live_translation,
        )

        if completed_chunk:
            if (
                not st.session_state.llm_context_chunks
                or st.session_state.llm_context_chunks[-1] != completed_chunk
            ):
                st.session_state.llm_context_chunks.append(completed_chunk)
                st.session_state.llm_context_chunks = (
                    st.session_state.llm_context_chunks[-MAX_LLM_CONTEXT_CHUNKS:]
                )

        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.last_update_time = ""

    elif item_type == "cleared":
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.caption_history = []
        st.session_state.last_update_time = ""
        st.session_state.llm_context_chunks = []
        st.session_state.llm_corrections = []

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
# Pull LLM results
# ============================================================

while not st.session_state.llm_result_queue.empty():
    item = st.session_state.llm_result_queue.get()
    item_type = item.get("type")

    if item_type == "llm_hint":
        st.session_state.llm_main_idea = item.get("main_idea", "")
        st.session_state.llm_say_it_simply = item.get("say_it_simply", "")
        st.session_state.llm_key_terms = item.get("key_terms", [])
        st.session_state.llm_corrections = item.get("corrections", [])
        st.session_state.llm_error = ""
        st.session_state.llm_running = False

    elif item_type == "llm_error":
        st.session_state.llm_error = item.get("message", "")
        st.session_state.llm_running = False


# ============================================================
# Start LLM hint worker
# ============================================================

if use_llm_hints and gemini_api_key:
    source_text = build_llm_context(
        st.session_state.llm_context_chunks,
        st.session_state.live_original,
        st.session_state.live_translation,
    )

    enough_text = len(source_text) >= MIN_LLM_CONTEXT_CHARS
    changed_text = source_text != st.session_state.llm_last_source_text
    interval_ready = (
        time.time() - float(st.session_state.llm_last_call_time)
        >= float(llm_hint_interval)
    )

    if (
        st.session_state.soniox_running
        and enough_text
        and changed_text
        and interval_ready
        and not st.session_state.llm_running
    ):
        detected_terms = extract_key_terms_for_llm(
            st.session_state.live_original,
            st.session_state.live_translation,
            terms_file,
        )

        st.session_state.llm_running = True
        st.session_state.llm_error = ""
        st.session_state.llm_last_call_time = time.time()
        st.session_state.llm_last_source_text = source_text

        st.session_state.llm_thread = threading.Thread(
            target=llm_hint_worker,
            args=(
                st.session_state.llm_result_queue,
                gemini_api_key,
                llm_model_name,
                source_text,
                st.session_state.live_translation,
                detected_terms,
            ),
            daemon=True,
        )

        st.session_state.llm_thread.start()


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

if use_llm_hints and st.session_state.llm_error:
    st.warning(f"LLM error: {st.session_state.llm_error}")


# ============================================================
# Caption display data
# ============================================================

st.subheader("Live Captions")

if subtitle_display == "History":
    caption_text = "\n\n".join(st.session_state.caption_history[-MAX_HISTORY_ITEMS:])
else:
    caption_text = st.session_state.live_translation

corrected_original = apply_llm_corrections(
    st.session_state.live_original,
    st.session_state.llm_corrections,
)

corrected_translation = apply_llm_corrections(
    caption_text,
    st.session_state.llm_corrections,
)

corrected_original = light_original_cleanup(corrected_original)
corrected_translation = light_caption_cleanup(corrected_translation)

display_japanese = trim_caption_soft(
    corrected_original,
    max_chars=MAX_ORIGINAL_CHARS,
)

english_max_chars = (
    MAX_TRANSLATION_CHARS * 2
    if subtitle_display == "History"
    else MAX_TRANSLATION_CHARS
)

display_english = trim_caption_soft(
    corrected_translation,
    max_chars=english_max_chars,
)

if use_llm_hints:
    if st.session_state.llm_running:
        simple_text = "Generating simple interpreter sentence..."
    elif st.session_state.llm_say_it_simply:
        simple_text = apply_llm_corrections(
            st.session_state.llm_say_it_simply,
            st.session_state.llm_corrections,
        )
    elif st.session_state.llm_main_idea:
        simple_text = apply_llm_corrections(
            st.session_state.llm_main_idea,
            st.session_state.llm_corrections,
        )
    else:
        simple_text = "Waiting for enough lecture context..."

    if st.session_state.llm_key_terms:
        llm_terms_lines = []

        for item in st.session_state.llm_key_terms[:5]:
            term = str(item.get("term", "")).strip()
            meaning = str(item.get("meaning", "")).strip()

            term = apply_llm_corrections(term, st.session_state.llm_corrections)
            meaning = apply_llm_corrections(meaning, st.session_state.llm_corrections)

            if term == "ABC":
                term = "TTC"
                if not meaning:
                    meaning = "Time To Collision"

            if term and meaning:
                llm_terms_lines.append(f"{term} = {meaning}")
            elif term:
                llm_terms_lines.append(term)

        llm_terms_text = "\n".join(llm_terms_lines)
    else:
        llm_terms_text = "No LLM key terms yet."

else:
    simple_text = ""
    llm_terms_text = ""

safe_original = html.escape(display_japanese)
safe_caption_text = html.escape(display_english)
safe_simple_text = html.escape(simple_text)
safe_llm_terms = html.escape(llm_terms_text)

llm_html = ""

if use_llm_hints:
    llm_html = f"""
    <div>
        <div class="caption-label">Say It Simply</div>
        <div class="llm-simple-box">{safe_simple_text}</div>
    </div>

    <div>
        <div class="caption-label">LLM Key Terms</div>
        <div class="llm-terms-box">{safe_llm_terms}</div>
    </div>
    """


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

        st.write("Japanese raw:")
        st.code(
            st.session_state.live_original
            if st.session_state.live_original
            else "Empty"
        )

        st.write("Japanese corrected:")
        st.code(corrected_original if corrected_original else "Empty")

        st.write("English raw:")
        st.code(
            st.session_state.live_translation
            if st.session_state.live_translation
            else "Empty"
        )

        st.write("English corrected:")
        st.code(corrected_translation if corrected_translation else "Empty")

        st.write("History:")
        st.write(st.session_state.caption_history)

        st.write("LLM context chunks:")
        st.write(st.session_state.llm_context_chunks)

        st.write("LLM:")
        st.code(
            f"enabled={use_llm_hints}\n"
            f"running={st.session_state.llm_running}\n"
            f"main_idea={st.session_state.llm_main_idea}\n"
            f"say_it_simply={st.session_state.llm_say_it_simply}\n"
            f"corrections={st.session_state.llm_corrections}\n"
            f"error={st.session_state.llm_error}"
        )

        st.write("Mic instance:")
        st.code(str(st.session_state.mic_instance_id))

        st.write("Soniox error:")
        st.code(
            st.session_state.soniox_error
            if st.session_state.soniox_error
            else "No error"
        )


# ============================================================
# Caption display
# ============================================================

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

.llm-simple-box {{
    font-size: 22px;
    line-height: 1.3;
    font-weight: 750;
    padding: 15px;
    border-radius: 16px;
    background-color: #DBEAFE;
    color: #1E3A8A;
    min-height: 75px;
    max-height: 140px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #3B82F6;
    box-sizing: border-box;
}}

.llm-terms-box {{
    font-size: 16px;
    line-height: 1.35;
    font-weight: 600;
    padding: 12px;
    border-radius: 14px;
    background-color: #ECFDF5;
    color: #064E3B;
    min-height: 55px;
    max-height: 130px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #10B981;
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
    min-height: 115px;
    max-height: 210px;
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

    .llm-simple-box {{
        font-size: 19px;
        line-height: 1.25;
        padding: 12px;
        min-height: 65px;
        max-height: 125px;
    }}

    .llm-terms-box {{
        font-size: 15px;
        line-height: 1.3;
        padding: 10px;
        min-height: 50px;
        max-height: 115px;
    }}

    .en-caption-box {{
        font-size: {font_size}px;
        line-height: 1.25;
        padding: 12px;
        min-height: 105px;
        max-height: 185px;
    }}
}}
</style>

<div class="caption-wrapper">
    <div>
        <div class="caption-label">Japanese Original</div>
        <div class="jp-caption-box">{safe_original}</div>
    </div>

    {llm_html}

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