import os
import csv
import html
import json
import queue
import threading
import time
import io
import wave
import re
import logging
from difflib import SequenceMatcher
import urllib.parse
import urllib.request

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from groq import Groq


# ============================================================
# Settings
# ============================================================

# WebRTC/aiortc cleanup noise suppression.
# Streamlit Cloud can print this after stopping/restarting WebRTC:
#   AttributeError: 'NoneType' object has no attribute 'sendto'
#   AttributeError: 'NoneType' object has no attribute 'call_exception_handler'
# It happens when aioice retries STUN on a datagram transport that has already
# been closed. It is cleanup noise, not a Groq/Whisper failure.

try:
    from asyncio import selector_events

    if not getattr(selector_events._SelectorDatagramTransport, "_chatgpt_safe_sendto_patch", False):
        _original_selector_datagram_sendto = selector_events._SelectorDatagramTransport.sendto

        def _safe_selector_datagram_sendto(self, data, addr=None):
            if getattr(self, "_sock", None) is None or getattr(self, "_loop", None) is None:
                return None

            try:
                return _original_selector_datagram_sendto(self, data, addr)

            except AttributeError as exc:
                msg = str(exc)
                if (
                    "NoneType" in msg
                    and (
                        "sendto" in msg
                        or "call_exception_handler" in msg
                    )
                ):
                    return None
                raise

        selector_events._SelectorDatagramTransport.sendto = _safe_selector_datagram_sendto
        selector_events._SelectorDatagramTransport._chatgpt_safe_sendto_patch = True

except Exception:
    pass


DEFAULT_TERMS_FILE = "technical_terms.csv"

DEFAULT_RESET_SECONDS = 3.0
MAX_ORIGINAL_CHARS = 900
MAX_TRANSLATION_CHARS = 1400
MAX_HISTORY_ITEMS = 5
MAX_DEBUG_MESSAGES = 10

# Groq Whisper receives mono 16 kHz signed PCM16 packaged as WAV.
AUDIO_INPUT_SAMPLE_RATE = 16000
PCM_BYTES_PER_SAMPLE = 2

# Groq free-plan Whisper is limited to 20 requests/minute, so keep calls
# at least a little more than three seconds apart. Phrase-end silence can
# trigger an early flush only when this minimum interval has elapsed.
GROQ_STT_MODEL_ACCURATE = "whisper-large-v3"
GROQ_STT_MODEL_FAST = "whisper-large-v3-turbo"
GROQ_TRANSLATION_MODEL = "llama-3.1-8b-instant"
GROQ_DEFAULT_CHUNK_SECONDS = 6.4
GROQ_MIN_REQUEST_INTERVAL_SECONDS = 3.1
GROQ_ENDPOINT_SILENCE_SECONDS = 0.90
GROQ_MIN_AUDIO_SECONDS = 1.2
GROQ_AUDIO_OVERLAP_SECONDS = 0.0
GROQ_VAD_RMS_THRESHOLD = 180.0

# Whisper quality gates. Metrics are used when verbose_json is available;
# the code automatically falls back to normal JSON when the SDK/model does
# not expose them. Missing metrics never cause a rejection by themselves.
WHISPER_MAX_NO_SPEECH_PROB = 0.68
WHISPER_MIN_AVG_LOGPROB = -1.25
WHISPER_MAX_COMPRESSION_RATIO = 2.60
WHISPER_PROMPT_MAX_TERMS = 14
WHISPER_PROMPT_LEAK_PATTERNS = [
    "日本語の技術講義",
    "用語を正確に表記",
    "話題:",
    "直前:",
    "用語:",
]
WHISPER_COMMON_HALLUCINATIONS = [
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "字幕をご覧",
    "字幕をオン",
]

# Bound the queue so a slow network cannot create minutes of delayed audio.
# At roughly 20 ms per WebRTC frame, 250 chunks is about five seconds.
AUDIO_QUEUE_MAX_CHUNKS = 250

# iPhone/Safari needs a new user tap after refresh before microphone capture.
# Kept generous because real ICE/TURN negotiation can take longer than a few
# seconds on slower networks; too short a timeout tears down a connection
# that was still actively negotiating and restarts it in an endless loop.
MOBILE_MIC_START_TIMEOUT_SECONDS = 25.0

# Japanese-only safety:
# Ignore accidental Spanish/English/other-language recognition.
JAPANESE_ONLY_MODE = True

# A translation-only update (no new Japanese in this message) is only trusted
# if real Japanese arrived this recently. Without this, background speech in
# another language can produce a translation that rides in on the fact that
# *some* Japanese existed earlier in the segment, even minutes ago.
TRANSLATION_WITHOUT_SOURCE_MAX_LAG_SECONDS = 2.5

# Key terms are evidence-based and belong only to the currently displayed
# finalized paragraph. They are cleared when a new paragraph begins.
KEY_TERM_STALE_SECONDS = 0.0

# Values in common_wrong are not all recognition mistakes. Some are valid,
# related concepts or translations and must never be used as automatic
# source replacements. Example: 自動車工学 is not identical to the ARE
# program name, and 面取り is not a wrong Japanese form of Chamfer.
RELATED_NOT_RECOGNITION_VARIANTS = {
    ("ARE", "自動車工学"),
    ("ARE", "自動車ロボティクス"),
    ("ARE", "自動車とロボット工学"),
    ("PDE", "プロダクトデザイン"),
    ("PDE", "製品設計"),
    ("PDE", "製品デザイン工学"),
    ("BE", "ビジネス工学"),
    ("BE", "ビジネスエンジニアリング"),
    ("Chamfer", "面取り"),
    ("三角", "三角形"),
    ("Automotive and Robotics Engineering", "Automotive Robotics Engineering"),
    ("Automotive and Robotics Engineering", "Automotive & Robotics Engineering"),
    ("Product Design Engineering", "Product Design"),
    ("Business Engineering", "business engineer"),
}

# Extra observed ASR variants. These are supplied to the second-pass AI as
# evidence candidates, but ambiguous forms are not blindly replaced.
ADDITIONAL_RECOGNITION_VARIANTS = {
    "治具": ["リグ", "GQ", "時具", "地具", "ジグ"],
    "幾何拘束": ["気化拘束", "記号拘束", "幾何高速", "幾何校則"],
    "面取り": ["メーカー", "面取", "面どり"],
    "ARE": ["AERI", "AROI", "Aroi", "ARO", "エーアールイー"],
    "サマーコース": ["お出様コース", "お客様コース", "サマコース", "サマー講座"],
    "三面図": ["三次元図", "三面図面", "3面図"],
    "ゲイン調整": ["ゲイン調節", "原因調整"],
    "応答性": ["オートセット", "応答制"],
    "車間距離": ["Shaken Carrier", "車間距離"],
    "相対速度": ["Sorter", "temperate speed", "相対速度"],
    "緊急ブレーキ": ["急ブレーキ"],
    "慣性補償": ["感性補償", "完成補償", "感性保証", "完成保証"],
    "慣性補償制御": ["感性補償制御", "完成補償制御", "完成報告書を制御"],
    "寸法拘束": ["寸法高速", "寸法校則", "寸法公則"],
    "CATIA": ["ガチャ", "カティア", "キャティア", "カチア"],
}

SECOND_PASS_MAX_LENGTH_RATIO = 1.70
SECOND_PASS_MIN_SIMILARITY_WITHOUT_EVIDENCE = 0.46
SECOND_PASS_MAX_CONFIRMED_TERMS = 16

# Paragraph-safe pipeline:
# Whisper receives no glossary prompt. Each audio chunk is finalized independently
# by the second-pass AI, then immutable phrase blocks are appended in the UI.
SECOND_PASS_IN_WORKER = True
MAX_FINALIZED_PHRASE_BLOCKS = 16

SOURCE_LANG_JA_ONLY = "Japanese only"

# Main pipeline: Groq Whisper Japanese STT followed by Groq text translation.
# The required second pass uses the same fast Llama model for key-term correction.
GROQ_HELPER_FAST = "llama-3.1-8b-instant"

ENGINE_GROQ_WHISPER = "Groq Whisper STT + Groq translation"

MAX_LLM_CONTEXT_CHUNKS = 2
MAX_GROQ_CONTEXT_CHARS = 1100
MAX_GROQ_TRANSLATION_CHARS = 450
MAX_GROQ_GLOSSARY_TERMS = 16

# Helper AI safety net.
# Main Groq captions keep running, but second-pass
# helper calls are limited so daily quota is protected.
LLM_BUDGET_MODES = {
    "Every Phrase": {
        "interval": 4.0,
        "min_chars": 8,
        "session_limit": 1000,
        "description": "Recommended: run the second-pass key-term corrector after each completed phrase.",
    },
    "Balanced": {
        "interval": 8.0,
        "min_chars": 20,
        "session_limit": 600,
        "description": "Fewer correction calls while still checking most completed technical phrases.",
    },
    "Saver": {
        "interval": 15.0,
        "min_chars": 40,
        "session_limit": 300,
        "description": "Use only when Groq quota is tight; some short phrases may not receive a second pass.",
    },
}


# Built-in glossary terms not covered by technical_terms.csv.
# These are always added even when the CSV does not include them.
# Duplicate terms (same jp already in the CSV) live only in the CSV now,
# with their common_wrong variants merged there.
EXTRA_GLOSSARY_ENTRIES = [
    {
        "domain": "school",
        "jp": "サマーコース",
        "reading": "さまーこーす",
        "en": "Summer Course",
        "common_wrong": "サマコース;サマー講座;summer course",
        "notes": "ASO/BINUS summer course program",
    },
    {
        "domain": "school",
        "jp": "ビヌス",
        "reading": "びぬす",
        "en": "BINUS",
        "common_wrong": "ビナス;ビーナス;ネウス;ヴィヌス;venus;Venus",
        "notes": "BINUS name in Japanese speech",
    },
    {
        "domain": "school",
        "jp": "ビヌスASO",
        "reading": "びぬすえーえすおー",
        "en": "BINUS ASO",
        "common_wrong": "ビヌスアソ;ビヌス麻生;ビナスASO;ビーナスASO;ネウスASO;ネウスアソ;BINUS ASO",
        "notes": "BINUS ASO program/school name",
    },
    {
        "domain": "school",
        "jp": "ビヌス大学",
        "reading": "びぬすだいがく",
        "en": "BINUS University",
        "common_wrong": "ビナス大学;ビーナス大学;ネウス大学;ヴィヌス大学;BINUS University",
        "notes": "BINUS University",
    },
    {
        "domain": "school",
        "jp": "ARE",
        "reading": "えーあーるいー",
        "en": "Automotive and Robotics Engineering",
        "common_wrong": "AROI;Aroi;ARO;A.R.E.;エーアールイー;エーアール;自動車工学;自動車ロボティクス;自動車とロボット工学",
        "notes": "BINUS ASO major: Automotive and Robotics Engineering",
    },
    {
        "domain": "school",
        "jp": "PDE",
        "reading": "ぴーでぃーいー",
        "en": "Product Design Engineering",
        "common_wrong": "PDA;PDE;PD;PE;ADC;ピーディーイー;ピーディー;プロダクトデザイン;製品設計;製品デザイン工学",
        "notes": "BINUS ASO major: Product Design Engineering",
    },
    {
        "domain": "school",
        "jp": "BE",
        "reading": "びーいー",
        "en": "Business Engineering",
        "common_wrong": "B;BA;ビー;ビーイー;ビジネス工学;ビジネスエンジニアリング",
        "notes": "BINUS ASO major: Business Engineering",
    },
    {
        "domain": "school",
        "jp": "Automotive and Robotics Engineering",
        "reading": "おーともちぶ あんど ろぼてぃくす えんじにありんぐ",
        "en": "Automotive and Robotics Engineering",
        "common_wrong": "Automotive Robotics Engineering;Automotive & Robotics Engineering;automotive and robotics engineering;automotive robotics",
        "notes": "Full English name for ARE",
    },
    {
        "domain": "school",
        "jp": "Product Design Engineering",
        "reading": "ぷろだくと でざいん えんじにありんぐ",
        "en": "Product Design Engineering",
        "common_wrong": "Product Design;product design engineering;product design;PDA;ADC",
        "notes": "Full English name for PDE",
    },
    {
        "domain": "school",
        "jp": "Business Engineering",
        "reading": "びじねす えんじにありんぐ",
        "en": "Business Engineering",
        "common_wrong": "business engineering;business engineer;BE;BA",
        "notes": "Full English name for BE",
    },
    {
        "domain": "cad",
        "jp": "Chamfer",
        "reading": "ちゃんふぁー",
        "en": "Chamfer",
        "common_wrong": "チャンファー;シャンファー;面取り;chamfer;Chamfering",
        "notes": "Beveled edge feature",
    },
    {
        "domain": "automotive",
        "jp": "ロータリーエンジン",
        "reading": "ろーたりーえんじん",
        "en": "rotary engine",
        "common_wrong": "Rotary Engine;rotary engine;ロータリエンジン;ロータリーエンジン;ロタリーエンジン;ロータリー",
        "notes": "Wankel-type rotary engine",
    },
    {
        "domain": "automotive",
        "jp": "レシプロエンジン",
        "reading": "れしぷろえんじん",
        "en": "reciprocating engine",
        "common_wrong": "reciprocating engine;piston engine;レシプロ;ピストンエンジン",
        "notes": "Conventional piston engine",
    },
    {
        "domain": "automotive",
        "jp": "ローター",
        "reading": "ろーたー",
        "en": "rotor",
        "common_wrong": "rotor;Rotor;ロータ",
        "notes": "Rotating element in a rotary engine",
    },
    {
        "domain": "automotive",
        "jp": "アペックスシール",
        "reading": "あぺっくすしーる",
        "en": "apex seal",
        "common_wrong": "apex seal;Apex seal;アペックス;アペックシール",
        "notes": "Seal at the rotor apex in a rotary engine",
    },
    {
        "domain": "cad",
        "jp": "三角",
        "reading": "さんかく",
        "en": "triangle",
        "common_wrong": "三角形;さんかくけい;triangle",
        "notes": "Basic geometry / CAD shape",
    },
]


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



@st.cache_data(ttl=1800, show_spinner=False)
def fetch_metered_ice_servers(api_key):
    """
    Fetch temporary STUN/TURN credentials from Metered.

    The API key stays on the Streamlit server. It is not sent to browser code.
    Returned ICE credentials are cached for 30 minutes.
    """
    api_key = str(api_key or "").strip()

    if not api_key:
        return [], "METERED_TURN_API_KEY is missing."

    query = urllib.parse.urlencode({"apiKey": api_key})
    url = (
        "https://translation.metered.live/api/v1/turn/credentials?"
        + query
    )

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "technical-interpreter-captioner/1.0",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)

    except Exception as exc:
        return [], f"Could not fetch Metered TURN credentials: {exc}"

    if not isinstance(data, list):
        return [], "Metered TURN API returned an unexpected response."

    ice_servers = []

    for item in data:
        if not isinstance(item, dict):
            continue

        urls = item.get("urls")

        if not urls:
            continue

        server = {"urls": urls}

        username = item.get("username")
        credential = item.get("credential")

        if username:
            server["username"] = str(username)

        if credential:
            server["credential"] = str(credential)

        ice_servers.append(server)

    if not ice_servers:
        return [], "Metered TURN API returned no ICE servers."

    return ice_servers, ""


def build_rtc_configuration():
    """
    Build the browser WebRTC ICE configuration.

    Primary:
        Metered temporary TURN/STUN credentials.

    Fallback:
        Google STUN only. This may work on desktop Wi-Fi, but mobile networks
        often require TURN.
    """
    metered_api_key = safe_get_secret_or_env("METERED_TURN_API_KEY")
    metered_servers, metered_error = fetch_metered_ice_servers(metered_api_key)

    if metered_servers:
        return (
            RTCConfiguration({
                "iceServers": metered_servers,
                "iceTransportPolicy": "all",
            }),
            True,
            "",
            metered_servers,
        )

    fallback_servers = [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
            ]
        }
    ]

    return (
        RTCConfiguration({
            "iceServers": fallback_servers,
            "iceTransportPolicy": "all",
        }),
        False,
        metered_error,
        fallback_servers,
    )


# ============================================================
# Glossary
# ============================================================

@st.cache_data(show_spinner=False)
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

        for extra in EXTRA_GLOSSARY_ENTRIES:
            jp = extra.get("jp", "").strip()
            en = extra.get("en", "").strip()
            reading = extra.get("reading", "").strip()
            common_wrong = extra.get("common_wrong", "").strip()

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


@st.cache_data(show_spinner=False)
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

    existing = set((item.get("jp", ""), item.get("en", "")) for item in entries)

    for extra in EXTRA_GLOSSARY_ENTRIES:
        key = (extra.get("jp", ""), extra.get("en", ""))
        if key not in existing:
            entries.append(extra)
            existing.add(key)

    return entries


def _is_ascii_token(text):
    text = str(text or "").strip()
    return bool(text) and all(ord(ch) < 128 for ch in text)


def _literal_term_in_text(term, text):
    """Exact source support with word boundaries for short Latin acronyms."""
    term = str(term or "").strip()
    text = str(text or "")

    if not term:
        return False

    if _is_ascii_token(term) and len(term) <= 6:
        return (
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
            is not None
        )

    if _is_ascii_token(term):
        return term.lower() in text.lower()

    return term in text


def recognition_variants_for_entry(entry):
    """
    Return only plausible ASR variants.

    This deliberately excludes related-but-correct concepts such as
    自動車工学 -> ARE and 面取り -> Chamfer.
    """
    canonical = str(entry.get("jp", "")).strip()
    variants = []
    seen = set()

    raw_values = [
        value.strip()
        for value in str(entry.get("common_wrong", "") or "").split(";")
        if value.strip()
    ]
    raw_values.extend(ADDITIONAL_RECOGNITION_VARIANTS.get(canonical, []))

    for value in raw_values:
        if not value or value == canonical or len(value) < 2:
            continue

        if (canonical, value) in RELATED_NOT_RECOGNITION_VARIANTS:
            continue

        # Avoid treating a valid longer/shorter canonical form as a mistake.
        if value.startswith(canonical) or canonical.startswith(value):
            continue

        key = value.lower()
        if key in seen:
            continue

        variants.append(value)
        seen.add(key)

    return variants


def glossary_entry_for_term(term, glossary_entries):
    term = str(term or "").strip()

    for entry in glossary_entries or []:
        canonical = str(entry.get("jp", "")).strip()
        if canonical == term:
            return entry

    return None


def source_evidence_for_term(term, source_text, glossary_entries):
    """
    Return an exact substring from the raw Japanese that supports a canonical
    term, or an empty string when the term is unsupported.
    """
    term = str(term or "").strip()
    source_text = str(source_text or "")

    if _literal_term_in_text(term, source_text):
        return term

    entry = glossary_entry_for_term(term, glossary_entries)
    if not entry:
        return ""

    variants = sorted(
        recognition_variants_for_entry(entry),
        key=len,
        reverse=True,
    )

    for variant in variants:
        if _literal_term_in_text(variant, source_text):
            return variant

    return ""


def glossary_priority(term, domain_mode):
    """Prefer domain-specific compound terms over generic nouns."""
    domain = normalize_domain_name(domain_mode) if "normalize_domain_name" in globals() else str(domain_mode or "").lower()
    priority = {
        "CATIA": 0,
        "治具": 1,
        "三次元モデル": 2,
        "寸法拘束": 3,
        "幾何拘束": 4,
        "Pad": 5,
        "フィレット": 6,
        "面取り": 7,
        "Chamfer": 8,
        "三面図": 9,
        "加工性": 10,
        "慣性補償制御": 0,
        "慣性補償": 1,
        "ゲイン調整": 2,
        "応答性": 3,
        "AEB": 4,
        "TTC": 5,
        "車間距離": 6,
        "相対速度": 7,
        "緊急ブレーキ": 8,
        "制動力": 9,
        "サマーコース": 0,
        "ビヌスASO": 1,
        "BINUS ASO": 1,
        "ARE": 2,
        "PDE": 3,
        "BE": 4,
    }
    base = priority.get(term, 50)
    # Longer Japanese compounds usually carry more technical information.
    return (base, -len(term), term)


def build_exact_confirmed_terms(
    corrected_japanese,
    raw_japanese,
    glossary_entries,
    domain_mode,
    ai_terms=None,
    max_terms=SECOND_PASS_MAX_CONFIRMED_TERMS,
):
    """
    Build the UI key-term list from evidence only.

    A term must be literal in corrected Japanese. It also needs either a
    literal/raw ASR variant in the source or explicit AI evidence that is an
    exact raw substring. English meanings are taken from the glossary when
    available, not invented by the model.
    """
    corrected_japanese = str(corrected_japanese or "").strip()
    raw_japanese = str(raw_japanese or "").strip()
    ai_terms = ai_terms or []

    ai_by_term = {}
    for item in ai_terms:
        term = str(item.get("term", item.get("jp", ""))).strip()
        if term:
            ai_by_term[term] = item

    output = []
    seen = set()

    for entry in glossary_entries or []:
        if not glossary_entry_matches_domain(entry, domain_mode):
            continue

        term = str(entry.get("jp", "")).strip()
        meaning = str(entry.get("en", "")).strip()
        if not term or not meaning:
            continue

        if not _literal_term_in_text(term, corrected_japanese):
            continue

        evidence = source_evidence_for_term(term, raw_japanese, glossary_entries)
        ai_item = ai_by_term.get(term, {})
        ai_evidence = str(
            ai_item.get("evidence", ai_item.get("source_match", ""))
        ).strip()

        if not evidence and ai_evidence and _literal_term_in_text(ai_evidence, raw_japanese):
            evidence = ai_evidence

        # A literal term in the corrected sentence with no raw evidence may
        # have been inserted by the model. Do not display it as confirmed.
        if not evidence:
            continue

        key = term.lower()
        if key in seen:
            continue

        output.append({
            "term": term,
            "meaning": meaning,
            "reading": str(entry.get("reading", "") or "").strip(),
            "source_match": evidence,
            "evidence": evidence,
            "confidence": str(ai_item.get("confidence", "high") or "high").lower(),
            "added_at": time.time(),
            "last_confirmed_at": time.time(),
        })
        seen.add(key)

    # Keep supported AI acronyms/terms that are not present in the CSV.
    for item in ai_terms:
        term = str(item.get("term", item.get("jp", ""))).strip()
        meaning = str(item.get("meaning", item.get("en", ""))).strip()
        evidence = str(item.get("evidence", item.get("source_match", ""))).strip()

        if not term or not meaning or term.lower() in seen:
            continue
        if not _literal_term_in_text(term, corrected_japanese):
            continue
        if not evidence:
            evidence = term if _literal_term_in_text(term, raw_japanese) else ""
        if not evidence or not _literal_term_in_text(evidence, raw_japanese):
            continue

        output.append({
            "term": term,
            "meaning": meaning,
            "reading": str(item.get("reading", "") or "").strip(),
            "source_match": evidence,
            "evidence": evidence,
            "confidence": str(item.get("confidence", "medium") or "medium").lower(),
            "added_at": time.time(),
            "last_confirmed_at": time.time(),
        })
        seen.add(term.lower())

    output.sort(key=lambda item: glossary_priority(item.get("term", ""), domain_mode))
    return output[:max_terms]


def validated_second_pass_corrections(
    raw_japanese,
    corrected_japanese,
    corrections,
    glossary_entries=None,
    domain_mode="auto",
):
    """
    Keep only correction pairs explicitly allowed by the glossary.

    The second-pass model may choose among approved ASR variants, but it may
    not invent arbitrary replacements or rewrite sentence structure.
    """
    raw_japanese = str(raw_japanese or "")
    corrected_japanese = str(corrected_japanese or "")
    allowed_pairs = set()

    for entry in glossary_entries or []:
        if not glossary_entry_matches_domain(entry, domain_mode):
            continue
        canonical = str(entry.get("jp", "")).strip()
        if not canonical:
            continue
        for variant in recognition_variants_for_entry(entry):
            if variant and variant != canonical:
                allowed_pairs.add((variant, canonical))

    # Small non-glossary repairs that were repeatedly observed and do not
    # change technical meaning.
    allowed_pairs.update({
        ("ちさくナルト", "小さくなると"),
        ("ちさくなると", "小さくなると"),
        ("小さくナルト", "小さくなると"),
    })

    valid = []
    seen = set()

    for item in corrections or []:
        wrong = str(item.get("wrong", "")).strip()
        correct = str(item.get("correct", "")).strip()
        reason = str(item.get("reason", "")).strip()

        if len(wrong) < 2 or not correct:
            continue
        if (wrong, correct) not in allowed_pairs:
            continue
        if not _literal_term_in_text(wrong, raw_japanese):
            continue
        if not _literal_term_in_text(correct, corrected_japanese):
            continue

        key = (wrong, correct)
        if key in seen:
            continue

        valid.append({
            "wrong": wrong,
            "correct": correct,
            "reason": reason,
        })
        seen.add(key)

    return valid

def second_pass_japanese_is_safe(raw_japanese, corrected_japanese, valid_corrections):
    """
    Reject free rewriting while still allowing strong evidence-based repair of
    badly misheard technical phrases.
    """
    raw = str(raw_japanese or "").strip()
    corrected = str(corrected_japanese or "").strip()

    if not raw or not corrected:
        return False, "missing raw or corrected Japanese"

    if len(corrected) > max(30, int(len(raw) * SECOND_PASS_MAX_LENGTH_RATIO)):
        return False, "corrected Japanese added too much text"

    # Preserve every number spoken in the raw transcript.
    raw_numbers = re.findall(r"\d+(?:\.\d+)?", raw)
    simple_number_kanji = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
        "10": "十",
    }
    for number in raw_numbers:
        equivalent_kanji = simple_number_kanji.get(number, "")
        if number not in corrected and (
            not equivalent_kanji or equivalent_kanji not in corrected
        ):
            return False, f"number {number} was dropped"

    # Preserve recognized technical acronyms unless an explicit grounded
    # correction changes that exact token.
    raw_acronyms = re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,7}(?![A-Za-z0-9])", raw)
    correction_wrongs = {item.get("wrong", "") for item in valid_corrections}
    for acronym in raw_acronyms:
        if acronym not in corrected and acronym not in correction_wrongs:
            return False, f"acronym {acronym} was dropped"

    norm_raw = re.sub(r"[\s、。,.!?！？「」『』（）()]", "", raw)
    norm_corrected = re.sub(r"[\s、。,.!?！？「」『』（）()]", "", corrected)
    similarity = SequenceMatcher(None, norm_raw, norm_corrected).ratio()

    if (
        similarity < SECOND_PASS_MIN_SIMILARITY_WITHOUT_EVIDENCE
        and not valid_corrections
    ):
        return False, f"unsupported rewrite similarity={similarity:.2f}"

    # Avoid duplicated explanatory additions.
    if corrected.count("このように") > raw.count("このように"):
        return False, "added explanatory phrase"
    if corrected.count("については") > raw.count("については") + 1:
        return False, "added topic framing"

    return True, ""




def align_english_to_confirmed_japanese(corrected_japanese, english_text):
    """
    Apply only source-licensed terminology alignment.

    These edits are allowed only when the canonical Japanese term is literal
    in the corrected caption.
    """
    jp = str(corrected_japanese or "")
    en = str(english_text or "").strip()

    conditional_replacements = [
        ("治具", r"\brigs?\b", "jig"),
        ("幾何拘束", r"\bcondensation constraints?\b", "geometric constraint"),
        ("面取り", r"\bmanufacturer\b", "chamfering"),
        ("三面図", r"\bthree-dimensional diagram\b", "three-view drawing"),
        ("三面図", r"\b3D diagram\b", "three-view drawing"),
        ("応答性", r"\bauto-set\b", "responsiveness"),
        ("ゲイン調整", r"\bgain adjustment\b", "gain adjustment"),
    ]

    for japanese_term, pattern, replacement in conditional_replacements:
        if japanese_term in jp:
            en = re.sub(pattern, replacement, en, flags=re.IGNORECASE)

    # Preserve plural form for geometric constraints when the English source
    # used the plural.
    if "幾何拘束" in jp:
        en = re.sub(
            r"\bgeometric constraint\b(?=\s+(?:on|and|are|were)\b)",
            "geometric constraints",
            en,
            flags=re.IGNORECASE,
        )

    return clean_plain_translation(en)


def second_pass_english_is_safe(initial_english, corrected_english):
    initial = str(initial_english or "").strip()
    corrected = str(corrected_english or "").strip()

    if not corrected:
        return False, "empty corrected English"

    banned_additions = [
        "this is how",
        "in other words",
        "for example",
        "as you can see",
        "let me explain",
        "we can see that",
    ]
    lower = corrected.lower()
    for phrase in banned_additions:
        if phrase in lower and phrase not in initial.lower():
            return False, f"added explanation: {phrase}"

    if initial and len(corrected) > max(80, int(len(initial) * 1.75)):
        return False, "corrected English added too much text"

    if corrected.count(".") > max(2, initial.count(".") + 1):
        return False, "corrected English added extra sentences"

    return True, ""



def sanitize_second_pass_output(
    raw_japanese,
    raw_english,
    parsed,
    glossary_entries,
    domain_mode,
):
    """
    Convert model suggestions into a grounded result.

    The model is not allowed to rewrite the sentence. The final Japanese is
    reconstructed from the raw transcript using deterministic corrections plus
    only approved glossary variant pairs returned by the second pass.
    """
    raw_japanese = str(raw_japanese or "").strip()
    raw_english = str(raw_english or "").strip()
    model_japanese = str(
        parsed.get("corrected_japanese_original", "")
    ).strip()

    if not model_japanese:
        model_japanese = raw_japanese

    model_japanese = light_original_cleanup(model_japanese)
    model_japanese, _ = light_domain_context_cleanup(
        model_japanese,
        "",
        domain_mode,
    )

    valid_corrections = validated_second_pass_corrections(
        raw_japanese,
        model_japanese,
        parsed.get("corrections", []),
        glossary_entries=glossary_entries,
        domain_mode=domain_mode,
    )

    # Reconstruct instead of trusting the model's rewritten Japanese.
    grounded_japanese = apply_glossary_source_corrections(
        raw_japanese,
        glossary_entries,
        domain_mode,
    )
    grounded_japanese = apply_llm_corrections(
        grounded_japanese,
        valid_corrections,
    )
    grounded_japanese = light_original_cleanup(grounded_japanese)
    grounded_japanese, _ = light_domain_context_cleanup(
        grounded_japanese,
        "",
        domain_mode,
    )

    safe_japanese, japanese_reason = second_pass_japanese_is_safe(
        raw_japanese,
        grounded_japanese,
        valid_corrections,
    )
    if not safe_japanese:
        grounded_japanese = apply_glossary_source_corrections(
            raw_japanese,
            glossary_entries,
            domain_mode,
        )
        parsed["is_unclear"] = True
        existing_reason = str(parsed.get("unclear_reason", "")).strip()
        parsed["unclear_reason"] = (
            f"{existing_reason}; {japanese_reason}".strip("; ")
        )

    model_english = str(
        parsed.get("corrected_english_caption", "")
    ).strip()
    model_english = align_english_to_confirmed_japanese(
        grounded_japanese,
        model_english or raw_english,
    )

    safe_english, english_reason = second_pass_english_is_safe(
        raw_english,
        model_english,
    )
    if not safe_english:
        model_english = raw_english
        parsed["is_unclear"] = True
        existing_reason = str(parsed.get("unclear_reason", "")).strip()
        parsed["unclear_reason"] = (
            f"{existing_reason}; {english_reason}".strip("; ")
        )

    confirmed_terms = build_exact_confirmed_terms(
        corrected_japanese=grounded_japanese,
        raw_japanese=raw_japanese,
        glossary_entries=glossary_entries,
        domain_mode=domain_mode,
        ai_terms=parsed.get(
            "confirmed_terms",
            parsed.get("key_terms", []),
        ),
    )

    safety_reason = "; ".join(
        reason
        for reason in [japanese_reason, english_reason]
        if reason
    )

    return {
        "corrected_japanese_original": grounded_japanese,
        "corrected_english_caption": model_english or raw_english,
        "is_unclear": bool(parsed.get("is_unclear", False)),
        "unclear_reason": str(parsed.get("unclear_reason", "")).strip(),
        "confirmed_terms": confirmed_terms,
        "corrections": valid_corrections,
        "safety_reason": safety_reason,
    }

def contains_any_text(text, words):
    text = text or ""
    lower = text.lower()

    for word in words:
        if not word:
            continue

        if word in text:
            return True

        if word.lower() in lower:
            return True

    return False


def is_school_context_active(original_text, translation_text, domain_mode):
    domain = (domain_mode or "auto").lower()

    if domain in ["school", "school/event", "event"]:
        return True

    combined = f"{original_text or ''}\n{translation_text or ''}"

    return contains_any_text(combined, SCHOOL_STRONG_CONTEXT_WORDS)


def is_school_related_term(term, meaning):
    combined = f"{term or ''}\n{meaning or ''}"
    return contains_any_text(combined, SCHOOL_TERM_WORDS)


def is_term_relevant_to_current_caption(term, meaning, original_text, translation_text, domain_mode):
    """
    Avoid stale or hallucinated key terms.

    Example problem:
    Current speech is about 挨拶 / schedule,
    but the LLM box shows サマーコース / ビヌス大学 from old context.
    This function blocks that unless the current caption actually supports it.
    """
    term = (term or "").strip()
    meaning = (meaning or "").strip()
    original_text = original_text or ""
    translation_text = translation_text or ""

    if not term and not meaning:
        return False

    combined = f"{original_text}\n{translation_text}"
    combined_lower = combined.lower()

    # Strong block: do not show school terms unless school context is currently active.
    if is_school_related_term(term, meaning):
        if not is_school_context_active(original_text, translation_text, domain_mode):
            return False

    # Direct mention is always okay.
    for value in [term, meaning]:
        value = (value or "").strip()

        if not value:
            continue

        if value in combined:
            return True

        if value.lower() in combined_lower:
            return True

    # Do not allow terms only because the selected domain is CAD/automotive.
    # Domain context is background, not proof. This prevents hallucinated
    # terms like CATIA appearing when the current speech did not mention it.
    return False


def filter_llm_key_terms_for_current_caption(key_terms, original_text, translation_text, domain_mode, max_terms=5):
    filtered = []

    for item in key_terms or []:
        term = str(item.get("term", item.get("jp", ""))).strip()
        meaning = str(item.get("meaning", item.get("en", ""))).strip()

        if not term:
            continue

        source_match = str(item.get("source_match", item.get("matched_candidate", ""))).strip()
        source_text = original_text or ""

        if source_match and glossary_candidate_matches(source_match, source_text):
            pass
        elif not is_term_relevant_to_current_caption(
            term,
            meaning,
            original_text,
            "",  # never use English translation as proof for key terms
            domain_mode,
        ):
            continue
        else:
            # Groq's own key_terms never carry a source_match. Stamp one now
            # that it's confirmed present, so it gets the same hold-for-the-
            # segment treatment as glossary-matched terms instead of being
            # re-validated (and pruned) on every single render.
            item = dict(item)
            item["source_match"] = term

        filtered.append(item)

        if len(filtered) >= max_terms:
            break

    return filtered


def filter_detected_terms_for_current_caption(detected_terms, original_text, translation_text, domain_mode, max_terms=8):
    """
    Same filter, but for glossary rows before sending them to Groq.
    This prevents the second AI prompt from being polluted by unrelated
    Summer Course/BINUS terms in Auto mode.
    """
    filtered = []

    for item in detected_terms or []:
        jp = str(item.get("jp", "")).strip()
        en = str(item.get("en", "")).strip()

        matched_candidate = str(item.get("matched_candidate", "")).strip()
        source_text = original_text or ""

        if matched_candidate and glossary_candidate_matches(matched_candidate, source_text):
            filtered.append(item)
        elif is_term_relevant_to_current_caption(
            jp,
            en,
            original_text,
            "",  # do not let wrong English translation support key terms
            domain_mode,
        ):
            filtered.append(item)
        else:
            continue

        if len(filtered) >= max_terms:
            break

    return filtered


def detected_terms_to_llm_key_terms(detected_terms, max_terms=8):
    """
    Convert glossary matches to the same format used by LLM key terms.
    This lets the UI show known terms immediately without waiting for Groq.
    """
    output = []
    seen = set()

    for item in detected_terms or []:
        jp = str(item.get("jp", "")).strip()
        en = str(item.get("en", "")).strip()

        if not jp:
            continue

        key = (jp, en)

        if key in seen:
            continue

        output.append({
            "term": jp,
            "meaning": en,
            "reading": item.get("reading", ""),
            "source_match": item.get("matched_candidate", ""),
            "added_at": time.time(),
        })
        seen.add(key)

        if len(output) >= max_terms:
            break

    return output


def merge_key_terms_preserve_order(primary_terms, secondary_terms, max_terms=5):
    merged = []
    seen = set()

    for item in list(primary_terms or []) + list(secondary_terms or []):
        term = str(item.get("term", item.get("jp", ""))).strip()
        meaning = str(item.get("meaning", item.get("en", ""))).strip()

        if not term:
            continue

        line = normalize_key_term_line(term, meaning)

        if not line:
            continue

        key = line.lower()

        if key in seen:
            continue

        added_at = float(item.get("added_at", time.time()) or time.time())

        merged.append({
            "term": term,
            "meaning": meaning,
            "reading": item.get("reading", ""),
            "source_match": item.get("source_match", item.get("matched_candidate", "")),
            "added_at": added_at,
            "last_confirmed_at": float(item.get("last_confirmed_at", added_at) or added_at),
        })
        seen.add(key)

        if len(merged) >= max_terms:
            break

    return merged


def glossary_candidate_matches(candidate, text, is_translation=False):
    """
    Preprocessing matcher.

    It prevents false matches like:
        BE inside "because"
        CAD inside a random longer word

    Japanese candidates are matched by substring because Japanese has no spaces.
    """
    candidate = (candidate or "").strip()
    text = text or ""

    if not candidate:
        return False

    if all(ord(ch) < 128 for ch in candidate):
        import re

        if len(candidate) <= 4:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(candidate) + r"(?![A-Za-z0-9])"
            return re.search(pattern, text, flags=re.IGNORECASE) is not None

        return candidate.lower() in text.lower()

    return candidate in text



def extract_key_terms_for_llm(original_text, translation_text, terms_file, max_terms=8):
    """
    Detect only source-supported glossary terms.

    English output is never used as evidence, and related-but-not-equivalent
    aliases such as 自動車工学 -> ARE are excluded from recognition matching.
    """
    original_text = original_text or ""
    entries = load_glossary_entries(terms_file)
    matched_terms = []

    for row in entries:
        jp = str(row.get("jp", "")).strip()
        en = str(row.get("en", "")).strip()
        reading = str(row.get("reading", "")).strip()
        notes = str(row.get("notes", "")).strip()

        if not jp or not en:
            continue

        candidates = [jp, reading]
        candidates.extend(recognition_variants_for_entry(row))

        matched_candidate = ""
        for candidate in candidates:
            if candidate and glossary_candidate_matches(candidate, original_text):
                matched_candidate = candidate
                break

        if matched_candidate:
            matched_terms.append({
                "jp": jp,
                "en": en,
                "reading": reading,
                "notes": notes,
                "matched_candidate": matched_candidate,
            })

    matched_terms.sort(
        key=lambda item: glossary_priority(item.get("jp", ""), "auto")
    )

    unique_terms = []
    seen = set()

    for item in matched_terms:
        key = (item["jp"], item["en"])
        if key in seen:
            continue
        unique_terms.append(item)
        seen.add(key)
        if len(unique_terms) >= max_terms:
            break

    return unique_terms

# ============================================================
# Cleanup and correction
# ============================================================

def apply_llm_corrections(text, corrections):
    import re

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

        if re.fullmatch(r"[A-Za-z0-9]+", wrong):
            # Latin-script terms (acronyms like ARE/BE/PDE) need whole-word
            # matching, otherwise a plain substring replace would also hit
            # unrelated words that merely contain the same letters
            # (e.g. "ARE" inside "prepare", "BE" inside "before").
            cleaned = re.sub(rf"\b{re.escape(wrong)}\b", correct, cleaned)
        else:
            # Japanese has no spaces, so word-boundary matching does not
            # reliably apply here; a direct substring replace is used instead.
            cleaned = cleaned.replace(wrong, correct)

    return cleaned.strip()


def is_japanese_text(text):
    """
    True when text contains Japanese script.
    English acronyms inside Japanese sentences are okay if the sentence
    also contains Hiragana/Katakana/Kanji.
    """
    if not text:
        return False

    for ch in text:
        cp = ord(ch)

        if (
            0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0x4E00 <= cp <= 0x9FFF  # Kanji
        ):
            return True

    return False


# Only real hesitation fillers.
# Do NOT include useful conversational words like はい, うん, そうですね,
# まあ, ちょっと, です. Those may carry meaning and should stay.
JAPANESE_FILLER_WORDS = [
    "えー",
    "ええ",
    "えっと",
    "えーっと",
    "えーと",
    "あの",
    "あのー",
    "あのう",
]


def japanese_char_ratio(text):
    text = (text or "").strip()

    if not text:
        return 0.0

    useful_chars = [
        ch for ch in text
        if not ch.isspace() and ch not in ["。", "、", "？", "?", "！", "!", ".", ",", "…", "・"]
    ]

    if not useful_chars:
        return 0.0

    jp_count = 0

    for ch in useful_chars:
        cp = ord(ch)
        if (
            0x3040 <= cp <= 0x309F
            or 0x30A0 <= cp <= 0x30FF
            or 0x4E00 <= cp <= 0x9FFF
        ):
            jp_count += 1

    return jp_count / max(1, len(useful_chars))


def normalize_for_filler_check(text):
    text = (text or "").strip()

    for mark in ["。", "、", "？", "?", "！", "!", ".", ",", "…", "・", " "]:
        text = text.replace(mark, "")

    return text.strip()


def strip_leading_japanese_fillers(text):
    """
    Remove leading filler only. Do not delete content inside normal sentences.
    """
    text = (text or "").strip()
    changed = True

    while changed:
        changed = False

        for filler in JAPANESE_FILLER_WORDS:
            for prefix in [filler + "、", filler + "。", filler + " ", filler]:
                if text.startswith(prefix) and len(text) > len(prefix):
                    text = text[len(prefix):].strip()
                    changed = True

    return text.strip()


def is_filler_only_japanese(text):
    text = normalize_for_filler_check(text)

    if not text:
        return True

    if not is_japanese_text(text):
        return False

    temp = text

    for filler in sorted(JAPANESE_FILLER_WORDS, key=len, reverse=True):
        temp = temp.replace(filler, "")

    return temp.strip() == ""


def is_filler_only_english(text):
    text = (text or "").strip().lower()

    for mark in [".", ",", "?", "!", "…"]:
        text = text.replace(mark, " ")

    words = [word for word in text.split() if word]

    if not words:
        return True

    filler_words = {
        "um", "uh", "er", "ah", "yeah", "yes", "okay", "ok",
        "well", "so", "like", "you", "know", "right", "hmm"
    }

    return all(word in filler_words for word in words)


def should_skip_as_filler(original, translation):
    original = (original or "").strip()
    translation = (translation or "").strip()

    if original and is_filler_only_japanese(original):
        return True

    if not original and translation and is_filler_only_english(translation):
        return True

    return False


def is_connection_noise_error(message):
    message = str(message or "").lower()
    noisy_parts = [
        "write operation timed out",
        "write timed out",
        "audio send error",
        "socket is already closed",
        "connection is already closed",
        "websocketconnectionclosed",
        "audio data decode timeout",
        "audio data decode",
        "decode timeout",
    ]
    return any(part in message for part in noisy_parts)


def friendly_live_error(message):
    message = str(message or "").strip()

    if not message:
        return ""

    lower = message.lower()

    if "request timeout" in lower or "timed out" in lower:
        return "Groq speech request timed out. Press Start Translation again."

    if "write operation timed out" in lower or "audio send error" in lower:
        return "Groq speech processing stopped. Press Start Translation again."

    return "Groq live caption worker stopped. Press Start Translation again."


def looks_like_valid_japanese_for_display(text):
    """
    Text gate before display/Groq.

    This cannot perfectly detect the original spoken language, but it blocks:
    - English/Indonesian text with no Japanese script
    - filler-only Japanese
    - tiny vague Japanese fragments that are not glossary terms
    """
    text = (text or "").strip()

    if not text:
        return False

    if not is_japanese_text(text):
        normalized_acronym = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        allowed_acronyms = {
            "AEB", "TTC", "ADAS", "ABS", "ECU", "CAN", "PWM", "PID",
            "IPM", "YOLO", "ARE", "PDE", "BE", "CATIA", "CAD",
        }
        if normalized_acronym not in allowed_acronyms:
            return False

    if is_japanese_text(text) and is_filler_only_japanese(text):
        return False

    # If Japanese ratio is very low, it is probably mostly English/noise.
    # Japanese sentence with acronyms like CAD still passes because ratio is usually > 0.15.
    if is_japanese_text(text) and japanese_char_ratio(text) < 0.15:
        return False

    vague = normalize_for_filler_check(text)

    if vague in {"それ", "これ", "あれ", "どれ", "ここ", "そこ", "あそこ", "です", "ます", "した", "する"}:
        return False

    return True


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

        # School / event terms
        "Venus University": "BINUS University",
        "Neus University": "BINUS University",
        "Binus University": "BINUS University",
        "BINUS university": "BINUS University",
        "Venus ASO": "BINUS ASO",
        "Neus ASO": "BINUS ASO",
        "Binus ASO": "BINUS ASO",
        "Venus": "BINUS",
        "Neus": "BINUS",
        "Binus": "BINUS",
        "summer course": "Summer Course",

        # BINUS ASO major names
        "AROI": "ARE",
        "Aroi": "ARE",
        "ARO": "ARE",
        "Automotive Robotics Engineering": "Automotive and Robotics Engineering",
        "Automotive & Robotics Engineering": "Automotive and Robotics Engineering",
        "automotive and robotics engineering": "Automotive and Robotics Engineering",
        "automotive robotics engineering": "Automotive and Robotics Engineering",

        "PDA": "PDE",
        "ADC": "PDE",
        "product design engineering": "Product Design Engineering",

        "business engineering": "Business Engineering",

        # CATIA / CAD / product design terms
        "Catia": "CATIA",
        "catia": "CATIA",
        "CADIA": "CATIA",
        "Catiya": "CATIA",
        "CADIA": "CATIA",
        "Computer Aided Design": "CAD",
        "computer aided design": "CAD",
        "Computer-Aided Design": "CAD",
        "cad": "CAD",
        "dimension constraint": "dimensional constraint",
        "dimensional constraints": "dimensional constraints",
        "geometry constraint": "geometric constraint",
        "geometrical constraint": "geometric constraint",
        "fully constraint": "fully constrained",
        "full constraint": "fully constrained",
        "degree of freedom": "degrees of freedom",
        "filet": "fillet",
        "Fillet": "fillet",
        "chamfering": "chamfering",
        "Chamfering": "chamfering",
        "manufacturing feasibility": "manufacturability",
        "spacetime CAD": "jig CAD",
        "space-time CAD": "jig CAD",
        "space time CAD": "jig CAD",
        "spacetime": "jig",
        "space-time": "jig",
        "space time": "jig",
        "three-view drawing": "three-view drawing",
        "orthographic drawing": "orthographic drawing",

        # Rotary engine terms
        "Rotary Engine": "rotary engine",
        "rotary-engine": "rotary engine",
        "Wankel engine": "rotary engine",
        "apex seals": "apex seals",
        "Apex seal": "apex seal",
        "rotor": "rotor",
        "reciprocating engine": "reciprocating engine",
        "piston engine": "reciprocating engine",
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
        "abc": "TTC",
        "ＡＢＣ": "TTC",
        "A B C": "TTC",
        "エービーシー": "TTC",
        "エイビーシー": "TTC",
        "エー・ビー・シー": "TTC",
        "エービーシーが": "TTCが",
        "ABCが": "TTCが",

        # 慣性補償 correction - single term
        "感性補償": "慣性補償",
        "感性保証": "慣性補償",
        "感性保障": "慣性補償",
        "完成保証": "慣性補償",
        "完成補償": "慣性補償",
        "完成保障": "慣性補償",
        "慣性保障": "慣性補償",
        "慣性補償性": "慣性補償",
        "慣性補償御": "慣性補償",
        "慣性補償償": "慣性補償",

        # 慣性補償制御 correction - control term
        "感性補償制御": "慣性補償制御",
        "感性保証制御": "慣性補償制御",
        "完成保証制御": "慣性補償制御",
        "完成補償制御": "慣性補償制御",
        "慣性保障制御": "慣性補償制御",
        "慣性補償性制御": "慣性補償制御",
        "慣性補償制御御": "慣性補償制御",
        "慣性補償制御制御": "慣性補償制御",
        "慣性補償制御について": "慣性補償制御について",

        # 慣性 / inertia context correction
        "感性の影響": "慣性の影響",
        "完成の影響": "慣性の影響",
        "慣性の駅": "慣性の影響",
        "完成の駅": "慣性の影響",
        "感性で位置": "慣性で位置",
        "完成で位置": "慣性で位置",
        "感性により": "慣性により",
        "完成により": "慣性により",

        # School / event terms
        # Safe direct corrections only.
        # Do NOT hard-replace チーム / 様々 / さまざま here;
        # those are handled only by the helper AI using context.
        "サマコース": "サマーコース",
        "サマー講座": "サマーコース",

        "ビナスASO": "ビヌスASO",
        "ビーナスASO": "ビヌスASO",
        "ネウスASO": "ビヌスASO",
        "ネウスアソ": "ビヌスASO",
        "ビヌスアソ": "ビヌスASO",
        "ビヌス麻生": "ビヌスASO",

        "ビナス大学": "ビヌス大学",
        "ビーナス大学": "ビヌス大学",
        "ネウス大学": "ビヌス大学",
        "ヴィヌス大学": "ビヌス大学",

        "ビナス": "ビヌス",
        "ビーナス": "ビヌス",
        "ネウス": "ビヌス",
        "ヴィヌス": "ビヌス",

        # BINUS ASO major names
        "AROI": "ARE",
        "Aroi": "ARE",
        "ARO": "ARE",
        "エーアールイー": "ARE",
        "エーアール": "ARE",

        "PDA": "PDE",
        "ADC": "PDE",
        "PD ": "PDE ",
        "PE ": "PDE ",
        "ピーディーイー": "PDE",
        "ピーディー": "PDE",

        "ビーイー": "BE",

        # CATIA / CAD / product design terms
        "キャティア": "CATIA",
        "カティア": "CATIA",
        "キャディア": "CATIA",
        "カディア": "CATIA",
        "カチア": "CATIA",
        "勝ティア": "CATIA",
        "勝ちア": "CATIA",
        "Catia": "CATIA",
        "catia": "CATIA",

        "キャド": "CAD",
        "cad": "CAD",
        "Computer Aided Design": "CAD",
        "Computer-Aided Design": "CAD",

        "スケッチヤー": "スケッチャー",
        "寸法高速": "寸法拘束",
        "寸法校則": "寸法拘束",
        "寸法公則": "寸法拘束",
        "幾何高速": "幾何拘束",
        "幾何校則": "幾何拘束",
        "記号拘束": "幾何拘束",
        "完全高速": "完全拘束",
        "完全校則": "完全拘束",
        "自由道": "自由度",

        "パッド": "Pad",
        "押出し": "押し出し",
        "filet": "フィレット",
        "Fillet": "フィレット",

        "チャンファー": "Chamfer",
        "シャンファー": "Chamfer",
        "chamfer": "Chamfer",
        "面どり": "面取り",

        "設計糸": "設計意図",
        "加工製": "加工性",
        "加工生": "加工性",

        # Jig / mechanical drawing terms
        "時空のCAD": "治具のCAD",
        "時空CAD": "治具CAD",
        "時空": "治具",
        "時具": "治具",
        "地具": "治具",
        "ジグ": "治具",
        "三面図面": "三面図",
        "3面図": "三面図",

        # Rotary engine terms
        "Rotary Engine": "ロータリーエンジン",
        "rotary engine": "ロータリーエンジン",
        "ロータリエンジン": "ロータリーエンジン",
        "ロタリーエンジン": "ロータリーエンジン",
        "ロータリー エンジン": "ロータリーエンジン",
        "ピストンエンジン": "レシプロエンジン",
        "アペックシール": "アペックスシール",

        # Common sentence cleanup
        "または急に止まると": "モーターが急に止まると",
        "急に止まると完成": "急に止まると、慣性",
        "位置がずる": "位置がずれる",
        "位置がずれます": "位置がずれます",
    }

    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)

    return cleaned.strip()


def light_domain_context_cleanup(original_text, translation_text, domain_mode):
    """
    Context-sensitive cleanup for terms that are dangerous to replace globally.

    Example:
    - 勝ち方 normally means "way to win", so we should not always replace it.
    - But in a CAD / product design classroom, when the lecture mentions
      parts, sketches, Pad, constraints, or modeling, 勝ち方 is often Whisper
      mishearing CATIA / キャティア.
    """
    original_text = (original_text or "").strip()
    translation_text = (translation_text or "").strip()
    domain = (domain_mode or "auto").lower()

    combined = f"{original_text}\n{translation_text}".lower()

    cad_context_words = [
        "catia",
        "cad",
        "sketch",
        "sketcher",
        "part",
        "parts",
        "pad",
        "extrusion",
        "extrude",
        "fillet",
        "chamfer",
        "hole",
        "constraint",
        "constraints",
        "dimensional",
        "geometric",
        "model",
        "modeling",
        "3d",
        "design",
        "product",
        "スケッチ",
        "スケッチャー",
        "パート",
        "部品",
        "寸法",
        "拘束",
        "幾何",
        "押し出し",
        "フィレット",
        "面取り",
        "設計",
        "形状",
        "モデル",
        "モデリング",
        "治具",
        "ジグ",
        "三面図",
        "図面",
        "製図",
    ]

    is_cad_domain = domain in ["auto", "cad", "product design"]
    has_cad_context = any(word in combined for word in cad_context_words)

    if is_cad_domain and has_cad_context:
        original_replacements = {
            "時空のCAD": "治具のCAD",
            "時空CAD": "治具CAD",
            "時空": "治具",
            "時具": "治具",
            "地具": "治具",
            "ジグ": "治具",
            "カチア": "CATIA",
            "勝ティア": "CATIA",
            "勝ちア": "CATIA",
            "キャティア": "CATIA",
            "カティア": "CATIA",
            "キャディア": "CATIA",
            "カディア": "CATIA",
        }

        translation_replacements = {
            "spacetime CAD": "jig CAD",
            "space-time CAD": "jig CAD",
            "space time CAD": "jig CAD",
            "spacetime": "jig",
            "space-time": "jig",
            "space time": "jig",
        }

        for wrong, correct in original_replacements.items():
            original_text = original_text.replace(wrong, correct)

        for wrong, correct in translation_replacements.items():
            translation_text = translation_text.replace(wrong, correct)

    return original_text.strip(), translation_text.strip()


def prepare_next_ai_check_after_new_live_text():
    """
    When new live speech arrives after an AI-corrected segment, keep the
    corrected caption visible and let the live worker continue from that
    corrected base. The next helper AI call will update the continued text.
    """
    if st.session_state.caption_stage != "ai_corrected":
        return

    st.session_state.correction_status = "pending"
    st.session_state.caption_stage = "raw_continuing"


def contains_japanese(text):
    if not text:
        return False

    for ch in text:
        cp = ord(ch)

        if (
            0x3040 <= cp <= 0x309F  # Hiragana
            or 0x30A0 <= cp <= 0x30FF  # Katakana
            or 0x4E00 <= cp <= 0x9FFF  # Kanji
        ):
            return True

    return False


def normalize_key_term_line(term, meaning):
    """
    Avoid useless English-to-English terms.
    Prefer Japanese technical source terms:
        ブレーキワイヤー = brake wire
        慣性補償 = inertia compensation
    Keep important acronyms:
        TTC = Time To Collision
    """
    term = (term or "").strip()
    meaning = (meaning or "").strip()

    if not term:
        return ""

    term = light_original_cleanup(term)
    meaning = light_caption_cleanup(meaning)

    allowed_acronyms = {
        "TTC",
        "AEB",
        "ADAS",
        "ABS",
        "ECU",
        "CAN",
        "PWM",
        "PID",
        "IPM",
        "YOLO",
        "ARE",
        "PDE",
        "BE",
        "CATIA",
        "CAD",
    }

    # If the LLM gives English-to-English, convert common terms back to
    # Japanese-source display for classroom use.
    english_to_jp = {
        "inertia compensation control": ("慣性補償制御", "inertia compensation control"),
        "inertia compensation": ("慣性補償", "inertia compensation"),
        "inertia": ("慣性", "inertia"),
        "brake wire": ("ブレーキワイヤー", "brake wire"),
        "brake cable": ("ブレーキワイヤー", "brake wire"),
        "servo motor": ("サーボモーター", "servo motor"),
        "braking force": ("制動力", "braking force"),
        "emergency braking": ("急ブレーキ", "emergency braking"),
        "time to collision": ("TTC", "Time To Collision"),
        "following distance": ("車間距離", "following distance"),
        "relative speed": ("相対速度", "relative speed"),
        "lever": ("レバー", "lever"),
        "summer course": ("サマーコース", "Summer Course"),
        "binus aso": ("ビヌスASO", "BINUS ASO"),
        "binus university": ("ビヌス大学", "BINUS University"),
        "binus": ("ビヌス", "BINUS"),
        "automotive and robotics engineering": ("ARE", "Automotive and Robotics Engineering"),
        "automotive robotics engineering": ("ARE", "Automotive and Robotics Engineering"),
        "aroi": ("ARE", "Automotive and Robotics Engineering"),
        "are": ("ARE", "Automotive and Robotics Engineering"),
        "product design engineering": ("PDE", "Product Design Engineering"),
        "product design": ("PDE", "Product Design Engineering"),
        "pda": ("PDE", "Product Design Engineering"),
        "adc": ("PDE", "Product Design Engineering"),
        "pde": ("PDE", "Product Design Engineering"),
        "business engineering": ("BE", "Business Engineering"),
        "business engineer": ("BE", "Business Engineering"),
        "be": ("BE", "Business Engineering"),

        "catia": ("CATIA", "CATIA"),
        "cad": ("CAD", "Computer-Aided Design"),
        "computer aided design": ("CAD", "Computer-Aided Design"),
        "computer-aided design": ("CAD", "Computer-Aided Design"),
        "sketcher": ("スケッチャー", "Sketcher"),
        "dimensional constraint": ("寸法拘束", "dimensional constraint"),
        "geometric constraint": ("幾何拘束", "geometric constraint"),
        "fully constrained": ("完全拘束", "fully constrained"),
        "degrees of freedom": ("自由度", "degrees of freedom"),
        "pad": ("Pad", "Pad / extrusion"),
        "extrusion": ("押し出し", "extrusion"),
        "fillet": ("フィレット", "fillet"),
        "chamfer": ("Chamfer", "chamfer"),
        "chamfering": ("面取り", "chamfering"),
        "design intent": ("設計意図", "design intent"),
        "manufacturability": ("加工性", "manufacturability"),
        "jig": ("治具", "jig"),
        "fixture": ("治具", "jig / fixture"),
        "spacetime": ("治具", "jig"),
        "space-time": ("治具", "jig"),
        "space time": ("治具", "jig"),
        "jig cad": ("治具のCAD", "jig CAD model"),
        "spacetime cad": ("治具のCAD", "jig CAD model"),
        "space-time cad": ("治具のCAD", "jig CAD model"),
        "space time cad": ("治具のCAD", "jig CAD model"),
        "three-view drawing": ("三面図", "three-view drawing / orthographic drawing"),
        "three view drawing": ("三面図", "three-view drawing / orthographic drawing"),
        "orthographic drawing": ("三面図", "three-view drawing / orthographic drawing"),
        "three views": ("三面図", "three-view drawing / orthographic drawing"),
        "triangle": ("三角形", "triangle"),
        "triangular shape": ("三角形", "triangle"),
        "rectangle": ("長方形", "rectangle"),
        "square": ("正方形", "square"),
        "circle": ("円", "circle"),
        "radius": ("半径", "radius"),
        "diameter": ("直径", "diameter"),
        "rotary engine": ("ロータリーエンジン", "rotary engine"),
        "wankel engine": ("ロータリーエンジン", "rotary engine"),
        "reciprocating engine": ("レシプロエンジン", "reciprocating engine"),
        "piston engine": ("レシプロエンジン", "reciprocating engine"),
        "rotor": ("ローター", "rotor"),
        "apex seal": ("アペックスシール", "apex seal"),
    }

    lowered_term = term.lower()
    lowered_meaning = meaning.lower()

    for key, value in english_to_jp.items():
        if key in lowered_term:
            term, meaning = value
            break

        if key in lowered_meaning and not contains_japanese(term):
            term, meaning = value
            break

    if contains_japanese(term):
        if meaning:
            return f"{term} = {meaning}"
        return term

    if term.upper() in allowed_acronyms:
        term = term.upper()

        if not meaning:
            if term == "TTC":
                meaning = "Time To Collision"
            elif term == "AEB":
                meaning = "Autonomous Emergency Braking"
            elif term == "ADAS":
                meaning = "Advanced Driver Assistance Systems"
            elif term == "ARE":
                meaning = "Automotive and Robotics Engineering"
            elif term == "PDE":
                meaning = "Product Design Engineering"
            elif term == "BE":
                meaning = "Business Engineering"
            elif term == "CATIA":
                meaning = "CAD software for 3D product design"
            elif term == "CAD":
                meaning = "Computer-Aided Design"

        if meaning:
            return f"{term} = {meaning}"
        return term

    # Drop English-to-English non-acronym terms.
    return ""


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


def source_text_matches_for_correction(current_source, corrected_source):
    """
    Live captions change every rerun, so exact equality is too strict.
    This allows LLM corrected Japanese/English to apply when the current
    live text is still basically the same segment.
    """
    current_source = (current_source or "").strip()
    corrected_source = (corrected_source or "").strip()

    if not current_source or not corrected_source:
        return False

    if current_source == corrected_source:
        return True

    if corrected_source in current_source:
        return True

    if current_source in corrected_source:
        return True

    # Japanese/English live captions may change by a few characters.
    # Compare character overlap.
    current_set = set(current_source)
    corrected_set = set(corrected_source)

    if not current_set or not corrected_set:
        return False

    overlap_a = len(current_set & corrected_set) / max(1, len(corrected_set))
    overlap_b = len(current_set & corrected_set) / max(1, len(current_set))

    # Stricter match:
    # old AI correction should not keep overriding a new live segment.
    return overlap_a >= 0.90 and overlap_b >= 0.80


def extract_latest_pair_from_llm_source(source_text):
    """
    Source text is built like:
        Japanese: ...
        English: ...

        ---

        Japanese: latest...
        English: latest...

    Return the latest Japanese/English pair the AI actually saw.
    """
    source_text = (source_text or "").strip()

    if not source_text:
        return "", ""

    chunks = [
        chunk.strip()
        for chunk in source_text.split("\n\n---\n\n")
        if chunk.strip()
    ]

    if not chunks:
        chunks = [source_text]

    latest = chunks[-1]
    latest_japanese = ""
    latest_english = ""

    lines = latest.splitlines()
    current_label = None
    jp_lines = []
    en_lines = []

    for line in lines:
        if line.startswith("Japanese:"):
            current_label = "jp"
            value = line.replace("Japanese:", "", 1).strip()
            if value:
                jp_lines.append(value)
            continue

        if line.startswith("English:"):
            current_label = "en"
            value = line.replace("English:", "", 1).strip()
            if value:
                en_lines.append(value)
            continue

        if current_label == "jp":
            jp_lines.append(line.strip())
        elif current_label == "en":
            en_lines.append(line.strip())

    latest_japanese = " ".join([x for x in jp_lines if x]).strip()
    latest_english = " ".join([x for x in en_lines if x]).strip()

    return latest_japanese, latest_english


def text_char_overlap_ratio(shorter_text, longer_text):
    shorter_text = (shorter_text or "").strip()
    longer_text = (longer_text or "").strip()

    if not shorter_text or not longer_text:
        return 0.0

    shorter_set = set(shorter_text)
    longer_set = set(longer_text)

    if not shorter_set:
        return 0.0

    return len(shorter_set & longer_set) / max(1, len(shorter_set))


def ai_text_is_full_enough(live_text, ai_text, min_ratio=0.72):
    """
    True when the AI text looks like a full corrected caption, not just
    a short correction phrase.
    """
    live_text = (live_text or "").strip()
    ai_text = (ai_text or "").strip()

    if not ai_text:
        return False

    if not live_text:
        return True

    live_len = len(live_text)
    ai_len = len(ai_text)

    if ai_len >= live_len * min_ratio:
        return True

    if live_text in ai_text:
        return True

    if ai_len >= 40 and text_char_overlap_ratio(ai_text, live_text) >= 0.85:
        return True

    return False


def merge_ai_text_preserve_current(
    current_text,
    ai_source_text,
    ai_corrected_text,
    corrections,
    text_kind,
    domain_mode,
):
    """
    Patch the current live caption without deleting newer text.

    Priority:
    1. Apply correction pairs to the current caption.
    2. If the AI source text is inside the current caption, replace ONLY
       that old source segment with the corrected segment.
    3. If AI corrected text looks like a full caption, allow full replace.
    4. Otherwise keep the patched current caption.
    """
    current_text = (current_text or "").strip()
    ai_source_text = (ai_source_text or "").strip()
    ai_corrected_text = (ai_corrected_text or "").strip()

    patched_current = apply_llm_corrections(current_text, corrections)

    if ai_source_text and ai_corrected_text and ai_source_text in patched_current:
        merged = patched_current.replace(ai_source_text, ai_corrected_text, 1)

    elif ai_corrected_text and ai_text_is_full_enough(patched_current, ai_corrected_text):
        merged = ai_corrected_text

    else:
        merged = patched_current

    if text_kind == "original":
        merged = light_original_cleanup(merged)
        merged, _ = light_domain_context_cleanup(
            merged,
            "",
            domain_mode,
        )
        return merged.strip()

    if text_kind == "translation":
        merged = light_caption_cleanup(merged)
        _, merged = light_domain_context_cleanup(
            "",
            merged,
            domain_mode,
        )
        return merged.strip()

    return merged.strip()


def merge_ai_result_into_live_caption(
    live_original,
    live_translation,
    ai_source_text,
    ai_corrected_original,
    ai_corrected_translation,
    corrections,
    domain_mode,
):
    """
    Merge the second AI result into the current live caption safely.
    """
    source_original, source_translation = extract_latest_pair_from_llm_source(ai_source_text)

    merged_original = merge_ai_text_preserve_current(
        current_text=live_original,
        ai_source_text=source_original,
        ai_corrected_text=ai_corrected_original,
        corrections=corrections,
        text_kind="original",
        domain_mode=domain_mode,
    )

    merged_translation = merge_ai_text_preserve_current(
        current_text=live_translation,
        ai_source_text=source_translation,
        ai_corrected_text=ai_corrected_translation,
        corrections=corrections,
        text_kind="translation",
        domain_mode=domain_mode,
    )

    return merged_original, merged_translation


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


def compact_text(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].strip()


def is_short_japanese_term_query(text):
    """
    True when the current speech looks like the interpreter/speaker only said
    a short Japanese word or phrase, for example:
        三角形
        慣性補償
        寸法拘束
        面取りって何
    In that case we should still run key-term support even if the full
    sentence is too short for normal caption correction.
    """
    text = (text or "").strip()

    if not text:
        return False

    if not is_japanese_text(text):
        return False

    # Ignore long lecture sentences. This is only for short term/phrase.
    if len(text) > 45:
        return False

    # If it has many sentence separators, it is probably normal speech.
    separator_count = (
        text.count("。")
        + text.count("、")
        + text.count("？")
        + text.count("?")
        + text.count("！")
        + text.count("!")
    )

    if separator_count >= 3:
        return False

    return True


def build_slim_llm_context(context_chunks, current_original, current_translation):
    """
    Small context for Groq free-tier TPM.
    Keep only last finished chunk + current chunk.
    """
    chunks = list(context_chunks or [])[-1:]
    current_chunk = make_context_chunk(current_original, current_translation)
    if current_chunk:
        chunks.append(current_chunk)
    slim = "\n\n---\n\n".join([c for c in chunks if c]).strip()
    return compact_text(slim, MAX_GROQ_CONTEXT_CHARS)


def make_compact_domain_context(domain_mode, self_context=""):
    """
    Give the text models only broad topic context.

    Do not place vocabulary lists here. Model prompts containing unrelated
    technical terms caused those terms to be inserted into captions.
    """
    domain = normalize_domain_name(domain_mode)
    self_context = compact_text(self_context, 220)

    if domain == "automotive":
        base = "Domain: automotive engineering and vehicle-control lecture."
    elif domain == "cad":
        base = "Domain: CAD and mechanical-design lecture."
    elif domain == "product design":
        base = "Domain: product-design and manufacturing lecture."
    elif domain == "school":
        base = "Domain: school program or academic event."
    else:
        base = "Domain: general Japanese technical classroom."

    if self_context:
        return f"{base}\nOptional background only: {self_context}"

    return base

def shorten_error_for_ui(message, max_chars=180):
    message = str(message or "").strip()
    if len(message) <= max_chars:
        return message
    return message[:max_chars] + " ..."


def contains_kanji(text):
    for ch in text or "":
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return True
    return False


def lookup_reading_for_term(term, provided_reading=""):
    """
    Use the glossary reading when displaying kanji key terms.
    Example:
        三面図 -> さんめんず
        治具 -> じぐ
    """
    term = (term or "").strip()
    provided_reading = (provided_reading or "").strip()

    if provided_reading:
        return provided_reading

    if not term:
        return ""

    try:
        entries = load_glossary_entries(DEFAULT_TERMS_FILE)
    except Exception:
        entries = []

    for row in entries:
        jp = str(row.get("jp", "")).strip()
        reading = str(row.get("reading", "")).strip()

        if jp == term and reading:
            return reading

    return ""


def key_term_supported_by_source(item, source_text):
    """
    Final UI guard against hallucinated key terms.

    A key term can be shown when:
    - its glossary source_match still appears in the Japanese source, or
    - the displayed Japanese term itself appears in the Japanese source.

    English translation is intentionally NOT used as proof.
    """
    source_text = source_text or ""

    if not source_text:
        return False

    source_match = str(item.get("source_match", item.get("matched_candidate", ""))).strip()

    if source_match and glossary_candidate_matches(source_match, source_text):
        return True

    term = str(item.get("term", item.get("jp", ""))).strip()
    meaning = str(item.get("meaning", item.get("en", ""))).strip()
    line = normalize_key_term_line(term, meaning)

    if not line:
        return False

    display_term = line.split("=", 1)[0].strip()

    if display_term and glossary_candidate_matches(display_term, source_text):
        return True

    # If term is Japanese and appears directly after cleanup, keep it.
    cleaned_term = light_original_cleanup(term)

    if cleaned_term and contains_japanese(cleaned_term) and cleaned_term in source_text:
        return True

    return False



def key_term_display_allowed(item, source_text):
    """
    Show only terms supported by the currently displayed corrected Japanese.

    There is no cross-phrase staleness hold. When the visible phrase changes,
    unsupported old terms disappear immediately.
    """
    term = str(item.get("term", item.get("jp", ""))).strip()
    evidence = str(
        item.get(
            "evidence",
            item.get("source_match", item.get("matched_candidate", "")),
        )
    ).strip()

    if not term:
        return False

    if not _literal_term_in_text(term, source_text):
        return False

    # The current corrected Japanese is enough for display, but every stored
    # AI-confirmed term should also carry raw evidence.
    if item.get("confidence") and not evidence:
        return False

    item["last_confirmed_at"] = time.time()
    return True

def format_key_term_line(term, meaning, show_meaning=True, reading=""):
    """
    Display key terms with furigana-like hiragana in parentheses when the
    source term contains kanji.

    Example:
        三面図 (さんめんず)： three-view drawing / orthographic drawing
        治具 (じぐ)： jig
        CAD = Computer-Aided Design
    """
    line = normalize_key_term_line(term, meaning)

    if not line:
        return ""

    if "=" in line:
        display_term, display_meaning = [part.strip() for part in line.split("=", 1)]
    else:
        display_term = line.strip()
        display_meaning = ""

    display_reading = lookup_reading_for_term(display_term, reading)

    if display_reading and contains_kanji(display_term):
        display_term = f"{display_term} ({display_reading})"

    if not show_meaning:
        return display_term

    if display_meaning:
        if display_reading and contains_kanji(line.split("=", 1)[0].strip()):
            return f"{display_term}： {display_meaning}"
        return f"{display_term} = {display_meaning}"

    return display_term


# ============================================================
# Selected domain context for Groq helper
# ============================================================

def make_selected_domain_context(domain_mode):
    """
    Fixed background context sent to the Groq helper AI.
    This makes the sidebar Technical domain actually guide correction,
    not only rule-based cleanup.
    """
    domain = (domain_mode or "auto").lower()

    if domain == "cad":
        return """
Selected technical domain: CAD / CATIA classroom.

The speaker is probably explaining CAD/CATIA operations such as:
- CATIA, CAD, Sketcher, part file, XY plane
- sketch, line, circle, rectangle, profile
- dimensional constraint, geometric constraint, fully constrained sketch
- degrees of freedom, origin, horizontal, vertical, center alignment
- Pad, extrusion, Pocket, Hole, Fillet, Chamfer, chamfering
- design intent, dimensions, shape, modeling, 3D model, manufacturability

Correction priorities:
- If the Japanese sounds like 勝ち方 / 書き方 / キャリア in a CAD sentence, it is probably CATIA.
- If English says "way to win" in a CAD sentence, it is probably CATIA.
- If English says "line enters" or similar when Japanese mentions スケッチャー, it probably means "enter Sketcher".
- Do not force CATIA when the sentence is really about winning, writing method, or career.
""".strip()

    if domain == "product design":
        return """
Selected technical domain: Product Design Engineering / CAD modeling classroom.

The speaker is probably explaining:
- Product Design Engineering, product design process, design intent
- CATIA, CAD modeling, Sketcher, dimensions, constraints
- usability, strength, material, cost, manufacturability
- prototype, product development, shape, part design
- Pad, extrusion, Fillet, Chamfer, Hole, assembly basics

Correction priorities:
- Preserve Product Design Engineering as a program/major name when relevant.
- If the lecture mentions CAD, parts, sketching, or modeling, CATIA-related terms are likely.
- Do not replace normal business/design words unless the current sentence clearly supports it.
""".strip()

    if domain == "automotive":
        return """
Selected technical domain: Automotive engineering classroom.

The speaker is probably explaining:
- vehicle systems, braking system, drivetrain, steering, suspension
- AEB, ADAS, TTC, Time To Collision, distance estimation
- servo motor, brake wire, braking force, emergency braking
- inertia, inertia compensation, control, motor, sensor
- rotary engine, reciprocating engine, rotor, apex seal

Correction priorities:
- If the caption says ABC in AEB/TTC context, correct it to TTC.
- Preserve AEB, ADAS, TTC, ECU, CAN, PWM, PID as technical acronyms.
- If the lecture mentions rotary engine, preserve ロータリーエンジン, ローター, アペックスシール.
""".strip()

    if domain in ["school", "school/event", "event"]:
        return """
Selected technical domain: school event / Summer Course / BINUS ASO.

The speaker is probably explaining:
- Summer Course, BINUS, BINUS ASO, BINUS University
- students coming to Japan, lectures, internship, training, Japanese culture
- ARE, PDE, BE majors only when the topic is clearly BINUS ASO majors

Correction priorities:
- サマーコース = Summer Course.
- ビヌス大学 = BINUS University.
- Do not force Summer Course/BINUS terms when the current sentence is not about this event.
""".strip()

    return """
Selected technical domain: mixed Japanese technical classroom.

Possible topics include:
- automotive engineering, AEB, TTC, braking systems, inertia compensation
- CAD/CATIA, Sketcher, dimensional constraints, geometric constraints
- Product Design Engineering, design intent, manufacturability
- BINUS ASO, BINUS University, Summer Course, ARE, PDE, BE

Correction priorities:
- Use the recent Japanese/English context to decide which domain is active.
- Do not force a domain term unless it fits the current sentence.
- Preserve important acronyms and proper nouns.
""".strip()


# ============================================================
# LLM Interpreter Support - Groq second AI
# ============================================================


def parse_llm_json(text):
    empty = {
        "corrected_japanese_original": "",
        "corrected_english_caption": "",
        "is_unclear": False,
        "unclear_reason": "",
        "confirmed_terms": [],
        "key_terms": [],
        "corrections": [],
        "parse_ok": False,
        "raw_text": text or "",
    }

    if not text:
        return empty

    cleaned = str(text).strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx >= 0 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1].strip()

    try:
        data = json.loads(cleaned)

        confirmed_terms = data.get(
            "confirmed_terms",
            data.get("key_terms", []),
        )
        if not isinstance(confirmed_terms, list):
            confirmed_terms = []

        corrections = data.get("corrections", [])
        if not isinstance(corrections, list):
            corrections = []

        unclear_value = data.get(
            "is_unclear",
            data.get("uncertain", False),
        )

        return {
            "corrected_japanese_original": str(
                data.get(
                    "corrected_japanese_original",
                    data.get("corrected_japanese", ""),
                )
            ).strip(),
            "corrected_english_caption": str(
                data.get(
                    "corrected_english_caption",
                    data.get("faithful_english", ""),
                )
            ).strip(),
            "is_unclear": (
                str(unclear_value).strip().lower()
                in ["true", "1", "yes", "y"]
            ),
            "unclear_reason": str(
                data.get(
                    "unclear_reason",
                    data.get("uncertain_reason", ""),
                )
            ).strip(),
            "confirmed_terms": confirmed_terms,
            "key_terms": confirmed_terms,
            "corrections": corrections,
            "parse_ok": True,
            "raw_text": text or "",
        }

    except Exception:
        return empty


def llm_hint_worker(
    result_queue,
    api_key,
    model_name,
    current_japanese,
    current_translation,
    previous_context,
    key_terms,
    full_glossary_entries,
    domain_mode,
    class_context="",
    self_context="",
    source_version=-1,
):
    """
    Evidence-based second pass.

    The model may repair ASR errors, but the program validates its Japanese,
    correction pairs, and confirmed key terms before anything reaches the UI.
    """
    try:
        client = Groq(api_key=api_key)

        current_japanese = compact_text(current_japanese, MAX_GROQ_CONTEXT_CHARS)
        current_translation = compact_text(
            current_translation,
            MAX_GROQ_TRANSLATION_CHARS,
        )
        previous_context = compact_text(previous_context, 500)
        class_context = compact_text(class_context, 500)
        self_context = compact_text(self_context, 350)

        glossary_lines = []
        for item in (key_terms or [])[:MAX_GROQ_GLOSSARY_TERMS]:
            jp = str(item.get("jp", "")).strip()
            en = str(item.get("en", "")).strip()
            variants = item.get("recognition_variants")
            if variants is None:
                variants = recognition_variants_for_entry(item)

            variants = [
                str(value).strip()
                for value in (variants or [])
                if str(value).strip()
            ]
            evidence = str(item.get("matched_evidence", "") or "").strip()

            if not jp or not en:
                continue

            line = f"canonical={jp} | English={en}"
            if variants:
                line += " | possible ASR forms=" + "; ".join(variants[:8])
            if evidence:
                line += f" | raw evidence={evidence}"
            glossary_lines.append(line)

        glossary_text = "\n".join(glossary_lines) or "None"

        prompt = f"""
Return ONLY one JSON object. No markdown and no commentary.

You are a conservative second-pass corrector for Japanese technical captions.
The raw Japanese below is the primary evidence. Correct only likely ASR errors.
Do not rewrite for style.

STRICT FIDELITY RULES:
1. Output exactly one corrected version of the current phrase. Do not repeat it.
2. Preserve the speaker's actors, actions, particles, negation, numbers,
   acronyms, sentence order, and clause count.
3. Do not add explanations such as 「このように」, 「については」,
   "this is how", examples, or background knowledge unless they exist in raw.
4. Use a canonical glossary spelling only when the Candidate glossary shows
   exact raw evidence for that entry.
5. Candidate glossary entries are possibilities only. Never insert a candidate
   merely because it belongs to the selected domain.
6. English must translate corrected Japanese only. Do not summarize or explain.
7. If the intended phrase is uncertain, preserve the raw wording where
   possible and set is_unclear=true instead of guessing.

CONFIRMED TERM RULES:
- Include a term only if it appears literally in corrected_japanese_original.
- Every term must include "evidence", copied exactly from the raw Japanese.
- Evidence may be the canonical term or a plausible ASR form.
- If no exact raw evidence exists, omit the term.
- Use the glossary English meaning exactly.
- confidence must be "high" or "medium".

CORRECTION RULES:
- Every corrections[].wrong must be an exact substring of raw Japanese.
- Every corrections[].correct must be an exact substring of corrected Japanese.
- Do not list stylistic edits as corrections.

Selected context:
{class_context}

User context:
{self_context or "None"}

Previous phrase for context only; do not include it in output:
{previous_context or "None"}

Raw current Japanese:
{current_japanese}

Initial English translation:
{current_translation or "None"}

Candidate glossary:
{glossary_text}

Required JSON schema:
{{
  "corrected_japanese_original": "",
  "corrected_english_caption": "",
  "is_unclear": false,
  "unclear_reason": "",
  "confirmed_terms": [
    {{
      "term": "",
      "meaning": "",
      "evidence": "",
      "confidence": "high"
    }}
  ],
  "corrections": [
    {{
      "wrong": "",
      "correct": "",
      "reason": ""
    }}
  ]
}}
""".strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative JSON-only Japanese ASR correction "
                    "and faithful translation engine."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response_text = ""
        last_error = ""

        for response_format in [{"type": "json_object"}, None]:
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 650,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                completion = client.chat.completions.create(**kwargs)
                response_text = completion.choices[0].message.content or ""

                if response_text.strip():
                    break

            except Exception as exc:
                last_error = str(exc)

        if not response_text.strip():
            raise RuntimeError(last_error or "Groq returned empty text")

        parsed = parse_llm_json(response_text)

        if not parsed.get("parse_ok"):
            parsed["corrected_japanese_original"] = current_japanese
            parsed["corrected_english_caption"] = current_translation
            parsed["is_unclear"] = True
            parsed["unclear_reason"] = (
                "Second-pass model returned invalid JSON; raw caption kept."
            )

        sanitized = sanitize_second_pass_output(
            raw_japanese=current_japanese,
            raw_english=current_translation,
            parsed=parsed,
            glossary_entries=full_glossary_entries,
            domain_mode=domain_mode,
        )

        result_queue.put({
            "type": "llm_hint",
            "corrected_japanese_original": sanitized[
                "corrected_japanese_original"
            ],
            "corrected_english_caption": sanitized[
                "corrected_english_caption"
            ],
            "is_unclear": sanitized["is_unclear"],
            "unclear_reason": sanitized["unclear_reason"],
            "source_text": make_context_chunk(
                current_japanese,
                current_translation,
            ),
            "source_japanese": current_japanese,
            "source_translation": current_translation,
            "source_version": int(source_version),
            "key_terms": sanitized["confirmed_terms"],
            "confirmed_terms": sanitized["confirmed_terms"],
            "corrections": sanitized["corrections"],
            "used_model": model_name,
            "raw_response_preview": response_text[:700],
            "safety_reason": sanitized.get("safety_reason", ""),
        })

    except Exception as exc:
        result_queue.put({
            "type": "llm_error",
            "message": shorten_error_for_ui(str(exc)),
            "source_version": int(source_version),
        })

# ============================================================
# Audio processor
# ============================================================

class AudioProcessor:
    """Convert browser audio to mono 16 kHz PCM16 exactly once."""

    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX_CHUNKS)
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=AUDIO_INPUT_SAMPLE_RATE,
        )
        self.frame_count = 0
        self.dropped_audio_chunks = 0
        self.last_audio_time = 0.0
        self.last_audio_level = 0.0

    def _put_latest_audio(self, audio_bytes):
        """Keep latency bounded by dropping the oldest chunk if necessary."""
        if not audio_bytes:
            return

        try:
            self.audio_queue.put_nowait(audio_bytes)
            return
        except queue.Full:
            pass

        try:
            self.audio_queue.get_nowait()
            self.dropped_audio_chunks += 1
        except queue.Empty:
            pass

        try:
            self.audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            self.dropped_audio_chunks += 1

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            resampled_frames = self.resampler.resample(frame)

            for resampled_frame in resampled_frames:
                audio = resampled_frame.to_ndarray().reshape(-1)

                if audio.size == 0:
                    continue

                pcm16 = np.ascontiguousarray(audio, dtype="<i2")
                self._put_latest_audio(pcm16.tobytes())

                self.frame_count += 1
                self.last_audio_time = time.time()

                try:
                    self.last_audio_level = float(
                        np.sqrt(np.mean(np.square(pcm16.astype(np.float32))))
                    )
                except Exception:
                    self.last_audio_level = 0.0

        except Exception as exc:
            logging.debug("Audio processing error: %s", exc)

        return frame


# ============================================================
# Groq Whisper live-caption worker
# ============================================================

def pcm16_rms(pcm_bytes):
    if not pcm_bytes:
        return 0.0

    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    if samples.size == 0:
        return 0.0

    values = samples.astype(np.float32)
    return float(np.sqrt(np.mean(values * values)))


def pcm16_to_wav_bytes(pcm_bytes, sample_rate=AUDIO_INPUT_SAMPLE_RATE):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM_BYTES_PER_SAMPLE)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


def normalize_domain_name(domain_mode):
    domain = (domain_mode or "auto").strip().lower()
    if domain in {"school/event", "event"}:
        return "school"
    return domain


def glossary_entry_matches_domain(entry, domain_mode):
    selected = normalize_domain_name(domain_mode)
    entry_domain = normalize_domain_name(entry.get("domain", ""))

    if selected == "auto":
        return True
    if not entry_domain:
        return True
    return entry_domain == selected


def build_whisper_prompt(glossary_entries, domain_mode, self_context="", previous_text=""):
    """
    Return no Whisper prompt.

    Vocabulary prompts improved isolated spellings but contaminated continuous
    speech by making Whisper emit terms that were never spoken. Technical term
    recovery is handled after transcription by the evidence-based second pass.
    """
    return ""

def select_second_pass_glossary_candidates(
    glossary_entries,
    domain_mode,
    source_text,
    max_terms=16,
):
    """
    Return only glossary entries supported by the current raw ASR chunk.

    No domain-priority filler entries are added. This prevents the second pass
    from seeing CATIA, BINUS ASO, AEB, or TTC unless the raw chunk contains the
    canonical term or one of its approved recognition variants.
    """
    source_text = str(source_text or "")
    candidates = []
    seen = set()

    for entry in glossary_entries or []:
        if not glossary_entry_matches_domain(entry, domain_mode):
            continue

        jp = str(entry.get("jp", "")).strip()
        en = str(entry.get("en", "")).strip()
        if not jp or not en:
            continue

        evidence = source_evidence_for_term(jp, source_text, glossary_entries)
        if not evidence:
            continue

        key = (jp, en)
        if key in seen:
            continue

        enriched = dict(entry)
        enriched["recognition_variants"] = recognition_variants_for_entry(entry)
        enriched["matched_evidence"] = evidence
        candidates.append(enriched)
        seen.add(key)

    candidates.sort(
        key=lambda item: glossary_priority(
            str(item.get("jp", "")).strip(),
            domain_mode,
        )
    )
    return candidates[:max_terms]

def apply_glossary_source_corrections(text, glossary_entries, domain_mode):
    """
    Apply only deterministic, low-risk ASR corrections.

    Ambiguous observed forms such as メーカー -> 面取り and リグ -> 治具 are
    intentionally left to the evidence-based second pass.
    """
    corrected = light_original_cleanup(text or "")
    replacements = []

    ambiguous_second_pass_only = {
        ("治具", "リグ"),
        ("治具", "GQ"),
        ("面取り", "メーカー"),
        ("応答性", "オートセット"),
        ("車間距離", "Shaken Carrier"),
        ("相対速度", "Sorter"),
        ("相対速度", "temperate speed"),
        ("慣性補償制御", "完成報告書を制御"),
    }

    for entry in glossary_entries or []:
        if not glossary_entry_matches_domain(entry, domain_mode):
            continue

        canonical = str(entry.get("jp", "")).strip()
        if not canonical:
            continue

        for wrong in recognition_variants_for_entry(entry):
            if len(wrong) < 2 or wrong == canonical:
                continue
            if (canonical, wrong) in ambiguous_second_pass_only:
                continue
            replacements.append((wrong, canonical))

    replacements.sort(key=lambda item: len(item[0]), reverse=True)

    for wrong, canonical in replacements:
        if _is_ascii_token(wrong):
            corrected = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(wrong)}(?![A-Za-z0-9])",
                canonical,
                corrected,
                flags=re.IGNORECASE,
            )
        else:
            corrected = corrected.replace(wrong, canonical)

    # Safe phrase-level repairs observed in repeated tests.
    phrase_repairs = {
        "ちさくナルト": "小さくなると",
        "ちさくなると": "小さくなると",
        "小さくナルト": "小さくなると",
    }
    for wrong, canonical in phrase_repairs.items():
        corrected = corrected.replace(wrong, canonical)

    corrected = light_original_cleanup(corrected)
    corrected, _ = light_domain_context_cleanup(corrected, "", domain_mode)
    return corrected.strip()

def merge_overlapping_japanese(previous_text, new_text, max_chars=MAX_ORIGINAL_CHARS * 2):
    previous = (previous_text or "").strip()
    new = (new_text or "").strip()

    if not previous:
        return new
    if not new:
        return previous
    if new in previous:
        return previous
    if previous in new:
        return trim_caption_soft(new, max_chars)

    max_overlap = min(len(previous), len(new), 80)
    overlap = 0

    for size in range(max_overlap, 1, -1):
        if previous[-size:] == new[:size]:
            overlap = size
            break

    if overlap:
        combined = previous + new[overlap:]
    else:
        # Whisper chunks usually end at phrase boundaries. Avoid gluing two
        # independent phrases without punctuation when no overlap is found.
        separator = "" if previous.endswith(("。", "？", "！", "、")) else "。"
        combined = previous + separator + new

    return trim_caption_soft(combined, max_chars)


def matched_translation_glossary(text, glossary_entries, domain_mode, max_terms=16):
    matches = []
    seen = set()

    for entry in glossary_entries or []:
        if not glossary_entry_matches_domain(entry, domain_mode):
            continue

        jp = str(entry.get("jp", "")).strip()
        en = str(entry.get("en", "")).strip()

        if not jp or not en or jp not in (text or ""):
            continue

        key = (jp, en)
        if key in seen:
            continue

        matches.append(f"{jp} = {en}")
        seen.add(key)

        if len(matches) >= max_terms:
            break

    return matches


def clean_plain_translation(text):
    cleaned = (text or "").strip()
    for prefix in ["English:", "Translation:", "English translation:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    cleaned = cleaned.strip('"').strip()
    return light_caption_cleanup(cleaned)


def translate_japanese_with_groq(
    client,
    japanese_text,
    previous_japanese,
    glossary_entries,
    domain_mode,
    self_context,
):
    glossary_lines = matched_translation_glossary(
        japanese_text,
        glossary_entries,
        domain_mode,
    )

    domain_context = make_compact_domain_context(domain_mode, self_context)
    glossary_block = "\n".join(glossary_lines) if glossary_lines else "None"
    previous = compact_text(previous_japanese, 180)

    system_prompt = (
        "You are a precise Japanese-to-English technical caption translator. "
        "Translate only the current Japanese caption. Preserve actors, actions, "
        "negation, numbers, technical meaning, and acronyms. Use the glossary "
        "exactly when its Japanese term appears. Do not explain, summarize, "
        "guess missing speech, or add labels. Output English only."
    )

    user_prompt = (
        f"{domain_context}\n\n"
        f"Required glossary:\n{glossary_block}\n\n"
        f"Previous Japanese context (do not translate):\n{previous or 'None'}\n\n"
        f"Current Japanese caption:\n{japanese_text}"
    )

    completion = client.chat.completions.create(
        model=GROQ_TRANSLATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=320,
    )

    output = completion.choices[0].message.content or ""
    return clean_plain_translation(output)


def _groq_object_to_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                data = method()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    try:
        return dict(value)
    except Exception:
        return {}


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_whisper_metrics(transcription):
    data = _groq_object_to_dict(transcription)
    segments = data.get("segments") or getattr(transcription, "segments", None) or []
    segment_dicts = [_groq_object_to_dict(segment) for segment in segments]

    no_speech_values = []
    avg_logprob_values = []
    compression_values = []

    for segment in segment_dicts:
        no_speech = _safe_float(segment.get("no_speech_prob"))
        avg_logprob = _safe_float(segment.get("avg_logprob"))
        compression = _safe_float(segment.get("compression_ratio"))
        if no_speech is not None:
            no_speech_values.append(no_speech)
        if avg_logprob is not None:
            avg_logprob_values.append(avg_logprob)
        if compression is not None:
            compression_values.append(compression)

    # Some responses expose metrics at the top level rather than per segment.
    top_no_speech = _safe_float(data.get("no_speech_prob"))
    top_avg_logprob = _safe_float(data.get("avg_logprob"))
    top_compression = _safe_float(data.get("compression_ratio"))
    if top_no_speech is not None:
        no_speech_values.append(top_no_speech)
    if top_avg_logprob is not None:
        avg_logprob_values.append(top_avg_logprob)
    if top_compression is not None:
        compression_values.append(top_compression)

    return {
        "no_speech_prob": max(no_speech_values) if no_speech_values else None,
        "avg_logprob": (
            sum(avg_logprob_values) / len(avg_logprob_values)
            if avg_logprob_values else None
        ),
        "compression_ratio": max(compression_values) if compression_values else None,
    }


def strip_whisper_prompt_leak(text):
    cleaned = (text or "").strip()
    for pattern in WHISPER_PROMPT_LEAK_PATTERNS:
        cleaned = cleaned.replace(pattern, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" 、。")
    return cleaned.strip()


def transcript_is_repetitive(text):
    compact = re.sub(r"[\s、。,.!?！？]", "", text or "")
    if len(compact) < 12:
        return False

    # Catch duplicated phrases such as によってによって or a short phrase
    # repeated three or more times by the decoder.
    for size in range(2, min(14, len(compact) // 2 + 1)):
        for start in range(0, min(size, len(compact) - size)):
            unit = compact[start:start + size]
            if unit and compact.count(unit) >= 3 and len(unit) * compact.count(unit) >= len(compact) * 0.55:
                return True
    return False


def assess_whisper_transcription(result, self_context=""):
    text = strip_whisper_prompt_leak(result.get("text", ""))
    metrics = result.get("metrics", {}) or {}

    if not text:
        return False, "", "empty transcript"

    no_speech = metrics.get("no_speech_prob")
    avg_logprob = metrics.get("avg_logprob")
    compression = metrics.get("compression_ratio")

    if no_speech is not None and no_speech > WHISPER_MAX_NO_SPEECH_PROB:
        return False, text, f"high no_speech_prob={no_speech:.2f}"
    if avg_logprob is not None and avg_logprob < WHISPER_MIN_AVG_LOGPROB:
        return False, text, f"low avg_logprob={avg_logprob:.2f}"
    if compression is not None and compression > WHISPER_MAX_COMPRESSION_RATIO:
        return False, text, f"high compression_ratio={compression:.2f}"
    if transcript_is_repetitive(text):
        return False, text, "repetitive transcript"

    context_lower = (self_context or "").lower()
    for phrase in WHISPER_COMMON_HALLUCINATIONS:
        if phrase in text and phrase.lower() not in context_lower:
            return False, text, f"common hallucination phrase: {phrase}"

    if "チャンネル" in text and "登録" in text and "チャンネル" not in (self_context or ""):
        return False, text, "likely channel-subscription hallucination"

    if not looks_like_valid_japanese_for_display(text):
        return False, text, "not enough valid Japanese"

    return True, text, ""


def transcribe_japanese_chunk(
    client,
    pcm_bytes,
    model_name,
    whisper_prompt,
):
    wav_bytes = pcm16_to_wav_bytes(pcm_bytes)
    last_error = None

    # verbose_json exposes confidence-like segment metadata. Older SDK/model
    # combinations may reject it, so fall back to normal JSON automatically.
    for response_format in ("verbose_json", "json"):
        try:
            transcription = client.audio.transcriptions.create(
                file=("live_caption.wav", wav_bytes, "audio/wav"),
                model=model_name,
                language="ja",
                prompt=whisper_prompt or None,
                response_format=response_format,
                temperature=0.0,
            )
            text_value = str(getattr(transcription, "text", "") or "").strip()
            if not text_value:
                data = _groq_object_to_dict(transcription)
                text_value = str(data.get("text", "") or "").strip()
            return {
                "text": text_value,
                "metrics": summarize_whisper_metrics(transcription),
                "response_format": response_format,
            }
        except Exception as exc:
            last_error = exc
            if response_format == "json":
                raise

    raise RuntimeError(str(last_error or "Whisper transcription failed"))

def run_second_pass_for_phrase(
    api_key,
    raw_japanese,
    previous_corrected_japanese,
    glossary_entries,
    domain_mode,
    self_context,
    source_version,
):
    """
    Run the existing evidence-based corrector synchronously for one immutable
    phrase block. Normally this is the only text-model call after Whisper.
    """
    candidate_terms = select_second_pass_glossary_candidates(
        glossary_entries,
        domain_mode,
        raw_japanese,
        max_terms=MAX_GROQ_GLOSSARY_TERMS,
    )
    local_queue = queue.Queue()

    llm_hint_worker(
        result_queue=local_queue,
        api_key=api_key,
        model_name=GROQ_HELPER_FAST,
        current_japanese=raw_japanese,
        current_translation="",
        previous_context=make_context_chunk(
            previous_corrected_japanese,
            "",
        ),
        key_terms=candidate_terms,
        full_glossary_entries=glossary_entries,
        domain_mode=domain_mode,
        class_context=make_compact_domain_context(domain_mode, ""),
        self_context=self_context,
        source_version=source_version,
    )

    try:
        result = local_queue.get(timeout=45)
    except queue.Empty:
        return {
            "type": "llm_error",
            "message": "Second-pass correction timed out.",
        }

    return result


def join_japanese_phrase_blocks(phrases, max_chars=MAX_ORIGINAL_CHARS):
    """Join finalized Japanese blocks without rewriting older blocks."""
    combined = ""
    for phrase in phrases or []:
        part = str(phrase.get("original", "") or "").strip()
        if not part:
            continue
        combined = merge_overlapping_japanese(
            combined,
            part,
            max_chars=max_chars,
        )
    return trim_caption_soft(combined, max_chars)


def join_english_phrase_blocks(phrases, max_chars=MAX_TRANSLATION_CHARS):
    """Join finalized English blocks while preserving phrase order."""
    parts = []
    for phrase in phrases or []:
        part = clean_plain_translation(
            str(phrase.get("translation", "") or "")
        )
        if not part:
            continue
        if parts and part == parts[-1]:
            continue
        parts.append(part)
    return trim_caption_soft(" ".join(parts), max_chars)


def collect_phrase_key_terms(phrases, max_terms=SECOND_PASS_MAX_CONFIRMED_TERMS):
    """Collect unique confirmed terms from the currently displayed paragraph."""
    output = []
    seen = set()

    for phrase in phrases or []:
        for item in phrase.get("terms", []) or []:
            term = str(item.get("term", item.get("jp", ""))).strip()
            meaning = str(item.get("meaning", item.get("en", ""))).strip()
            if not term or not meaning:
                continue

            key = (term.lower(), meaning.lower())
            if key in seen:
                continue

            output.append(dict(item))
            seen.add(key)

    return output[:max_terms]


def groq_whisper_translate_worker(
    audio_queue,
    result_queue,
    stop_event,
    control_queue,
    api_key,
    glossary_entries,
    domain_mode="auto",
    self_context="",
    stt_model=GROQ_STT_MODEL_ACCURATE,
    chunk_seconds=GROQ_DEFAULT_CHUNK_SECONDS,
    caption_reset_seconds=DEFAULT_RESET_SECONDS,
    drain_backlog=True,
):
    """
    Finalize every audio chunk independently.

    Pipeline per block:
        audio -> Whisper raw Japanese -> evidence-based second pass
        -> immutable corrected Japanese/English/confirmed terms

    Older phrase blocks are never sent back for rewriting. The previous phrase
    is supplied only as read-only context.
    """
    if drain_backlog:
        try:
            while True:
                audio_queue.get_nowait()
        except queue.Empty:
            pass

    client = Groq(api_key=api_key)
    sample_rate = AUDIO_INPUT_SAMPLE_RATE
    bytes_per_second = sample_rate * PCM_BYTES_PER_SAMPLE
    max_chunk_bytes = max(
        int(GROQ_MIN_AUDIO_SECONDS * bytes_per_second),
        int(float(chunk_seconds) * bytes_per_second),
    )
    min_audio_bytes = int(GROQ_MIN_AUDIO_SECONDS * bytes_per_second)

    audio_buffer = bytearray()
    previous_corrected_japanese = ""
    last_voice_time = 0.0
    last_request_time = 0.0
    last_finalized_time = 0.0
    speech_seen_in_buffer = False
    paragraph_boundary_sent = False
    consecutive_errors = 0
    phrase_counter = 0
    reset_seconds_local = float(caption_reset_seconds)

    result_queue.put({
        "type": "debug",
        "message": (
            f"Paragraph-safe Groq Whisper worker started with {stt_model}; "
            "Whisper vocabulary prompt disabled."
        ),
    })

    try:
        while not stop_event.is_set():
            while control_queue is not None and not control_queue.empty():
                try:
                    command = control_queue.get_nowait()
                except queue.Empty:
                    break

                if command == "clear":
                    audio_buffer = bytearray()
                    previous_corrected_japanese = ""
                    last_voice_time = 0.0
                    last_finalized_time = 0.0
                    speech_seen_in_buffer = False
                    paragraph_boundary_sent = False
                    phrase_counter = 0
                    result_queue.put({"type": "cleared"})

                elif isinstance(command, dict) and command.get("type") == "set_reset_seconds":
                    try:
                        reset_seconds_local = float(
                            command.get("value", reset_seconds_local)
                        )
                    except Exception:
                        pass

                # set_base_caption belonged to the old accumulated-caption
                # architecture. It is intentionally ignored now.

            now = time.time()
            pcm_chunk = b""

            try:
                pcm_chunk = audio_queue.get(timeout=0.05)
            except queue.Empty:
                pass

            if pcm_chunk:
                audio_buffer.extend(pcm_chunk)
                level = pcm16_rms(pcm_chunk)
                if level >= GROQ_VAD_RMS_THRESHOLD:
                    last_voice_time = now
                    speech_seen_in_buffer = True
                    paragraph_boundary_sent = False

                # Catch up quickly after a model call while preserving order.
                while len(audio_buffer) < max_chunk_bytes:
                    try:
                        extra = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    audio_buffer.extend(extra)
                    if pcm16_rms(extra) >= GROQ_VAD_RMS_THRESHOLD:
                        last_voice_time = time.time()
                        speech_seen_in_buffer = True
                        paragraph_boundary_sent = False

            silence_seconds = (
                now - last_voice_time
                if last_voice_time > 0
                else 999999.0
            )
            request_interval_ok = (
                last_request_time <= 0
                or now - last_request_time >= GROQ_MIN_REQUEST_INTERVAL_SECONDS
            )

            full_chunk_ready = (
                speech_seen_in_buffer
                and len(audio_buffer) >= max_chunk_bytes
                and request_interval_ok
            )
            endpoint_ready = (
                speech_seen_in_buffer
                and len(audio_buffer) >= min_audio_bytes
                and silence_seconds >= GROQ_ENDPOINT_SILENCE_SECONDS
                and request_interval_ok
            )

            if full_chunk_ready or endpoint_ready:
                if endpoint_ready:
                    request_pcm = bytes(audio_buffer)
                    audio_buffer = bytearray()
                    speech_seen_in_buffer = False
                else:
                    request_pcm = bytes(audio_buffer[:max_chunk_bytes])
                    audio_buffer = audio_buffer[max_chunk_bytes:]
                    speech_seen_in_buffer = bool(audio_buffer)

                if pcm16_rms(request_pcm) < GROQ_VAD_RMS_THRESHOLD * 0.55:
                    continue

                last_request_time = time.time()
                phrase_counter += 1

                try:
                    whisper_result = transcribe_japanese_chunk(
                        client,
                        request_pcm,
                        stt_model,
                        whisper_prompt="",
                    )
                    accepted, raw_japanese, reject_reason = assess_whisper_transcription(
                        whisper_result,
                        self_context=self_context,
                    )
                    metrics = whisper_result.get("metrics", {}) or {}

                    if not accepted:
                        result_queue.put({
                            "type": "debug",
                            "message": (
                                f"Rejected Whisper block {phrase_counter} "
                                f"({reject_reason}): {raw_japanese[:120]} | "
                                f"metrics={metrics}"
                            ),
                        })
                        continue

                    second_pass = run_second_pass_for_phrase(
                        api_key=api_key,
                        raw_japanese=raw_japanese,
                        previous_corrected_japanese=previous_corrected_japanese,
                        glossary_entries=glossary_entries,
                        domain_mode=domain_mode,
                        self_context=self_context,
                        source_version=phrase_counter,
                    )

                    if second_pass.get("type") == "llm_hint":
                        corrected_japanese = str(
                            second_pass.get(
                                "corrected_japanese_original",
                                raw_japanese,
                            )
                            or raw_japanese
                        ).strip()
                        translation = str(
                            second_pass.get(
                                "corrected_english_caption",
                                "",
                            )
                            or ""
                        ).strip()
                        confirmed_terms = second_pass.get(
                            "confirmed_terms",
                            second_pass.get("key_terms", []),
                        ) or []
                        corrections = second_pass.get("corrections", []) or []
                        is_unclear = bool(second_pass.get("is_unclear", False))
                        unclear_reason = str(
                            second_pass.get("unclear_reason", "") or ""
                        ).strip()
                    else:
                        corrected_japanese = apply_glossary_source_corrections(
                            raw_japanese,
                            glossary_entries,
                            domain_mode,
                        )
                        translation = ""
                        confirmed_terms = build_exact_confirmed_terms(
                            corrected_japanese=corrected_japanese,
                            raw_japanese=raw_japanese,
                            glossary_entries=glossary_entries,
                            domain_mode=domain_mode,
                            ai_terms=[],
                        )
                        corrections = []
                        is_unclear = True
                        unclear_reason = str(
                            second_pass.get("message", "Second pass unavailable.")
                        )

                    second_pass_translation = translation
                    try:
                        # Translate the grounded Japanese after correction.
                        # This prevents fluent English from being based on a
                        # model-rewritten or contaminated Japanese sentence.
                        translation = translate_japanese_with_groq(
                            client=client,
                            japanese_text=corrected_japanese,
                            previous_japanese=previous_corrected_japanese,
                            glossary_entries=glossary_entries,
                            domain_mode=domain_mode,
                            self_context=self_context,
                        )
                    except Exception:
                        translation = second_pass_translation

                    if not looks_like_valid_japanese_for_display(
                        corrected_japanese
                    ):
                        result_queue.put({
                            "type": "debug",
                            "message": (
                                "Ignored finalized low-quality Japanese block: "
                                f"{corrected_japanese[:120]}"
                            ),
                        })
                        continue

                    previous_corrected_japanese = corrected_japanese
                    last_finalized_time = time.time()
                    paragraph_boundary_sent = False
                    consecutive_errors = 0

                    result_queue.put({
                        "type": "final_phrase",
                        "phrase_id": phrase_counter,
                        "raw_original": raw_japanese,
                        "original": corrected_japanese,
                        "translation": clean_plain_translation(translation),
                        "terms": confirmed_terms,
                        "corrections": corrections,
                        "is_unclear": is_unclear,
                        "unclear_reason": unclear_reason,
                        "endpoint": bool(endpoint_ready),
                        "metrics": metrics,
                    })

                except Exception as exc:
                    consecutive_errors += 1
                    message = shorten_error_for_ui(str(exc), 240)
                    result_queue.put({
                        "type": "debug",
                        "message": (
                            f"Finalized phrase request failed "
                            f"({consecutive_errors}): {message}"
                        ),
                    })
                    if consecutive_errors >= 3:
                        result_queue.put({
                            "type": "error",
                            "message": (
                                "Groq speech/correction error: "
                                f"{message}"
                            ),
                        })
                        consecutive_errors = 0

            # A long pause closes the paragraph, but the finished paragraph
            # stays visible. The next final phrase starts a fresh paragraph.
            if (
                last_finalized_time > 0
                and not paragraph_boundary_sent
                and not speech_seen_in_buffer
                and len(audio_buffer) < min_audio_bytes
                and silence_seconds >= reset_seconds_local
            ):
                result_queue.put({"type": "paragraph_boundary"})
                paragraph_boundary_sent = True
                previous_corrected_japanese = ""

            if (
                not speech_seen_in_buffer
                and len(audio_buffer) > max_chunk_bytes
            ):
                audio_buffer = bytearray()

    except Exception as exc:
        if not stop_event.is_set():
            result_queue.put({
                "type": "error",
                "message": f"Groq live worker error: {exc}",
            })
    finally:
        result_queue.put({"type": "stopped"})


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Technical Interpreter Captioner",
    layout="wide",
)

st.title("Technical Interpreter Captioner")

st.caption(
    "Japanese → English phrase captions using Groq Whisper STT, "
    "prompt-free Whisper, evidence-based per-block second-pass AI, and immutable paragraph assembly."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Settings")

    translation_engine = ENGINE_GROQ_WHISPER
    st.info("Translation engine: Groq Whisper + Groq translation")
    st.caption("Captions update by phrase, not character-by-character.")

    domain_mode = st.selectbox(
        "Technical domain",
        ["auto", "automotive", "cad", "product design", "school/event"],
        index=0,
        help="Choose the real class topic. Use 'school/event' only for Summer Course/BINUS topics.",
    )
    st.caption(f"Selected technical context: {domain_mode}")

    st.caption("Source mode: Japanese is forced in Whisper for better accuracy and latency.")

    stt_quality = st.selectbox(
        "Whisper recognition model",
        [
            "High accuracy (whisper-large-v3)",
            "Faster (whisper-large-v3-turbo)",
        ],
        index=0,
        help=(
            "Use High accuracy for technical Japanese. Turbo is cheaper/faster "
            "but is slightly less accurate."
        ),
    )
    stt_model_name = (
        GROQ_STT_MODEL_FAST
        if "turbo" in stt_quality.lower()
        else GROQ_STT_MODEL_ACCURATE
    )

    stt_chunk_seconds = st.slider(
        "Caption audio chunk",
        min_value=5.0,
        max_value=10.0,
        value=GROQ_DEFAULT_CHUNK_SECONDS,
        step=0.2,
        help=(
            "For continuous technical Japanese, 6–8 seconds gives Whisper more context. "
            "The new pipeline finalizes each chunk independently and uses no vocabulary prompt."
        ),
    )

    microphone_profile = st.selectbox(
        "Microphone processing",
        [
            "Technical speech (recommended)",
            "Raw microphone",
            "Speakerphone / strong echo control",
        ],
        index=0,
        help=(
            "Technical speech keeps noise suppression but disables browser "
            "echo cancellation and automatic gain, which can clip short "
            "Japanese syllables. Use Raw with a close external microphone. "
            "Use Speakerphone only when loudspeaker echo is a real problem."
        ),
    )

    if microphone_profile == "Raw microphone":
        microphone_audio_constraints = {
            "channelCount": 1,
            "echoCancellation": False,
            "noiseSuppression": False,
            "autoGainControl": False,
        }
    elif microphone_profile == "Speakerphone / strong echo control":
        microphone_audio_constraints = {
            "channelCount": 1,
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        }
    else:
        microphone_audio_constraints = {
            "channelCount": 1,
            "echoCancellation": False,
            "noiseSuppression": True,
            "autoGainControl": False,
        }

    main_display_mode = st.radio(
        "Main display",
        ["Terms + meaning", "Captions + terms"],
        index=1,
        help="Terms + meaning = interpreter key terms only. Captions + terms = captions plus key terms.",
    )

    # Always show meanings because this app is now key-term support first.
    show_term_meaning = True

    subtitle_display = st.radio(
        "Caption history",
        ["Latest only", "History"],
        index=0,
    )

    self_context = st.text_area(
        "Self context / today's context",
        value=st.session_state.get("self_context_text", ""),
        height=80,
        placeholder="Example: Today is about automotive braking and inertia compensation.",
        help="This guides Whisper spelling and English translation. It is not shown to the audience.",
    )
    st.session_state.self_context_text = self_context

    show_error_details = st.checkbox(
        "Show AI error details",
        value=False,
        help="Keep OFF during interpreting. Details stay in debug.",
    )

    font_size = st.slider(
        "English caption font size",
        min_value=12,
        max_value=32,
        value=15,
        step=1,
    )

    jp_font_size = st.slider(
        "Japanese original font size",
        min_value=11,
        max_value=28,
        value=13,
        step=1,
    )

    reset_seconds = st.slider(
        "Prepare new caption after pause",
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

    st.write("Second-Pass Key-Term Correction")

    # Key terms are the main purpose of this interpreter, so the contextual
    # second pass is always enabled. It runs after completed phrases rather
    # than trying to rewrite every partial transcript.
    use_llm_hints = True
    st.success("Second-pass AI is always ON inside the phrase worker. Every block is finalized once and older blocks cannot be rewritten.")

    llm_model_name = GROQ_HELPER_FAST
    st.caption(f"Second-pass model: {llm_model_name}")

    llm_budget_mode = "Every Phrase"
    selected_budget = LLM_BUDGET_MODES[llm_budget_mode]
    llm_hint_interval = float(selected_budget["interval"])
    llm_min_context_chars = int(selected_budget["min_chars"])
    llm_session_limit = int(selected_budget["session_limit"])

    st.caption(
        "Each Whisper block is corrected once before it is appended. "
        "Older blocks are immutable and are never rewritten."
    )

    current_helper_calls = st.session_state.get("llm_calls_this_session", 0)
    st.caption(f"Helper calls this session: {current_helper_calls} / {llm_session_limit}")

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
        f"Loaded {len(context_terms)} glossary terms and "
        f"{len(translation_terms)} translation mappings."
    )

    active_glossary_entries = load_glossary_entries(terms_file)
    st.caption("Whisper vocabulary prompting is disabled to prevent unspoken term leakage.")


# ============================================================
# Session state
# ============================================================

defaults = {
    "app_active": False,
    "pending_start_translation": False,
    "mic_instance_id": 0,
    "mic_generation": 0,
    "live_started_for_generation": None,
    "mobile_mic_failure_message": "",
    "current_engine": "",

    "live_original": "",
    "live_translation": "",
    "finalized_phrases": [],
    "pending_new_paragraph": False,
    "last_japanese_token_time": 0.0,
    "caption_history": [],
    "live_running": False,
    "live_error": "",
    "live_result_queue": queue.Queue(),
    "live_control_queue": queue.Queue(),
    "live_stop_event": threading.Event(),
    "live_thread": None,
    "debug_messages": [],
    "mic_wait_start_time": 0.0,
    "mic_wait_notice": "",
    "last_update_time": "",
    "last_reset_seconds": DEFAULT_RESET_SECONDS,
    "pending_visual_reset": False,
    "caption_stage": "idle",
    "last_raw_input_time": "",
    "last_raw_translation_time": "",
    "last_helper_fix_time": "",
    "last_ai_check_time": "",
    "correction_status": "idle",
    "live_token_version": 0,
    "last_llm_checked_token_version": -1,
    "last_live_endpoint": False,
    "llm_calls_this_session": 0,
    "llm_budget_reached": False,

    "llm_result_queue": queue.Queue(),
    "llm_thread": None,
    "llm_running": False,
    "llm_error": "",
    "llm_corrected_japanese_original": "",
    "llm_corrected_english_caption": "",
    "llm_corrected_source_text": "",
    "llm_is_unclear": False,
    "llm_unclear_reason": "",
    "llm_key_terms": [],
    "llm_corrections": [],
    "llm_last_call_time": 0.0,
    "llm_cooldown_until": 0.0,
    "llm_last_finish_time": "",
    "llm_last_source_text": "",
    "llm_context_chunks": [],

    "self_context_text": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


if st.session_state.current_engine and st.session_state.current_engine != translation_engine:
    st.session_state.app_active = False
    st.session_state.pending_start_translation = False
    st.session_state.live_running = False
    st.session_state.live_stop_event.set()
    # Keep the WebRTC component stable. Recreating it repeatedly can leave
    # old aioice STUN retry timers behind on Streamlit Cloud.
    st.session_state.current_engine = translation_engine
    st.rerun()

if not st.session_state.current_engine:
    st.session_state.current_engine = ENGINE_GROQ_WHISPER


if float(reset_seconds) != float(st.session_state.last_reset_seconds):
    st.session_state.last_reset_seconds = float(reset_seconds)

    if st.session_state.live_running:
        st.session_state.live_control_queue.put({
            "type": "set_reset_seconds",
            "value": float(reset_seconds),
        })


# ============================================================
# API keys
# ============================================================

groq_api_key = safe_get_secret_or_env("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is not set.\n\n"
        "The main Whisper transcription and English translation both require it. "
        "For Streamlit Cloud, add this in Secrets:\n\n"
        'GROQ_API_KEY = "your_groq_api_key_here"'
    )
    st.stop()


# ============================================================
# Microphone / WebRTC
# ============================================================

rtc_configuration, turn_server_ready, turn_error, active_ice_servers = (
    build_rtc_configuration()
)


st.subheader("Microphone")

if turn_server_ready:
    st.caption("Mobile WebRTC relay: Metered TURN connected.")
else:
    st.warning(
        "Metered TURN is unavailable, so the app is using STUN-only fallback. "
        "Phones and mobile networks may not work."
    )

    if turn_error and show_debug:
        st.code(turn_error)

webrtc_ctx = webrtc_streamer(
    # Stable key prevents repeated WebRTC peer-connection destruction/recreation.
    # Use the manual Reset mic connection button below if a hard reset is needed.
    key=f"groq-whisper-caption-mic-{st.session_state.mic_generation}",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": False,
        "audio": microphone_audio_constraints,
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

reset_mic_clicked = st.button(
    "Reset Mic Connection",
    use_container_width=True,
    help="Use only if the browser mic gets stuck. This refreshes the page and rebuilds the WebRTC connection.",
)

if reset_mic_clicked:
    st.session_state.app_active = False
    st.session_state.pending_start_translation = False
    st.session_state.live_running = False
    st.session_state.mic_generation += 1
    st.session_state.mobile_mic_failure_message = (
        "Mic connection reset. Tap Start Translation to enable the microphone again."
    )
    st.session_state.live_stop_event.set()
    st.session_state.live_result_queue = queue.Queue()
    st.session_state.live_control_queue = queue.Queue()
    st.session_state.live_thread = None
    st.session_state.mic_wait_start_time = 0.0
    st.session_state.mic_wait_notice = ""
    st.session_state.live_error = ""
    st.rerun()

if toggle_clicked:
    if st.session_state.app_active:
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.live_running = False
        st.session_state.live_stop_event.set()
        st.session_state.live_result_queue = queue.Queue()
        st.session_state.live_control_queue = queue.Queue()
        st.session_state.live_thread = None
        st.session_state.mic_wait_start_time = 0.0
        st.session_state.mic_wait_notice = ""

        st.rerun()

    else:
        st.session_state.live_stop_event = threading.Event()
        st.session_state.live_result_queue = queue.Queue()
        st.session_state.live_control_queue = queue.Queue()
        st.session_state.live_thread = None
        st.session_state.app_active = True
        st.session_state.pending_start_translation = True
        st.session_state.live_error = ""
        st.session_state.mobile_mic_failure_message = ""
        st.session_state.mic_wait_start_time = time.time()
        st.session_state.mic_wait_notice = "Waiting for browser microphone audio..."

        st.rerun()

if clear_clicked:
    st.session_state.mic_wait_start_time = 0.0
    st.session_state.mic_wait_notice = ""
    st.session_state.live_original = ""
    st.session_state.live_translation = ""
    st.session_state.finalized_phrases = []
    st.session_state.pending_new_paragraph = False
    st.session_state.last_japanese_token_time = 0.0
    st.session_state.caption_history = []
    st.session_state.last_update_time = ""
    st.session_state.live_error = ""
    st.session_state.pending_visual_reset = False
    st.session_state.caption_stage = "idle"
    st.session_state.last_raw_input_time = ""
    st.session_state.last_raw_translation_time = ""
    st.session_state.last_helper_fix_time = ""
    st.session_state.last_ai_check_time = ""
    st.session_state.correction_status = "idle"
    st.session_state.live_token_version = 0
    st.session_state.last_llm_checked_token_version = -1
    st.session_state.last_live_endpoint = False
    st.session_state.llm_calls_this_session = 0
    st.session_state.llm_budget_reached = False

    st.session_state.llm_context_chunks = []
    st.session_state.llm_corrected_japanese_original = ""
    st.session_state.llm_corrected_english_caption = ""
    st.session_state.llm_corrected_source_text = ""
    st.session_state.llm_is_unclear = False
    st.session_state.llm_unclear_reason = ""
    st.session_state.llm_key_terms = []
    st.session_state.llm_corrections = []
    st.session_state.llm_error = ""
    st.session_state.llm_last_source_text = ""
    st.session_state.llm_cooldown_until = 0.0
    st.session_state.llm_last_finish_time = ""

    if st.session_state.live_running:
        st.session_state.live_control_queue.put("clear")
        st.session_state.debug_messages.append("Clear requested.")
        st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]


# ============================================================
# Translator Ask AI removed
# ============================================================

# The app is now key-terms-first.
# If the speaker/interpreter says one short Japanese word and pauses,
# the normal live key-term pipeline handles it automatically.


# ============================================================
# Auto-start Groq Whisper captions as soon as the mic connection exists
# ============================================================

if (
    st.session_state.pending_start_translation
    and st.session_state.app_active
    and not st.session_state.live_running
    and webrtc_ctx.audio_processor
):
    # Start the Groq Whisper worker immediately, in parallel with the browser
    # mic connection still coming up, instead of waiting to confirm audio
    # first. This is what makes the status go green (and speech start being
    # sent) right away instead of serializing "wait for mic" then "wait for
    # Groq worker". Anything said before real audio frames arrive buffers in
    # the queue and gets sent once it's ready, so nothing said during
    # startup is lost. mic_wait_start_time is kept as
    # the anchor for the watchdog below, which still catches a mic that
    # never actually starts (e.g. the iOS Safari refresh bug).
    processor = webrtc_ctx.audio_processor

    st.session_state.live_stop_event = threading.Event()
    st.session_state.live_result_queue = queue.Queue()
    st.session_state.live_control_queue = queue.Queue()
    st.session_state.live_error = ""
    st.session_state.debug_messages = []
    st.session_state.live_original = ""
    st.session_state.live_translation = ""
    st.session_state.finalized_phrases = []
    st.session_state.pending_new_paragraph = False
    st.session_state.last_japanese_token_time = 0.0
    st.session_state.caption_history = []
    st.session_state.last_update_time = ""
    st.session_state.pending_visual_reset = False
    st.session_state.caption_stage = "idle"
    st.session_state.last_raw_input_time = ""
    st.session_state.last_raw_translation_time = ""
    st.session_state.last_helper_fix_time = ""
    st.session_state.last_ai_check_time = ""
    st.session_state.correction_status = "idle"
    st.session_state.live_token_version = 0
    st.session_state.last_llm_checked_token_version = -1
    st.session_state.last_live_endpoint = False
    st.session_state.llm_calls_this_session = 0
    st.session_state.llm_budget_reached = False

    st.session_state.llm_context_chunks = []
    st.session_state.llm_corrected_japanese_original = ""
    st.session_state.llm_corrected_english_caption = ""
    st.session_state.llm_corrected_source_text = ""
    st.session_state.llm_is_unclear = False
    st.session_state.llm_unclear_reason = ""
    st.session_state.llm_key_terms = []
    st.session_state.llm_corrections = []
    st.session_state.llm_error = ""
    st.session_state.llm_last_source_text = ""
    st.session_state.llm_last_call_time = 0.0
    st.session_state.llm_cooldown_until = 0.0
    st.session_state.llm_last_finish_time = ""

    st.session_state.live_running = True
    st.session_state.pending_start_translation = False
    st.session_state.mic_wait_notice = ""
    # Anchors the watchdog below: how long has it been since we started,
    # with no confirmed audio yet.
    st.session_state.mic_wait_start_time = time.time()

    # Only drain queued audio on a reconnect within the same mic
    # connection (Stop then Start again), where audio kept accumulating
    # while stopped. A genuinely fresh mic connection has nothing stale
    # to drop, and draining it would discard whatever was already said
    # while the connection was still coming up.
    drain_backlog = (
        st.session_state.get("live_started_for_generation")
        == st.session_state.mic_generation
    )

    worker_target = groq_whisper_translate_worker
    worker_args = (
        processor.audio_queue,
        st.session_state.live_result_queue,
        st.session_state.live_stop_event,
        st.session_state.live_control_queue,
        groq_api_key,
        active_glossary_entries,
        domain_mode,
        self_context,
        stt_model_name,
        float(stt_chunk_seconds),
        float(reset_seconds),
        drain_backlog,
    )

    st.session_state.live_thread = threading.Thread(
        target=worker_target,
        args=worker_args,
        daemon=True,
    )

    st.session_state.live_thread.start()
    st.session_state.live_started_for_generation = st.session_state.mic_generation

# ============================================================
# Watchdog: recover if the mic connection never actually sends audio
# ============================================================

if (
    st.session_state.live_running
    and webrtc_ctx.audio_processor
    and st.session_state.get("mic_wait_start_time", 0.0)
):
    processor_probe = webrtc_ctx.audio_processor
    mic_frame_count = int(getattr(processor_probe, "frame_count", 0) or 0)

    if mic_frame_count > 0:
        # Confirmed working; stop watching.
        st.session_state.mic_wait_start_time = 0.0
    else:
        waited = time.time() - float(st.session_state.mic_wait_start_time)

        if waited > MOBILE_MIC_START_TIMEOUT_SECONDS:
            # iOS Safari does not automatically restart microphone capture
            # after refresh, and some other environments can silently never
            # deliver audio either. Return to stopped state so the next
            # Start button press is a real user gesture and rebuilds the
            # WebRTC component.
            st.session_state.app_active = False
            st.session_state.pending_start_translation = False
            st.session_state.live_running = False
            st.session_state.live_stop_event.set()
            st.session_state.mic_generation += 1
            st.session_state.mic_wait_start_time = 0.0
            st.session_state.mic_wait_notice = ""
            st.session_state.mobile_mic_failure_message = (
                "Microphone did not start. Check that mic access is allowed "
                "for this site in your browser, then press Start Translation again."
            )
            st.rerun()

# ============================================================
# Pull Groq Whisper results into UI state
# ============================================================

while not st.session_state.live_result_queue.empty():
    item = st.session_state.live_result_queue.get()
    item_type = item.get("type")

    if item_type == "final_phrase":
        phrase = {
            "phrase_id": int(item.get("phrase_id", 0) or 0),
            "raw_original": str(item.get("raw_original", "") or "").strip(),
            "original": str(item.get("original", "") or "").strip(),
            "translation": str(item.get("translation", "") or "").strip(),
            "terms": item.get("terms", []) or [],
            "corrections": item.get("corrections", []) or [],
            "is_unclear": bool(item.get("is_unclear", False)),
            "unclear_reason": str(item.get("unclear_reason", "") or "").strip(),
        }

        if st.session_state.pending_new_paragraph:
            st.session_state.finalized_phrases = []
            st.session_state.caption_history = []
            st.session_state.pending_new_paragraph = False

        # Ignore accidental duplicate result delivery.
        existing_ids = {
            int(existing.get("phrase_id", -1) or -1)
            for existing in st.session_state.finalized_phrases
        }
        if phrase["phrase_id"] not in existing_ids:
            st.session_state.finalized_phrases.append(phrase)
            st.session_state.finalized_phrases = (
                st.session_state.finalized_phrases[
                    -MAX_FINALIZED_PHRASE_BLOCKS:
                ]
            )

        st.session_state.live_original = join_japanese_phrase_blocks(
            st.session_state.finalized_phrases,
            max_chars=MAX_ORIGINAL_CHARS,
        )
        st.session_state.live_translation = join_english_phrase_blocks(
            st.session_state.finalized_phrases,
            max_chars=MAX_TRANSLATION_CHARS,
        )
        st.session_state.llm_key_terms = collect_phrase_key_terms(
            st.session_state.finalized_phrases,
            max_terms=SECOND_PASS_MAX_CONFIRMED_TERMS,
        )
        st.session_state.llm_corrections = []
        st.session_state.llm_corrected_japanese_original = (
            st.session_state.live_original
        )
        st.session_state.llm_corrected_english_caption = (
            st.session_state.live_translation
        )
        st.session_state.llm_corrected_source_text = ""
        st.session_state.llm_is_unclear = phrase["is_unclear"]
        st.session_state.llm_unclear_reason = phrase["unclear_reason"]
        st.session_state.caption_stage = "ai_corrected"
        st.session_state.correction_status = (
            "unclear_applied" if phrase["is_unclear"] else "applied"
        )
        st.session_state.last_helper_fix_time = time.strftime("%H:%M:%S")
        st.session_state.last_update_time = time.strftime("%H:%M:%S")
        st.session_state.last_japanese_token_time = time.time()
        st.session_state.live_token_version += 1
        st.session_state.last_live_endpoint = bool(
            item.get("endpoint", False)
        )

        if phrase["translation"]:
            st.session_state.caption_history.append(
                phrase["translation"]
            )
            st.session_state.caption_history = (
                st.session_state.caption_history[-MAX_HISTORY_ITEMS:]
            )

    elif item_type == "paragraph_boundary":
        # Keep the completed paragraph visible. The first finalized block after
        # this boundary replaces it with a fresh paragraph.
        st.session_state.pending_new_paragraph = True
        st.session_state.last_live_endpoint = True
        st.session_state.correction_status = "paragraph_complete"

    elif item_type == "tokens":
        original = item.get("original", "")
        translation = item.get("translation", "")
        st.session_state.last_live_endpoint = bool(item.get("endpoint", False))
        original, translation = light_domain_context_cleanup(
            original,
            translation,
            domain_mode,
        )

        # Filler guard:
        # Do not display filler-only Japanese like えー、あの、うん.
        original = strip_leading_japanese_fillers(original)

        if should_skip_as_filler(original, translation):
            st.session_state.debug_messages.append(
                f"Ignored filler-only speech: {original[:80] or translation[:80]}"
            )
            st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
            continue

        # Japanese-only guard:
        # Do not display or translate Indonesian/English/other-language speech.
        if JAPANESE_ONLY_MODE:
            if original and not looks_like_valid_japanese_for_display(original):
                st.session_state.debug_messages.append(
                    f"Ignored non-Japanese or low-quality live text: {original[:80]}"
                )
                st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
                continue

            if original:
                st.session_state.last_japanese_token_time = time.time()

            if translation and not original:
                # A translation-only update is only trusted if real Japanese
                # arrived recently. "Some Japanese existed earlier in this
                # segment" is not proof this translation belongs to it —
                # background speech in another language can otherwise ride
                # in on an old, already-confirmed Japanese source.
                last_japanese_time = float(
                    st.session_state.get("last_japanese_token_time", 0.0) or 0.0
                )
                japanese_recent_enough = (
                    last_japanese_time > 0
                    and time.time() - last_japanese_time <= TRANSLATION_WITHOUT_SOURCE_MAX_LAG_SECONDS
                )

                if not japanese_recent_enough:
                    st.session_state.debug_messages.append(
                        f"Ignored translation without recent Japanese source: {translation[:80]}"
                    )
                    st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
                    continue

        if st.session_state.pending_visual_reset and (original or translation):
            st.session_state.live_original = ""
            st.session_state.live_translation = ""
            st.session_state.last_japanese_token_time = 0.0
            st.session_state.caption_history = []
            st.session_state.llm_corrected_japanese_original = ""
            st.session_state.llm_corrected_english_caption = ""
            st.session_state.llm_corrected_source_text = ""
            st.session_state.llm_is_unclear = False
            st.session_state.llm_unclear_reason = ""
            # Key terms belong to one corrected phrase only. Clear them before
            # the new phrase so stale terms never leak into the next caption.
            st.session_state.llm_key_terms = []
            st.session_state.llm_corrections = []
            st.session_state.caption_stage = "raw_started"
            st.session_state.last_helper_fix_time = ""
            st.session_state.last_ai_check_time = ""
            st.session_state.correction_status = "pending"
            st.session_state.pending_visual_reset = False

        if original or translation:
            st.session_state.live_token_version += 1
            prepare_next_ai_check_after_new_live_text()
            # New live speech means the previous unclear warning/key terms may be stale.
            st.session_state.llm_is_unclear = False
            st.session_state.llm_unclear_reason = ""

            # First layer: check our glossary immediately.
            # This makes terms like 三角形 = triangle appear without waiting for Groq.
            # Runs after the reset-clear above, so a fresh segment's own terms
            # do not get detected here only to be wiped by that clear.
            instant_detected_terms = extract_key_terms_for_llm(
                original or st.session_state.live_original,
                translation or st.session_state.live_translation,
                terms_file,
            )
            instant_detected_terms = filter_detected_terms_for_current_caption(
                instant_detected_terms,
                original or st.session_state.live_original,
                translation or st.session_state.live_translation,
                domain_mode,
            )
            instant_key_terms = detected_terms_to_llm_key_terms(instant_detected_terms)

            # Before the second pass finishes, show only exact/raw-supported
            # canonical terms from the current phrase. The final second-pass
            # result replaces this list with evidence-validated terms.
            st.session_state.llm_key_terms = merge_key_terms_preserve_order(
                st.session_state.llm_key_terms,
                instant_key_terms,
                max_terms=SECOND_PASS_MAX_CONFIRMED_TERMS,
            )

        if original:
            # If the Japanese source changed a lot but no matching new English
            # translation has arrived yet, remove the old English caption so the
            # two boxes do not describe different speech.
            previous_original = st.session_state.live_original or ""
            if (
                previous_original
                and original != previous_original
                and previous_original not in original
                and original not in previous_original
                and not translation
            ):
                st.session_state.live_translation = ""
                st.session_state.llm_corrected_english_caption = ""
                st.session_state.llm_corrected_source_text = ""
                st.session_state.llm_corrections = []
                st.session_state.correction_status = "waiting_for_english"

            st.session_state.live_original = original
            st.session_state.caption_stage = "raw_japanese"
            st.session_state.last_raw_input_time = time.strftime("%H:%M:%S")

            if not st.session_state.live_translation:
                st.session_state.correction_status = "waiting_for_english"

        if translation:
            # Use current live translation directly.
            # Keeping an older, longer translation made English and Japanese
            # captions become different conversations.
            st.session_state.live_translation = translation
            st.session_state.caption_stage = "raw_english"
            st.session_state.last_raw_translation_time = time.strftime("%H:%M:%S")

            if use_llm_hints and groq_api_key:
                if not st.session_state.llm_running and not st.session_state.llm_corrected_source_text:
                    st.session_state.correction_status = "pending"
            else:
                st.session_state.correction_status = "off"

            if (
                not st.session_state.caption_history
                or st.session_state.caption_history[-1] != translation
            ):
                st.session_state.caption_history.append(translation)
                st.session_state.caption_history = (
                    st.session_state.caption_history[-MAX_HISTORY_ITEMS:]
                )

        st.session_state.last_update_time = time.strftime("%H:%M:%S")

    elif item_type == "page_reset_legacy":
        st.session_state.last_live_endpoint = True
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

        # Do not clear the visible caption immediately after pause.
        # Keep it on screen so the reader has time to read it.
        # The next incoming token will clear/replace the old caption.
        st.session_state.pending_visual_reset = True
        if st.session_state.live_translation and use_llm_hints and groq_api_key:
            st.session_state.correction_status = "pending"

    elif item_type == "cleared":
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.finalized_phrases = []
        st.session_state.pending_new_paragraph = False
        st.session_state.last_japanese_token_time = 0.0
        st.session_state.caption_history = []
        st.session_state.last_update_time = ""
        st.session_state.llm_context_chunks = []
        st.session_state.llm_corrected_japanese_original = ""
        st.session_state.llm_corrected_english_caption = ""
        st.session_state.llm_corrected_source_text = ""
        st.session_state.llm_key_terms = []
        st.session_state.llm_corrections = []
        st.session_state.llm_error = ""
        st.session_state.llm_last_source_text = ""
        st.session_state.llm_running = False
        st.session_state.pending_visual_reset = False
        st.session_state.caption_stage = "idle"
        st.session_state.last_raw_input_time = ""
        st.session_state.last_raw_translation_time = ""
        st.session_state.last_helper_fix_time = ""
        st.session_state.last_ai_check_time = ""
        st.session_state.correction_status = "idle"
        st.session_state.live_token_version = 0
        st.session_state.last_llm_checked_token_version = -1
        st.session_state.llm_calls_this_session = 0
        st.session_state.llm_budget_reached = False

    elif item_type == "debug":
        message = item.get("message", "")
        if message:
            st.session_state.debug_messages.append(message)
            st.session_state.debug_messages = (
                st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
            )

    elif item_type == "error":
        error_message = item.get("message", "")
        if is_connection_noise_error(error_message):
            st.session_state.debug_messages.append(error_message)
            st.session_state.debug_messages = (
                st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
            )
        else:
            st.session_state.live_error = error_message
        st.session_state.live_running = False
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.mic_wait_start_time = 0.0
        st.session_state.mic_wait_notice = ""

    elif item_type == "stopped":
        st.session_state.live_running = False


# ============================================================
# Pull LLM results
# ============================================================

while not st.session_state.llm_result_queue.empty():
    item = st.session_state.llm_result_queue.get()
    item_type = item.get("type")

    if item_type == "llm_hint":
        result_source_version = int(item.get("source_version", -1) or -1)
        result_source_japanese = str(item.get("source_japanese", "") or "").strip()
        current_version = int(st.session_state.live_token_version)

        # A completed phrase may still be visible after page_reset, so equality
        # is preferred. If newer speech has already started, never let the old
        # correction or its key terms overwrite the new phrase.
        stale_result = (
            result_source_version >= 0
            and result_source_version != current_version
            and not source_text_matches_for_correction(
                st.session_state.live_original,
                result_source_japanese,
            )
        )

        st.session_state.llm_running = False
        st.session_state.llm_last_finish_time = time.strftime("%H:%M:%S")
        st.session_state.llm_cooldown_until = (
            time.time() + float(llm_hint_interval)
        )
        st.session_state.llm_calls_this_session += 1

        if stale_result:
            st.session_state.debug_messages.append(
                "Ignored stale second-pass result for an older phrase."
            )
            st.session_state.debug_messages = (
                st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
            )
            st.session_state.correction_status = "stale_ignored"
            continue

        st.session_state.llm_corrected_japanese_original = item.get(
            "corrected_japanese_original",
            "",
        )
        st.session_state.llm_corrected_english_caption = item.get(
            "corrected_english_caption",
            "",
        )
        st.session_state.llm_corrected_source_text = item.get(
            "source_text",
            "",
        )
        st.session_state.llm_is_unclear = bool(
            item.get("is_unclear", False)
        )
        st.session_state.llm_unclear_reason = item.get(
            "unclear_reason",
            "",
        )
        st.session_state.llm_corrections = item.get("corrections", [])
        st.session_state.llm_error = ""

        corrected_base_original, corrected_base_translation = (
            merge_ai_result_into_live_caption(
                live_original=st.session_state.live_original,
                live_translation=st.session_state.live_translation,
                ai_source_text=st.session_state.llm_corrected_source_text,
                ai_corrected_original=(
                    st.session_state.llm_corrected_japanese_original
                ),
                ai_corrected_translation=(
                    st.session_state.llm_corrected_english_caption
                ),
                corrections=st.session_state.llm_corrections,
                domain_mode=domain_mode,
            )
        )

        if (
            corrected_base_original.strip()
            or not corrected_base_translation.strip()
        ):
            st.session_state.live_original = corrected_base_original
            st.session_state.live_translation = corrected_base_translation

        # Replace, rather than merge, key terms. Every term was validated by
        # the worker against this phrase's raw evidence and corrected Japanese.
        confirmed_terms = item.get(
            "confirmed_terms",
            item.get("key_terms", []),
        )
        st.session_state.llm_key_terms = [
            term_item
            for term_item in confirmed_terms
            if key_term_display_allowed(
                term_item,
                st.session_state.live_original,
            )
        ][:SECOND_PASS_MAX_CONFIRMED_TERMS]

        has_ai_update = (
            bool(st.session_state.llm_corrected_japanese_original)
            or bool(st.session_state.llm_corrected_english_caption)
            or bool(st.session_state.llm_key_terms)
            or bool(st.session_state.llm_corrections)
        )

        if has_ai_update:
            st.session_state.caption_stage = "ai_corrected"
            st.session_state.correction_status = (
                "unclear_applied"
                if st.session_state.llm_is_unclear
                else "applied"
            )

            st.session_state.llm_last_source_text = build_slim_llm_context(
                st.session_state.llm_context_chunks,
                st.session_state.live_original,
                st.session_state.live_translation,
            )

            if st.session_state.live_running:
                st.session_state.live_control_queue.put({
                    "type": "set_base_caption",
                    "original": st.session_state.live_original,
                    "translation": st.session_state.live_translation,
                })
        else:
            st.session_state.caption_stage = "raw_english"
            st.session_state.correction_status = (
                "unclear"
                if st.session_state.llm_is_unclear
                else "no_change"
            )

        st.session_state.last_helper_fix_time = time.strftime("%H:%M:%S")

    elif item_type == "llm_error":
        st.session_state.llm_error = item.get("message", "")
        st.session_state.llm_running = False
        st.session_state.llm_last_finish_time = time.strftime("%H:%M:%S")
        # Error cooldown is longer because Groq TPM/rate errors often happen
        # when the next request starts too soon.
        st.session_state.llm_cooldown_until = time.time() + max(float(llm_hint_interval), 45.0)
        st.session_state.llm_calls_this_session += 1
        st.session_state.correction_status = "error"


# ============================================================
# Pull Ask AI results removed
# ============================================================

# Ask AI UI has been removed. Groq is used only for the live key-term helper.


# ============================================================
# Start LLM hint worker
# ============================================================

if use_llm_hints and groq_api_key and not SECOND_PASS_IN_WORKER:
    source_text = build_slim_llm_context(
        st.session_state.llm_context_chunks,
        st.session_state.live_original,
        st.session_state.live_translation,
    )

    short_term_query_ready = is_short_japanese_term_query(
        st.session_state.live_original,
    )

    enough_text = (
        len(source_text) >= int(llm_min_context_chars)
        or short_term_query_ready
    )
    changed_text = source_text != st.session_state.llm_last_source_text
    now_for_llm = time.time()
    cooldown_until = float(st.session_state.get("llm_cooldown_until", 0.0) or 0.0)
    interval_ready = now_for_llm >= cooldown_until

    translated_text_ready = bool(st.session_state.live_translation.strip()) or short_term_query_ready
    has_new_live_tokens_for_llm = (
        st.session_state.live_token_version
        > st.session_state.last_llm_checked_token_version
    )
    helper_budget_available = (
        use_llm_hints
        and not st.session_state.llm_budget_reached
        and st.session_state.llm_calls_this_session < int(llm_session_limit)
    )
    completed_phrase_ready = bool(
        st.session_state.get("last_live_endpoint", False)
        or st.session_state.pending_visual_reset
    )

    if (
        st.session_state.live_running
        and completed_phrase_ready
        and translated_text_ready
        and enough_text
        and changed_text
        and interval_ready
        and has_new_live_tokens_for_llm
        and helper_budget_available
        and not st.session_state.llm_running
    ):
        detected_terms = select_second_pass_glossary_candidates(
            active_glossary_entries,
            domain_mode,
            st.session_state.live_original,
            max_terms=MAX_GROQ_GLOSSARY_TERMS,
        )

        st.session_state.llm_running = True
        st.session_state.llm_error = ""
        # Keep the previous corrected text visible while the next AI check runs.
        # New result will replace it when ready.
        st.session_state.caption_stage = "ai_checking"
        st.session_state.correction_status = "checking"
        st.session_state.last_ai_check_time = time.strftime("%H:%M:%S")
        st.session_state.llm_last_call_time = time.time()
        st.session_state.llm_last_source_text = source_text
        st.session_state.last_llm_checked_token_version = st.session_state.live_token_version
        st.session_state.last_live_endpoint = False

        selected_class_context = make_compact_domain_context(
            domain_mode,
            self_context,
        )

        current_pair = make_context_chunk(
            st.session_state.live_original,
            st.session_state.live_translation,
        )
        prior_chunks = list(st.session_state.llm_context_chunks)
        if prior_chunks and prior_chunks[-1] == current_pair:
            prior_chunks = prior_chunks[:-1]
        previous_context_for_helper = "\n\n---\n\n".join(
            prior_chunks[-1:]
        )

        st.session_state.llm_thread = threading.Thread(
            target=llm_hint_worker,
            args=(
                st.session_state.llm_result_queue,
                groq_api_key,
                llm_model_name,
                st.session_state.live_original,
                st.session_state.live_translation,
                previous_context_for_helper,
                detected_terms,
                active_glossary_entries,
                domain_mode,
                selected_class_context,
                self_context,
                st.session_state.live_token_version,
            ),
            daemon=True,
        )

        st.session_state.llm_thread.start()


# ============================================================
# Helper AI budget safety
# ============================================================

if use_llm_hints and st.session_state.llm_calls_this_session >= int(llm_session_limit):
    st.session_state.llm_budget_reached = True
    if st.session_state.live_running:
        st.session_state.correction_status = "budget_reached"

# ============================================================
# Status
# ============================================================

if st.session_state.mobile_mic_failure_message:
    st.warning(st.session_state.mobile_mic_failure_message)

if st.session_state.live_running:
    st.success("Groq Whisper paragraph-safe live captions running.")
elif st.session_state.app_active:
    if st.session_state.mic_wait_notice:
        st.info(st.session_state.mic_wait_notice)
    else:
        st.info("Starting Groq Whisper captions...")
else:
    st.info("Live translation stopped.")

if st.session_state.live_error:
    if show_error_details or show_debug:
        st.error(st.session_state.live_error)
    else:
        st.caption(friendly_live_error(st.session_state.live_error))

if use_llm_hints and st.session_state.llm_error:
    if show_error_details or show_debug:
        st.warning(f"LLM error: {st.session_state.llm_error}")
    else:
        st.caption("AI helper skipped this update. Debug has details.")

if use_llm_hints and st.session_state.llm_budget_reached:
    st.warning("Second-pass AI session budget reached. Main Whisper and glossary translation are still active.")


# ============================================================
# Caption display data
# ============================================================

st.subheader("Live Captions")

if subtitle_display == "History":
    non_empty_history = [
        item
        for item in st.session_state.caption_history[-MAX_HISTORY_ITEMS:]
        if str(item or "").strip()
    ]
    caption_text = "\n\n".join(non_empty_history)
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

corrected_original, corrected_translation = light_domain_context_cleanup(
    corrected_original,
    corrected_translation,
    domain_mode,
)

current_source_for_display = build_llm_context(
    st.session_state.llm_context_chunks,
    st.session_state.live_original,
    st.session_state.live_translation,
)

if (
    use_llm_hints
    and not SECOND_PASS_IN_WORKER
    and st.session_state.caption_stage == "ai_corrected"
    and st.session_state.llm_corrected_source_text
    and source_text_matches_for_correction(
        current_source_for_display,
        st.session_state.llm_corrected_source_text,
    )
):
    merged_original, merged_translation = merge_ai_result_into_live_caption(
        live_original=st.session_state.live_original,
        live_translation=caption_text,
        ai_source_text=st.session_state.llm_corrected_source_text,
        ai_corrected_original=st.session_state.llm_corrected_japanese_original,
        ai_corrected_translation=st.session_state.llm_corrected_english_caption,
        corrections=st.session_state.llm_corrections,
        domain_mode=domain_mode,
    )

    # Never let an AI merge show an English translation with no Japanese
    # source behind it. Keep the pre-merge text instead of accepting a
    # merge result that dropped the Japanese side but kept the English side.
    if merged_original.strip() or not merged_translation.strip():
        corrected_original, corrected_translation = merged_original, merged_translation

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

if st.session_state.live_original and not display_english:
    display_english = "Waiting for English translation..."

source_is_corrected = (
    st.session_state.caption_stage == "ai_corrected"
    and bool(st.session_state.finalized_phrases)
)

if source_is_corrected:
    jp_status_text = "AI-corrected Japanese"
    en_status_text = "AI-corrected English"
elif st.session_state.live_original and not st.session_state.live_translation:
    jp_status_text = "Live Japanese"
    en_status_text = "Waiting for English translation..."
elif st.session_state.caption_stage == "raw_continuing":
    jp_status_text = "Live Japanese / continuing after AI correction"
    en_status_text = "Live English / continuing after AI correction"
elif st.session_state.live_translation:
    jp_status_text = "Live Japanese"
    en_status_text = "Live English translation"
else:
    jp_status_text = "Waiting for Japanese speech..."
    en_status_text = "Waiting for English translation..."


if SECOND_PASS_IN_WORKER and st.session_state.finalized_phrases:
    if st.session_state.llm_is_unclear:
        correction_status_text = (
            "Phrase finalized with cautious AI correction"
            + (
                f": {st.session_state.llm_unclear_reason}"
                if st.session_state.llm_unclear_reason
                else ""
            )
        )
    elif st.session_state.pending_new_paragraph:
        correction_status_text = "Paragraph complete; waiting for new speech"
    else:
        correction_status_text = "Second-pass correction applied per finalized phrase"
elif not use_llm_hints:
    correction_status_text = "AI correction off"
elif not groq_api_key:
    correction_status_text = "AI correction unavailable: GROQ_API_KEY missing"
elif st.session_state.llm_error:
    if show_error_details or show_debug:
        correction_status_text = f"AI helper skipped: {shorten_error_for_ui(st.session_state.llm_error)}"
    else:
        correction_status_text = "AI helper skipped this update"
elif st.session_state.llm_running:
    correction_status_text = "AI correction checking..."
elif st.session_state.llm_is_unclear and st.session_state.last_helper_fix_time:
    unclear_reason = st.session_state.llm_unclear_reason.strip()
    if unclear_reason:
        correction_status_text = (
            f"⚠️ AI checked at {st.session_state.last_helper_fix_time}, "
            f"but speech was unclear: {unclear_reason}"
        )
    else:
        correction_status_text = (
            f"⚠️ AI checked at {st.session_state.last_helper_fix_time}, "
            "but speech was unclear. Correction is cautious."
        )
elif source_is_corrected and st.session_state.last_helper_fix_time:
    correction_status_text = f"AI correction applied at {st.session_state.last_helper_fix_time}"
elif st.session_state.live_translation:
    correction_status_text = "AI correction pending"
elif st.session_state.live_original:
    correction_status_text = "Waiting for English before AI correction"
else:
    correction_status_text = "Waiting for speech"

if st.session_state.llm_running:
    jp_status_text = "Live Japanese / AI checking..."
    en_status_text = "Live English / AI checking..."
elif st.session_state.correction_status == "pending" and st.session_state.live_translation:
    en_status_text = "Live English / still updating"


if use_llm_hints:
    # Final UI guard:
    # Keep source-matched glossary terms stable for a short time, but block
    # unsupported Groq hallucinations like ABS immediately.
    current_llm_key_terms = [
        item
        for item in st.session_state.llm_key_terms
        if key_term_display_allowed(item, display_japanese)
    ]

    # Also prune the stored list, so old terms do not survive forever.
    st.session_state.llm_key_terms = current_llm_key_terms

    if current_llm_key_terms:
        # A background AI check running does not mean the currently displayed
        # terms are stale, so keep showing them instead of blanking to a
        # "checking" placeholder on every periodic re-check.
        llm_terms_lines = []

        for item in current_llm_key_terms[:SECOND_PASS_MAX_CONFIRMED_TERMS]:
            term = str(item.get("term", "")).strip()
            meaning = str(item.get("meaning", "")).strip()
            reading = str(item.get("reading", "")).strip()

            line = format_key_term_line(term, meaning, show_term_meaning, reading)

            if line and line not in llm_terms_lines:
                llm_terms_lines.append(line)

            if len(llm_terms_lines) >= SECOND_PASS_MAX_CONFIRMED_TERMS:
                break

        llm_terms_text = "\n".join(llm_terms_lines) if llm_terms_lines else "No key terms yet."
    elif st.session_state.llm_running:
        llm_terms_text = "Checking key terms..."
    else:
        llm_terms_text = "No key terms yet."

else:
    llm_terms_text = ""

safe_original = html.escape(display_japanese)
safe_caption_text = html.escape(display_english)
safe_llm_terms = html.escape(llm_terms_text)
safe_jp_status = html.escape(jp_status_text)
safe_en_status = html.escape(en_status_text)
safe_correction_status = html.escape(correction_status_text)

show_captions_in_ui = main_display_mode == "Captions + terms"

llm_html = ""

if use_llm_hints:
    llm_html = f"""
    <div>
        <div class="caption-label">Key Terms</div>
        <div class="llm-terms-box">{safe_llm_terms}</div>
    </div>
    """


# ============================================================
# Debug panel
# ============================================================

if show_debug:
    with st.expander("Debug", expanded=True):
        st.write("Engine:")
        st.code(translation_engine)

        st.write("Caption stage:")
        st.code(st.session_state.caption_stage)

        st.write("Live token version:")
        st.code(str(st.session_state.live_token_version))

        st.write("Mic wait notice:")
        st.code(str(st.session_state.get("mic_wait_notice", "")))

        st.write("WebRTC mic generation:")
        st.code(str(st.session_state.get("mic_generation", 0)))

        st.write("Metered TURN connected:")
        st.code(str(turn_server_ready))

        st.write("TURN fetch error:")
        st.code(str(turn_error))

        st.write("Microphone processing profile:")
        st.code(microphone_profile)

        st.write("Whisper input sample rate:")
        st.code(f"{AUDIO_INPUT_SAMPLE_RATE} Hz")

        st.write("Whisper model:")
        st.code(stt_model_name)

        st.write("Audio chunk target:")
        st.code(f"{stt_chunk_seconds:.1f} seconds")

        if webrtc_ctx.audio_processor:
            st.write("Mic processor frame count:")
            st.code(str(getattr(webrtc_ctx.audio_processor, "frame_count", 0)))
            st.write("Mic processor audio level:")
            st.code(str(getattr(webrtc_ctx.audio_processor, "last_audio_level", 0.0)))
            st.write("Dropped stale audio chunks:")
            st.code(str(getattr(webrtc_ctx.audio_processor, "dropped_audio_chunks", 0)))
            try:
                st.write("Audio queue size:")
                st.code(str(webrtc_ctx.audio_processor.audio_queue.qsize()))
            except Exception:
                pass

        st.write("Last LLM checked token version:")
        st.code(str(st.session_state.last_llm_checked_token_version))

        st.write("Helper calls this session:")
        st.code(str(st.session_state.llm_calls_this_session))

        st.write("Helper budget reached:")
        st.code(str(st.session_state.llm_budget_reached))

        st.write("Correction status:")
        st.code(st.session_state.correction_status)

        st.write("Last AI check time:")
        st.code(st.session_state.last_ai_check_time or "None")

        st.write("Last raw Japanese time:")
        st.code(st.session_state.last_raw_input_time or "None")

        st.write("Last raw English time:")
        st.code(st.session_state.last_raw_translation_time or "None")

        st.write("Last helper fix time:")
        st.code(st.session_state.last_helper_fix_time or "None")

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

        st.write("Debug messages:")
        st.write(st.session_state.debug_messages)

        st.write("LLM context chunks:")
        st.write(st.session_state.llm_context_chunks)

        st.write("LLM:")
        st.code(
            f"enabled={use_llm_hints}\n"
            f"running={st.session_state.llm_running}\n"
            f"pending_visual_reset={st.session_state.pending_visual_reset}\n"
            f"corrected_japanese_original={st.session_state.llm_corrected_japanese_original}\n"
            f"corrected_english_caption={st.session_state.llm_corrected_english_caption}\n"
            f"ai_patch_source_latest={extract_latest_pair_from_llm_source(st.session_state.llm_corrected_source_text)}\n"
            f"is_unclear={st.session_state.llm_is_unclear}\n"
            f"unclear_reason={st.session_state.llm_unclear_reason}\n"
            f"source_match_for_correction={source_text_matches_for_correction(current_source_for_display, st.session_state.llm_corrected_source_text)}\n"
            f"corrections={st.session_state.llm_corrections}\n"
            f"error={st.session_state.llm_error}\n"
            f"cooldown_until={st.session_state.get('llm_cooldown_until', 0.0)}\n"
            f"cooldown_remaining={max(0.0, st.session_state.get('llm_cooldown_until', 0.0) - time.time()):.1f}s\n"
            f"last_finish={st.session_state.get('llm_last_finish_time', '')}"
        )

        st.write("Japanese-only mode:")
        st.code(str(JAPANESE_ONLY_MODE))

        st.write("Selected helper class context:")
        st.code(make_selected_domain_context(domain_mode))

        st.write("Mic instance:")
        st.code(str(st.session_state.mic_instance_id))

        st.write("Groq live-caption error:")
        st.code(
            st.session_state.live_error
            if st.session_state.live_error
            else "No error"
        )


# ============================================================
# Caption display
# ============================================================

jp_section_html = ""

if show_captions_in_ui:
    jp_section_html = f"""
    <div>
        <div class="caption-label">Japanese Original <span class="caption-status">{safe_jp_status}</span></div>
        <div class="jp-caption-box">{safe_original}</div>
    </div>
    """

en_section_html = ""

if show_captions_in_ui:
    en_section_html = f"""
    <div>
        <div class="caption-label">English Caption <span class="caption-status">{safe_en_status}</span></div>
        <div class="en-caption-box">{safe_caption_text}</div>
    </div>
    """

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

.caption-status {{
    margin-left: 8px;
    padding: 2px 7px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    background-color: #1F2937;
    color: #D1D5DB;
}}

.correction-status-box {{
    font-size: 15px;
    line-height: 1.35;
    font-weight: 700;
    padding: 10px 12px;
    border-radius: 14px;
    background-color: #EFF6FF;
    color: #1E3A8A;
    min-height: 42px;
    max-height: 105px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #60A5FA;
    box-sizing: border-box;
}}

.jp-caption-box {{
    font-size: {jp_font_size}px;
    line-height: 1.35;
    padding: 12px;
    border-radius: 14px;
    background-color: #F3F4F6;
    color: #111827;
    min-height: 85px;
    max-height: 150px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #D1D5DB;
    box-sizing: border-box;
}}


.llm-terms-box {{
    font-size: 20px;
    line-height: 1.35;
    font-weight: 800;
    padding: 14px;
    border-radius: 14px;
    background-color: #ECFDF5;
    color: #064E3B;
    min-height: 60px;
    max-height: 150px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #10B981;
    box-sizing: border-box;
}}

.ask-ai-box {{
    font-size: 18px;
    line-height: 1.35;
    font-weight: 700;
    padding: 12px;
    border-radius: 14px;
    background-color: #FEF3C7;
    color: #78350F;
    min-height: 48px;
    max-height: 125px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #F59E0B;
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
    min-height: 130px;
    max-height: 220px;
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
        line-height: 1.25;
        padding: 7px;
        min-height: 52px;
        max-height: 125px;
    }}


    .correction-status-box {{
        font-size: 14px;
        line-height: 1.3;
        padding: 9px;
        min-height: 38px;
        max-height: 70px;
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
        line-height: 1.18;
        padding: 9px;
        min-height: 85px;
        max-height: 190px;
    }}
}}

@media (max-width: 768px) {{
    .caption-wrapper {{
        gap: 8px;
    }}
    .caption-label {{
        font-size: 11px;
        margin-bottom: 3px;
    }}
    .jp-caption-box {{
        font-size: min({jp_font_size}px, 15px);
        line-height: 1.22;
        padding: 7px;
        min-height: 48px;
        max-height: 115px;
    }}
    .correction-status-box {{
        font-size: 12px;
        line-height: 1.2;
        padding: 7px;
        min-height: 32px;
        max-height: 55px;
    }}
    .llm-terms-box {{
        font-size: 18px;
        line-height: 1.22;
        padding: 9px;
        min-height: 44px;
        max-height: 110px;
    }}

    .ask-ai-box {{
        font-size: 16px;
        line-height: 1.22;
        padding: 8px;
        min-height: 40px;
        max-height: 95px;
    }}
    .en-caption-box {{
        font-size: min({font_size}px, 16px);
        line-height: 1.18;
        padding: 8px;
        min-height: 80px;
        max-height: 145px;
    }}
}}
</style>

<div class="caption-wrapper">
    {jp_section_html}

    {llm_html}

    {en_section_html}
</div>
"""
st.html(caption_html)


# ============================================================
# Live refresh
# ============================================================

if st.session_state.app_active or st.session_state.live_running:
    time.sleep(0.2)
    st.rerun()