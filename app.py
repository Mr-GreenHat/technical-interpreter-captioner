import os
import csv
import html
import json
import queue
import threading
import time
import base64
import logging

import av
import numpy as np
import streamlit as st
import websocket
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
# been closed. It is cleanup noise, not a Soniox/Groq failure.

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


SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"  
DEFAULT_TERMS_FILE = "technical_terms.csv"

DEFAULT_RESET_SECONDS = 3.0
MAX_ORIGINAL_CHARS = 300
MAX_TRANSLATION_CHARS = 480
MAX_HISTORY_ITEMS = 5
MAX_DEBUG_MESSAGES = 10

# Soniox expects a steady raw s16le audio stream after connection.
# If WebRTC has not produced frames yet, send short silence frames so Soniox
# does not close with "Audio data decode timeout".
SONIOX_SILENCE_KEEPALIVE_SECONDS = 0.25
SONIOX_SILENCE_KEEPALIVE_BYTES = b"\x00\x00" * 4800  # 100 ms at 48 kHz mono s16le

# iPhone/Safari needs a new user tap after refresh before microphone capture.
# Kept generous because real ICE/TURN negotiation can take longer than a few
# seconds on slower networks; too short a timeout tears down a connection
# that was still actively negotiating and restarts it in an endless loop.
MOBILE_MIC_START_TIMEOUT_SECONDS = 25.0

# Japanese-only safety:
# Ignore accidental Spanish/English/other-language recognition.
JAPANESE_ONLY_MODE = True

SOURCE_LANG_JA_ONLY = "Japanese only"

# Helper / correction AI
# Main transcription/translation returns to Soniox.
# Second AI correction/cleanup uses Groq because it is fast and has a usable free tier.
GROQ_HELPER_FAST = "llama-3.1-8b-instant"

ENGINE_SONIOX = "Soniox STT + Soniox translation + Groq correction"

MAX_LLM_CONTEXT_CHUNKS = 2
MAX_GROQ_CONTEXT_CHARS = 1100
MAX_GROQ_TRANSLATION_CHARS = 450
MAX_GROQ_GLOSSARY_TERMS = 4

# Helper AI safety net.
# Soniox live transcription/translation keeps running, but Groq
# helper calls are limited so daily quota is protected.
LLM_BUDGET_MODES = {
    "High Accuracy": {
        "interval": 15.0,
        "min_chars": 50,
        "session_limit": 500,
        "description": "Fast helper checks. Use for short important demos.",
    },
    "Balanced": {
        "interval": 25.0,
        "min_chars": 60,
        "session_limit": 300,
        "description": "Recommended default for Soniox + Groq key-term support.",
    },
    "Saver": {
        "interval": 60.0,
        "min_chars": 160,
        "session_limit": 120,
        "description": "Safer for longer classes or low Groq quota.",
    },
    "Emergency Rule-Based Only": {
        "interval": 999999.0,
        "min_chars": 999999,
        "session_limit": 0,
        "description": "No helper AI calls. Built-in glossary cleanup only.",
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


SCHOOL_TERM_WORDS = [
    "サマーコース",
    "サマコース",
    "サマー講座",
    "Summer Course",
    "summer course",
    "BINUS",
    "Binus",
    "binus",
    "ビヌス",
    "ビナス",
    "ビーナス",
    "ネウス",
    "ヴィヌス",
    "BINUS ASO",
    "ビヌスASO",
    "ビヌス大学",
    "BINUS University",
    "ARE",
    "PDE",
    "Business Engineering",
    "Automotive and Robotics Engineering",
    "Product Design Engineering",
]

SCHOOL_STRONG_CONTEXT_WORDS = [
    "サマーコース",
    "サマコース",
    "Summer Course",
    "summer course",
    "BINUS",
    "binus",
    "ビヌス",
    "ビナス",
    "ビーナス",
    "ネウス",
    "BINUS ASO",
    "ビヌスASO",
    "ビヌス大学",
    "BINUS University",
]

CAD_STRONG_CONTEXT_WORDS = [
    "CATIA",
    "catia",
    "CAD",
    "Sketcher",
    "スケッチャー",
    "寸法拘束",
    "幾何拘束",
    "完全拘束",
    "Pad",
    "フィレット",
    "Chamfer",
    "面取り",
    "設計意図",
    "加工性",
]

AUTO_STRONG_CONTEXT_WORDS = [
    "TTC",
    "AEB",
    "ADAS",
    "慣性補償",
    "brake",
    "braking",
    "ブレーキ",
    "ロータリーエンジン",
    "rotary engine",
    "アペックスシール",
]


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


def detected_terms_to_llm_key_terms(detected_terms, max_terms=5):
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

        merged.append({
            "term": term,
            "meaning": meaning,
            "reading": item.get("reading", ""),
            "source_match": item.get("source_match", item.get("matched_candidate", "")),
            "added_at": float(item.get("added_at", time.time()) or time.time()),
        })
        seen.add(key)

        if len(merged) >= max_terms:
            break

    return merged


def filter_soniox_context_terms_for_domain(context_terms, translation_terms, domain_mode):
    """
    In Auto mode, do not bias Soniox toward Summer Course/BINUS.
    Use 'school/event' when you actually need those terms.
    """
    domain = (domain_mode or "auto").lower()

    if domain in ["school", "school/event", "event"]:
        return context_terms, translation_terms

    def keep_text(value):
        return not contains_any_text(value or "", SCHOOL_TERM_WORDS)

    filtered_context_terms = [
        term
        for term in context_terms or []
        if keep_text(term)
    ]

    filtered_translation_terms = []

    for item in translation_terms or []:
        source = item.get("source", "")
        target = item.get("target", "")

        if keep_text(source) and keep_text(target):
            filtered_translation_terms.append(item)

    return filtered_context_terms, filtered_translation_terms


def make_soniox_important_term_text(domain_mode):
    """
    Soniox context should be small and neutral.
    Do not push unrelated terms too strongly, or STT may hallucinate them.
    The real key-term truth comes from preprocessing + glossary after STT.
    """
    domain = (domain_mode or "auto").lower()

    if domain == "automotive":
        parts = [
            "TTC means Time To Collision.",
            "AEB means Autonomous Emergency Braking.",
            "慣性補償 means inertia compensation.",
            "ロータリーエンジン means rotary engine.",
            "アペックスシール means apex seal.",
        ]

    elif domain in ["cad", "product design"]:
        parts = [
            "CAD means Computer-Aided Design.",
            "治具 means jig.",
            "三面図 means three-view drawing.",
            "寸法拘束 means dimensional constraint.",
            "幾何拘束 means geometric constraint.",
            "面取り means chamfering.",
            "三角形 means triangle.",
        ]

    elif domain in ["school", "school/event", "event"]:
        parts = [
            "サマーコース means Summer Course.",
            "ビヌス大学 means BINUS University.",
            "ARE means Automotive and Robotics Engineering.",
            "PDE means Product Design Engineering.",
            "BE means Business Engineering.",
        ]

    else:
        parts = [
            "Japanese technical classroom interpretation.",
            "Only Japanese speech should be transcribed and translated.",
        ]

    return " ".join(parts)


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
        matched_candidate = ""

        for candidate in candidates:
            if not candidate:
                continue

            # Source-first key term detection:
            # Do NOT match glossary terms from English translation.
            # The English caption can be partial or wrong, and it caused
            # unrelated key terms like ABS/extrusion to appear.
            if glossary_candidate_matches(candidate, original_text):
                found = True
                matched_candidate = candidate
                break

        if found:
            matched_terms.append({
                "jp": jp,
                "en": en,
                "reading": reading,
                "notes": notes,
                "matched_candidate": matched_candidate,
            })

    def key_term_priority(item):
        jp = item.get("jp", "")
        priority = {
            "治具のCAD": 0,
            "治具": 1,
            "三面図": 2,
            "寸法拘束": 3,
            "幾何拘束": 4,
            "三角形": 5,
            "CATIA": 8,
            "CAD": 9,
        }
        return priority.get(jp, 5)

    matched_terms = sorted(matched_terms, key=key_term_priority)

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


def is_allowed_soniox_language(language_value):
    """
    If Soniox token language metadata exists, only accept Japanese.
    If metadata is absent, return True and let the text gates decide.
    """
    language_value = str(language_value or "").strip().lower()

    if not language_value:
        return True

    return language_value.startswith("ja") or language_value in ["jpn", "japanese"]


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


def friendly_soniox_error(message):
    message = str(message or "").strip()

    if not message:
        return ""

    lower = message.lower()

    if "audio data decode timeout" in lower or "audio data decode" in lower:
        return (
            "Soniox did not receive valid mic audio fast enough. "
            "Check the mic permission, then press Start Translation again."
        )

    if "request timeout" in lower or "timed out" in lower:
        return "Soniox connection timed out. Press Start Translation again."

    if "write operation timed out" in lower or "audio send error" in lower:
        return "Soniox audio connection stopped. Press Start Translation again."

    return "Soniox connection stopped. Press Start Translation again."


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
        return False

    if is_filler_only_japanese(text):
        return False

    # If Japanese ratio is very low, it is probably mostly English/noise.
    # Japanese sentence with acronyms like CAD still passes because ratio is usually > 0.15.
    if japanese_char_ratio(text) < 0.15:
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
        "Product Design": "Product Design Engineering",
        "product design engineering": "Product Design Engineering",
        "product design": "Product Design Engineering",

        "BA": "BE",
        "business engineering": "Business Engineering",
        "business engineer": "Business Engineering",

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
        "自動車ロボティクス": "ARE",
        "自動車とロボット工学": "ARE",

        "PDA": "PDE",
        "ADC": "PDE",
        "PD ": "PDE ",
        "PE ": "PDE ",
        "ピーディーイー": "PDE",
        "ピーディー": "PDE",
        "プロダクトデザイン": "PDE",
        "製品デザイン工学": "PDE",
        "製品設計": "PDE",

        "BA": "BE",
        "ビーイー": "BE",
        "ビジネス工学": "BE",
        "ビジネスエンジニアリング": "BE",

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
        "フィレ": "フィレット",
        "filet": "フィレット",
        "Fillet": "フィレット",

        "チャンファー": "Chamfer",
        "シャンファー": "Chamfer",
        "chamfer": "Chamfer",
        "面取": "面取り",
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
        "レシプロ": "レシプロエンジン",
        "ピストンエンジン": "レシプロエンジン",
        "ロータ": "ローター",
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
      parts, sketches, Pad, constraints, or modeling, 勝ち方 is often Gemini
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
    domain = (domain_mode or "auto").lower()
    self_context = compact_text(self_context, 350)

    if domain == "automotive":
        base = "Domain: automotive/AEB/braking/control. Important: TTC, AEB, ADAS, 慣性補償, braking force, rotary engine."
    elif domain == "cad":
        base = "Domain: CAD/CATIA. Important: CATIA, CAD, Sketcher, dimensional/geometric constraints, Pad, Fillet, Chamfer."
    elif domain == "product design":
        base = "Domain: product design/CAD. Important: design intent, manufacturability, CATIA, dimensions, constraints."
    elif domain in ["school", "school/event", "event"]:
        base = "Domain: school event. Important only if mentioned: Summer Course, BINUS, BINUS University, BINUS ASO, ARE, PDE, BE."
    else:
        base = "Domain: mixed Japanese technical classroom. Do not force unrelated proper nouns."

    if self_context:
        return f"{base}\nUser context: {self_context}"

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
    Prevent delete/reappear while still blocking unsupported Groq hallucinations.

    - Glossary terms with source_match hold for the entire current caption
      segment, so partial STT changes do not make them blink. They are only
      cleared when a new segment starts after a reset (see pending_visual_reset).
    - Groq-only terms with no source_match must be supported by the current
      Japanese source immediately.
    """
    source_match = str(item.get("source_match", item.get("matched_candidate", ""))).strip()

    if key_term_supported_by_source(item, source_text):
        return True

    if source_match:
        return True

    return False


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

        return {
            "corrected_japanese_original": str(data.get("corrected_japanese_original", "")).strip(),
            "corrected_english_caption": str(data.get("corrected_english_caption", "")).strip(),
            "is_unclear": (
                str(data.get("is_unclear", False)).strip().lower()
                in ["true", "1", "yes", "y"]
            ),
            "unclear_reason": str(data.get("unclear_reason", "")).strip(),
            "key_terms": data.get("key_terms", []),
            "corrections": data.get("corrections", []),
            "parse_ok": True,
            "raw_text": text or "",
        }

    except Exception:
        return empty


def llm_hint_worker(
    result_queue,
    api_key,
    model_name,
    context_text,
    current_translation,
    key_terms,
    class_context="",
    self_context="",
):
    try:
        client = Groq(api_key=api_key)

        context_text = compact_text(context_text, MAX_GROQ_CONTEXT_CHARS)
        current_translation = compact_text(current_translation, MAX_GROQ_TRANSLATION_CHARS)
        class_context = compact_text(class_context, 500)
        self_context = compact_text(self_context, 350)

        glossary_lines = []
        for item in (key_terms or [])[:MAX_GROQ_GLOSSARY_TERMS]:
            jp = item.get("jp", "")
            en = item.get("en", "")
            if jp and en:
                glossary_lines.append(f"{jp} = {en}")

        glossary_text = "\n".join(glossary_lines)

        prompt = f"""
Return ONLY JSON. No markdown.

You are the second AI after Soniox.
Main purpose: help an interpreter by extracting difficult key terms.
Also fix caption text only when the correction is obvious.

Rules:
- Output max 4 key_terms.
- Preprocessing/glossary is the source of truth. Prefer Matched glossary.
- If no glossary match, you may infer a simple key term ONLY when the Japanese source clearly contains it.
  Example: 三角形 = triangle, 時空のCAD/spacetime CAD = 治具のCAD/jig CAD, 三面図 = three-view drawing.
- Do not output unrelated key terms.
- Do not output CATIA, CAD, TTC, AEB, BINUS, BE, PDE, ARE, etc. unless they are in the current Japanese source or Matched glossary.
- Context/domain is background only, not evidence.
- Do not invent facts.
- If no key term, key_terms = [].
- Keep corrected captions close to Soniox text.
- If no correction needed, copy the current text.

Context:
{class_context}

Self context:
{self_context}

Recent caption:
{context_text}

Current English:
{current_translation}

Matched glossary:
{glossary_text}

JSON:
{{
  "corrected_japanese_original": "",
  "corrected_english_caption": "",
  "is_unclear": false,
  "unclear_reason": "",
  "key_terms": [
    {{"term": "Japanese term or acronym", "meaning": "English meaning"}}
  ],
  "corrections": [
    {{"wrong": "wrong word", "correct": "correct word", "reason": "short reason"}}
  ]
}}
""".strip()

        messages = [
            {"role": "system", "content": "You are a concise JSON-only interpreter key-term assistant."},
            {"role": "user", "content": prompt},
        ]

        text = ""
        last_error = ""

        for response_format in [{"type": "json_object"}, None]:
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 520,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                completion = client.chat.completions.create(**kwargs)
                text = completion.choices[0].message.content or ""
                if text.strip():
                    break
            except Exception as e:
                last_error = str(e)
                continue

        if not text.strip():
            raise RuntimeError(last_error or "Groq returned empty text")

        parsed = parse_llm_json(text)

        if not parsed.get("parse_ok"):
            parsed["corrected_english_caption"] = current_translation or ""
            parsed["corrected_japanese_original"] = ""
            parsed["is_unclear"] = True
            parsed["unclear_reason"] = "Groq returned non-JSON output. Raw caption kept."

        if not parsed.get("corrected_english_caption"):
            parsed["corrected_english_caption"] = current_translation or ""

        result_queue.put({
            "type": "llm_hint",
            "corrected_japanese_original": parsed.get("corrected_japanese_original", ""),
            "corrected_english_caption": parsed.get("corrected_english_caption", ""),
            "is_unclear": bool(parsed.get("is_unclear", False)),
            "unclear_reason": parsed.get("unclear_reason", ""),
            "source_text": context_text,
            "key_terms": parsed.get("key_terms", []),
            "corrections": parsed.get("corrections", []),
            "used_model": model_name,
            "raw_response_preview": text[:500],
        })

    except Exception as e:
        result_queue.put({
            "type": "llm_error",
            "message": shorten_error_for_ui(str(e)),
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
        self.frame_count = 0
        self.last_audio_time = 0.0
        self.last_audio_level = 0.0

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            resampled_frames = self.resampler.resample(frame)

            for resampled_frame in resampled_frames:
                audio = resampled_frame.to_ndarray().reshape(-1)

                if audio.size == 0:
                    continue

                pcm16 = audio.astype(np.int16)
                self.audio_queue.put(pcm16.tobytes())

                self.frame_count += 1
                self.last_audio_time = time.time()

                try:
                    self.last_audio_level = float(np.abs(pcm16).mean())
                except Exception:
                    self.last_audio_level = 0.0

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
        context_terms, translation_terms = filter_soniox_context_terms_for_domain(
            context_terms,
            translation_terms,
            domain_mode,
        )

        if domain_mode == "auto":
            domain_text = (
                "Advanced Japanese-only technical classroom interpretation. "
                "Possible topics: automotive engineering, CAD, product design, vehicle systems, "
                "braking systems, vehicle control, TTC, Time To Collision, AEB, "
                "inertia compensation, jig, mechanical drawing, three-view drawing, "
                "dimensional constraint, geometric constraint, Pad, Fillet, Chamfer, rotary engine, apex seal"
            )

        elif domain_mode == "automotive":
            domain_text = (
                "Japanese automotive engineering class, vehicle systems, braking systems, "
                "drivetrain, suspension, steering, ADAS, AEB, TTC, Time To Collision, "
                "vehicle control, inertia compensation, rotary engine, rotor, apex seal, reciprocating engine"
            )

        elif domain_mode == "cad":
            domain_text = (
                "Advanced Japanese-only CAD/mechanical drawing class. "
                "CAD, jig, fixture, three-view drawing, mechanical drawing, sketch constraints, dimensional constraints, "
                "geometric constraints, fully constrained sketch, degrees of freedom, Pad, extrusion, "
                "Hole, fillet, chamfering, projection drawing, product modeling"
            )

        elif domain_mode == "product design":
            domain_text = (
                "Japanese product design class, CAD modeling, design process, design intent, "
                "dimensions, materials, usability, strength, manufacturability, cost, product development, prototyping"
            )

        elif domain_mode in ["school", "school/event", "event"]:
            domain_text = (
                "Japanese school event interpretation, Summer Course, BINUS, BINUS ASO, "
                "BINUS University, students coming to Japan, lectures, internships, training, Japanese culture, "
                "ARE, PDE, BE majors"
            )

        else:
            domain_text = "Japanese technical classroom interpretation"

        task_text = (
            "Advanced Japanese-only mode. The target source speech is Japanese technical classroom speech. "
            "Do not force unrelated English or Indonesian into Japanese. Translate valid Japanese into English."
        )

        config = {
            "api_key": api_key,
            "model": "stt-rt-v5",
            "audio_format": "s16le",
            "sample_rate": 48000,
            "num_channels": 1,
            "enable_language_identification": True,
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
                        "value": make_soniox_important_term_text(domain_mode),
                    },
                    {
                        "key": "source_language_mode",
                        "value": SOURCE_LANG_JA_ONLY,
                    },
                    {
                        "key": "task",
                        "value": task_text,
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

        # Hint Japanese, but do NOT force it. This reduces fake Japanese
        # when English/Indonesian is spoken during testing.
        config["language_hints"] = ["ja"]

        # Drop stale audio frames before opening a fresh Soniox connection.
        # This avoids sending old buffered frames from a previous WebRTC session.
        try:
            while True:
                audio_queue.get_nowait()
        except queue.Empty:
            pass

        ws = websocket.create_connection(SONIOX_WS_URL, timeout=10)
        ws.settimeout(0.5)
        ws.send(json.dumps(config))

        result_queue.put({
            "type": "debug",
            "message": "Connected to Soniox.",
        })

        final_original = ""
        final_translation = ""
        last_token_time = time.time()
        current_reset_seconds = float(caption_reset_seconds)
        reset_sent = False

        def send_audio():
            last_audio_send_time = 0.0
            sent_silence_keepalive = False

            while not stop_event.is_set():
                try:
                    audio_bytes = audio_queue.get(timeout=0.1)

                    if audio_bytes:
                        ws.send_binary(audio_bytes)
                        last_audio_send_time = time.time()
                        sent_silence_keepalive = False

                except queue.Empty:
                    # If WebRTC is not producing frames yet, Soniox may close
                    # with "Audio data decode timeout". Send a tiny silence
                    # frame as a keepalive until real mic audio arrives.
                    now = time.time()

                    if (
                        now - last_audio_send_time
                        >= SONIOX_SILENCE_KEEPALIVE_SECONDS
                    ):
                        try:
                            ws.send_binary(SONIOX_SILENCE_KEEPALIVE_BYTES)
                            last_audio_send_time = now

                            if not sent_silence_keepalive:
                                result_queue.put({
                                    "type": "debug",
                                    "message": "Sent silence audio keepalive while waiting for mic frames.",
                                })
                                sent_silence_keepalive = True

                        except Exception as e:
                            message = f"Audio send error: {e}"
                            if not stop_event.is_set():
                                if is_connection_noise_error(message):
                                    result_queue.put({
                                        "type": "debug",
                                        "message": message,
                                    })
                                else:
                                    result_queue.put({
                                        "type": "error",
                                        "message": message,
                                    })
                            break

                    continue

                except Exception as e:
                    message = f"Audio send error: {e}"
                    if not stop_event.is_set():
                        if is_connection_noise_error(message):
                            result_queue.put({
                                "type": "debug",
                                "message": message,
                            })
                        else:
                            result_queue.put({
                                "type": "error",
                                "message": message,
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
                        # Clear worker-side accumulated text so old captions do not return.
                        final_original = ""
                        final_translation = ""
                        last_token_time = time.time()
                        reset_sent = False
                        result_queue.put({"type": "cleared"})

                    elif isinstance(command, dict) and command.get("type") == "set_base_caption":
                        # After AI correction is applied, do NOT clear the text.
                        # Use the corrected text as the new worker base, so
                        # the next live tokens continue from the fixed caption.
                        final_original = command.get("original", "") or final_original
                        final_translation = command.get("translation", "") or final_translation
                        last_token_time = time.time()
                        reset_sent = False
                        result_queue.put({
                            "type": "debug",
                            "message": "Soniox worker base updated after AI correction.",
                        })

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
                if (
                    (final_original or final_translation)
                    and not reset_sent
                    and time.time() - last_token_time > current_reset_seconds
                ):
                    result_queue.put({"type": "page_reset"})
                    reset_sent = True
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
                soniox_message = data.get("error_message", "Unknown Soniox error")
                result_queue.put({
                    "type": "error",
                    "message": soniox_message,
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

                reset_sent = False
                last_token_time = now

            non_final_original = ""
            non_final_translation = ""
            endpoint_detected = False

            # If this Soniox message contains a source token that gets rejected
            # for being non-Japanese, its paired translation token(s) in the same
            # message are for that same foreign speech and must not leak through
            # just because translation tokens aren't language-gated below.
            saw_rejected_source_token = any(
                token.get("text", "")
                and token.get("text", "") != "<end>"
                and token.get("translation_status") not in ["translation", "translated"]
                and not is_allowed_soniox_language(
                    token.get("language")
                    or token.get("language_code")
                    or token.get("detected_language")
                    or token.get("source_language")
                )
                for token in tokens
            )

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

                token_language = (
                    token.get("language")
                    or token.get("language_code")
                    or token.get("detected_language")
                    or token.get("source_language")
                )

                # Important:
                # Only source/original tokens must pass the Japanese language gate.
                # Translation tokens are target-language English, so filtering them
                # as non-Japanese makes the English caption stop early.
                if (
                    not is_translation_token
                    and not is_allowed_soniox_language(token_language)
                ):
                    continue

                # A translation token riding alongside a rejected non-Japanese
                # source token in this same message is a translation of that
                # foreign speech, not of confirmed Japanese. Drop it too.
                if is_translation_token and saw_rejected_source_token:
                    continue

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
            current_original = strip_leading_japanese_fillers(current_original)
            current_original = light_original_cleanup(current_original)

            current_translation = (final_translation + non_final_translation).strip()
            current_translation = light_caption_cleanup(current_translation)

            if should_skip_as_filler(current_original, current_translation):
                continue

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
    "Japanese → English live captions using Soniox STT/translation, "
    "with preprocessing/glossary first and Groq as an optional helper."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Settings")

    translation_engine = ENGINE_SONIOX
    st.info("Translation engine: Soniox STT + translation")
    st.caption("Second AI helper: Groq correction")

    domain_mode = st.selectbox(
        "Technical domain",
        ["auto", "automotive", "cad", "product design", "school/event"],
        index=0,
        help="Choose the real class topic. Use 'school/event' only for Summer Course/BINUS topics.",
    )
    st.caption(f"Helper AI fixed context: {domain_mode}")

    st.caption("Source mode: Soniox language identification + Japanese filter. Not forced.")

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
        help="This is sent only to Groq helper/Ask AI, not shown to audience.",
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

    st.write("LLM Interpreter Support")

    use_llm_hints = st.checkbox(
        "Use LLM support",
        value=True,
    )

    llm_model_name = GROQ_HELPER_FAST
    st.caption(f"Groq helper model: {llm_model_name}")

    llm_budget_mode = st.selectbox(
        "AI helper budget mode",
        list(LLM_BUDGET_MODES.keys()),
        index=0,
    )

    selected_budget = LLM_BUDGET_MODES[llm_budget_mode]
    llm_hint_interval = float(selected_budget["interval"])
    llm_min_context_chars = int(selected_budget["min_chars"])
    llm_session_limit = int(selected_budget["session_limit"])

    st.caption(selected_budget["description"])
    st.caption(
        f"Helper interval after finish: {int(llm_hint_interval)} sec | "
        f"Min new context: {llm_min_context_chars} chars | "
        f"Session limit: {llm_session_limit} calls"
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


# ============================================================
# Session state
# ============================================================

defaults = {
    "app_active": False,
    "pending_start_translation": False,
    "mic_instance_id": 0,
    "mic_generation": 0,
    "mobile_mic_failure_message": "",
    "current_engine": "",

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
    st.session_state.soniox_running = False
    st.session_state.soniox_stop_event.set()
    # Keep the WebRTC component stable. Recreating it repeatedly can leave
    # old aioice STUN retry timers behind on Streamlit Cloud.
    st.session_state.current_engine = translation_engine
    st.rerun()

if not st.session_state.current_engine:
    st.session_state.current_engine = ENGINE_SONIOX


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
groq_api_key = safe_get_secret_or_env("GROQ_API_KEY")

if not api_key:
    st.error(
        "SONIOX_API_KEY is not set.\n\n"
        "Soniox mode needs Soniox for Japanese STT/translation. For Streamlit Cloud, add this in Secrets:\n\n"
        'SONIOX_API_KEY = "your_soniox_api_key_here"'
    )
    st.stop()

if use_llm_hints and not groq_api_key:
    st.warning(
        "GROQ_API_KEY is not set. Second AI correction is disabled until you add it."
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
            },
            # Open Relay Project free/shared TURN fallback, used when a direct
            # STUN connection can't be established (common on Streamlit Cloud).
            # Public demo credentials, not rate-guaranteed: swap for a paid
            # provider (e.g. Metered.ca) if this becomes unreliable.
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": "turn:openrelay.metered.ca:443?transport=tcp",
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

st.subheader("Microphone")

webrtc_ctx = webrtc_streamer(
    # Stable key prevents repeated WebRTC peer-connection destruction/recreation.
    # Use the manual Reset mic connection button below if a hard reset is needed.
    key=f"soniox-live-caption-mic-{st.session_state.mic_generation}",
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

reset_mic_clicked = st.button(
    "Reset Mic Connection",
    use_container_width=True,
    help="Use only if the browser mic gets stuck. This refreshes the page and rebuilds the WebRTC connection.",
)

if reset_mic_clicked:
    st.session_state.app_active = False
    st.session_state.pending_start_translation = False
    st.session_state.soniox_running = False
    st.session_state.mic_generation += 1
    st.session_state.mobile_mic_failure_message = (
        "Mic connection reset. Tap Start Translation to enable the microphone again."
    )
    st.session_state.soniox_stop_event.set()
    st.session_state.soniox_result_queue = queue.Queue()
    st.session_state.soniox_control_queue = queue.Queue()
    st.session_state.soniox_thread = None
    st.session_state.mic_wait_start_time = 0.0
    st.session_state.mic_wait_notice = ""
    st.session_state.soniox_error = ""
    st.rerun()

if toggle_clicked:
    if st.session_state.app_active:
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.soniox_running = False
        st.session_state.soniox_stop_event.set()
        st.session_state.soniox_result_queue = queue.Queue()
        st.session_state.soniox_control_queue = queue.Queue()
        st.session_state.soniox_thread = None
        st.session_state.mic_wait_start_time = 0.0
        st.session_state.mic_wait_notice = ""

        st.rerun()

    else:
        st.session_state.soniox_stop_event = threading.Event()
        st.session_state.soniox_result_queue = queue.Queue()
        st.session_state.soniox_control_queue = queue.Queue()
        st.session_state.soniox_thread = None
        st.session_state.app_active = True
        st.session_state.pending_start_translation = True
        st.session_state.soniox_error = ""
        st.session_state.mobile_mic_failure_message = ""
        st.session_state.mic_wait_start_time = time.time()
        st.session_state.mic_wait_notice = "Waiting for browser microphone audio..."

        st.rerun()

if clear_clicked:
    st.session_state.mic_wait_start_time = 0.0
    st.session_state.mic_wait_notice = ""
    st.session_state.live_original = ""
    st.session_state.live_translation = ""
    st.session_state.caption_history = []
    st.session_state.last_update_time = ""
    st.session_state.soniox_error = ""
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

    if st.session_state.soniox_running:
        st.session_state.soniox_control_queue.put("clear")
        st.session_state.debug_messages.append("Clear requested.")
        st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]


# ============================================================
# Translator Ask AI removed
# ============================================================

# The app is now key-terms-first.
# If the speaker/interpreter says one short Japanese word and pauses,
# the normal live key-term pipeline handles it automatically.


# ============================================================
# Auto-start Soniox after WebRTC mic is actually sending audio
# ============================================================

if (
    st.session_state.pending_start_translation
    and st.session_state.app_active
    and not st.session_state.soniox_running
    and webrtc_ctx.audio_processor
):
    processor_probe = webrtc_ctx.audio_processor
    mic_frame_count = int(getattr(processor_probe, "frame_count", 0) or 0)
    mic_level = float(getattr(processor_probe, "last_audio_level", 0.0) or 0.0)
    wait_started = float(st.session_state.get("mic_wait_start_time", 0.0) or 0.0)
    waited = time.time() - wait_started if wait_started else 0.0

    if mic_frame_count <= 0:
        if waited > MOBILE_MIC_START_TIMEOUT_SECONDS:
            # iOS Safari does not automatically restart microphone capture
            # after refresh. Return to stopped state so the next Start button
            # press is a real user gesture and rebuild the WebRTC component.
            st.session_state.app_active = False
            st.session_state.pending_start_translation = False
            st.session_state.soniox_running = False
            st.session_state.soniox_stop_event.set()
            st.session_state.mic_generation += 1
            st.session_state.mic_wait_start_time = 0.0
            st.session_state.mic_wait_notice = ""
            st.session_state.mobile_mic_failure_message = (
                "Microphone did not start. Check that mic access is allowed "
                "for this site in your browser, then press Start Translation again."
            )
            st.rerun()
        else:
            st.session_state.mic_wait_notice = (
                "Waiting for browser microphone audio... "
                "Allow mic access if prompted, then speak once."
            )

    else:
        st.session_state.mic_wait_notice = (
            f"Mic audio detected. Frames: {mic_frame_count}, level: {mic_level:.1f}"
        )

        st.session_state.soniox_stop_event = threading.Event()
        st.session_state.soniox_result_queue = queue.Queue()
        st.session_state.soniox_control_queue = queue.Queue()
        st.session_state.soniox_error = ""
        st.session_state.debug_messages = []
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
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

        processor = processor_probe

        st.session_state.soniox_running = True
        st.session_state.pending_start_translation = False
        st.session_state.mic_wait_notice = ""
        st.session_state.mic_wait_start_time = 0.0

        worker_target = soniox_live_worker
        worker_args = (
            processor.audio_queue,
            st.session_state.soniox_result_queue,
            st.session_state.soniox_stop_event,
            st.session_state.soniox_control_queue,
            api_key,
            terms_file,
            domain_mode,
            float(reset_seconds),
        )

        st.session_state.soniox_thread = threading.Thread(
            target=worker_target,
            args=worker_args,
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
                    f"Ignored non-Japanese or low-quality Soniox text: {original[:80]}"
                )
                st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
                continue

            if translation and not original and not looks_like_valid_japanese_for_display(st.session_state.live_original):
                st.session_state.debug_messages.append(
                    f"Ignored translation without Japanese source: {translation[:80]}"
                )
                st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]
                continue

        if original or translation:
            st.session_state.live_token_version += 1
            prepare_next_ai_check_after_new_live_text()
            # New live speech means the previous unclear warning/key terms may be stale.
            st.session_state.llm_is_unclear = False
            st.session_state.llm_unclear_reason = ""

            # First layer: check our glossary immediately.
            # This makes terms like 三角形 = triangle appear without waiting for Groq.
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

            # Keep key terms stable during the current caption segment.
            # Live STT partials can temporarily remove words, so re-filtering old
            # terms on every rerun makes the key-term box delete/reappear.
            # We clear terms when a new caption segment starts, not on every token.
            st.session_state.llm_key_terms = merge_key_terms_preserve_order(
                st.session_state.llm_key_terms,
                instant_key_terms,
                max_terms=5,
            )

        if st.session_state.pending_visual_reset and (original or translation):
            st.session_state.live_original = ""
            st.session_state.live_translation = ""
            st.session_state.caption_history = []
            st.session_state.llm_corrected_japanese_original = ""
            st.session_state.llm_corrected_english_caption = ""
            st.session_state.llm_corrected_source_text = ""
            st.session_state.llm_is_unclear = False
            st.session_state.llm_unclear_reason = ""
            st.session_state.llm_key_terms = []
            st.session_state.llm_corrections = []
            st.session_state.caption_stage = "raw_started"
            st.session_state.last_helper_fix_time = ""
            st.session_state.last_ai_check_time = ""
            st.session_state.correction_status = "pending"
            st.session_state.pending_visual_reset = False

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
            # Use current Soniox translation directly.
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

        # Do not clear the visible caption immediately after pause.
        # Keep it on screen so the reader has time to read it.
        # The next incoming token will clear/replace the old caption.
        st.session_state.pending_visual_reset = True
        if st.session_state.live_translation and use_llm_hints and groq_api_key:
            st.session_state.correction_status = "pending"

    elif item_type == "cleared":
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
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
            st.session_state.soniox_error = error_message
        st.session_state.soniox_running = False
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.mic_wait_start_time = 0.0
        st.session_state.mic_wait_notice = ""

    elif item_type == "stopped":
        st.session_state.soniox_running = False


# ============================================================
# Pull LLM results
# ============================================================

while not st.session_state.llm_result_queue.empty():
    item = st.session_state.llm_result_queue.get()
    item_type = item.get("type")

    if item_type == "llm_hint":
        st.session_state.llm_corrected_japanese_original = item.get("corrected_japanese_original", "")
        st.session_state.llm_corrected_english_caption = item.get("corrected_english_caption", "")
        st.session_state.llm_corrected_source_text = item.get("source_text", "")
        st.session_state.llm_is_unclear = bool(item.get("is_unclear", False))
        st.session_state.llm_unclear_reason = item.get("unclear_reason", "")
        groq_terms = filter_llm_key_terms_for_current_caption(
            item.get("key_terms", []),
            st.session_state.live_original,
            "",  # do not use English translation as proof for Groq key terms
            domain_mode,
        )
        for term_item in groq_terms:
            term_item["added_at"] = time.time()

        st.session_state.llm_key_terms = merge_key_terms_preserve_order(
            st.session_state.llm_key_terms,
            groq_terms,
            max_terms=5,
        )
        st.session_state.llm_corrections = item.get("corrections", [])
        st.session_state.llm_error = ""
        st.session_state.llm_running = False
        st.session_state.llm_last_finish_time = time.strftime("%H:%M:%S")
        # Cooldown starts AFTER the helper finishes, not when it starts.
        # This prevents rapid back-to-back Groq calls that can trigger TPM/rate errors.
        st.session_state.llm_cooldown_until = time.time() + float(llm_hint_interval)
        st.session_state.llm_calls_this_session += 1

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

            # Important:
            # Do NOT replace the whole live caption with the second-AI answer.
            # The second AI may have corrected an older segment while new Soniox
            # text has already arrived. Patch only the source segment the AI saw.
            corrected_base_original, corrected_base_translation = merge_ai_result_into_live_caption(
                live_original=st.session_state.live_original,
                live_translation=st.session_state.live_translation,
                ai_source_text=st.session_state.llm_corrected_source_text,
                ai_corrected_original=st.session_state.llm_corrected_japanese_original,
                ai_corrected_translation=st.session_state.llm_corrected_english_caption,
                corrections=st.session_state.llm_corrections,
                domain_mode=domain_mode,
            )

            st.session_state.live_original = corrected_base_original
            st.session_state.live_translation = corrected_base_translation

            if st.session_state.soniox_running:
                st.session_state.soniox_control_queue.put({
                    "type": "set_base_caption",
                    "original": corrected_base_original,
                    "translation": corrected_base_translation,
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

if use_llm_hints and groq_api_key:
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
        and llm_budget_mode != "Emergency Rule-Based Only"
    )

    if (
        st.session_state.soniox_running
        and translated_text_ready
        and enough_text
        and changed_text
        and interval_ready
        and has_new_live_tokens_for_llm
        and helper_budget_available
        and not st.session_state.llm_running
    ):
        detected_terms = extract_key_terms_for_llm(
            st.session_state.live_original,
            st.session_state.live_translation,
            terms_file,
        )
        detected_terms = filter_detected_terms_for_current_caption(
            detected_terms,
            st.session_state.live_original,
            st.session_state.live_translation,
            domain_mode,
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

        selected_class_context = make_compact_domain_context(
            domain_mode,
            self_context,
        )

        st.session_state.llm_thread = threading.Thread(
            target=llm_hint_worker,
            args=(
                st.session_state.llm_result_queue,
                groq_api_key,
                llm_model_name,
                source_text,
                (
                    st.session_state.live_translation
                    if st.session_state.live_translation.strip()
                    else st.session_state.live_original
                ),
                detected_terms,
                selected_class_context,
                self_context,
            ),
            daemon=True,
        )

        st.session_state.llm_thread.start()


# ============================================================
# Helper AI budget safety
# ============================================================

if use_llm_hints:
    if llm_budget_mode == "Emergency Rule-Based Only":
        st.session_state.llm_budget_reached = True
        st.session_state.correction_status = "off"
    elif st.session_state.llm_calls_this_session >= int(llm_session_limit):
        st.session_state.llm_budget_reached = True
        if st.session_state.soniox_running:
            st.session_state.correction_status = "budget_reached"

# ============================================================
# Status
# ============================================================

if st.session_state.mobile_mic_failure_message:
    st.warning(st.session_state.mobile_mic_failure_message)

if st.session_state.soniox_running:
    st.success("Soniox STT/translation running.")
elif st.session_state.app_active:
    if st.session_state.mic_wait_notice:
        st.info(st.session_state.mic_wait_notice)
    else:
        st.info("Starting Soniox STT/translation...")
else:
    st.info("Live translation stopped.")

if st.session_state.soniox_error:
    if show_error_details or show_debug:
        st.error(st.session_state.soniox_error)
    else:
        st.caption(friendly_soniox_error(st.session_state.soniox_error))

if use_llm_hints and st.session_state.llm_error:
    if show_error_details or show_debug:
        st.warning(f"LLM error: {st.session_state.llm_error}")
    else:
        st.caption("AI helper skipped this update. Debug has details.")

if use_llm_hints and st.session_state.llm_budget_reached:
    if llm_budget_mode == "Emergency Rule-Based Only":
        st.warning("Helper AI is off. Rule-based glossary correction is still active.")
    else:
        st.warning("Helper AI session budget reached. Switched to rule-based glossary correction only.")


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
    and st.session_state.caption_stage == "ai_corrected"
    and st.session_state.llm_corrected_source_text
    and source_text_matches_for_correction(
        current_source_for_display,
        st.session_state.llm_corrected_source_text,
    )
):
    corrected_original, corrected_translation = merge_ai_result_into_live_caption(
        live_original=st.session_state.live_original,
        live_translation=caption_text,
        ai_source_text=st.session_state.llm_corrected_source_text,
        ai_corrected_original=st.session_state.llm_corrected_japanese_original,
        ai_corrected_translation=st.session_state.llm_corrected_english_caption,
        corrections=st.session_state.llm_corrections,
        domain_mode=domain_mode,
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

if st.session_state.live_original and not display_english:
    display_english = "Waiting for English translation..."

source_is_corrected = (
    use_llm_hints
    and st.session_state.caption_stage == "ai_corrected"
    and st.session_state.llm_corrected_source_text
    and source_text_matches_for_correction(
        current_source_for_display,
        st.session_state.llm_corrected_source_text,
    )
    and (
        bool(st.session_state.llm_corrected_japanese_original)
        or bool(st.session_state.llm_corrected_english_caption)
    )
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


if not use_llm_hints:
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

        for item in current_llm_key_terms[:8]:
            term = str(item.get("term", "")).strip()
            meaning = str(item.get("meaning", "")).strip()
            reading = str(item.get("reading", "")).strip()

            line = format_key_term_line(term, meaning, show_term_meaning, reading)

            if line and line not in llm_terms_lines:
                llm_terms_lines.append(line)

            if len(llm_terms_lines) >= 5:
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

        if webrtc_ctx.audio_processor:
            st.write("Mic processor frame count:")
            st.code(str(getattr(webrtc_ctx.audio_processor, "frame_count", 0)))
            st.write("Mic processor audio level:")
            st.code(str(getattr(webrtc_ctx.audio_processor, "last_audio_level", 0.0)))

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

        st.write("Soniox error:")
        st.code(
            st.session_state.soniox_error
            if st.session_state.soniox_error
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

if st.session_state.app_active or st.session_state.soniox_running:
    time.sleep(0.2)
    st.rerun()