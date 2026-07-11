import os
import csv
import html
import json
import queue
import threading
import time
import asyncio
import base64

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

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"  # unused; kept for compatibility
DEFAULT_TERMS_FILE = "technical_terms.csv"

DEFAULT_RESET_SECONDS = 3.0
MAX_ORIGINAL_CHARS = 300
MAX_TRANSLATION_CHARS = 480
MAX_HISTORY_ITEMS = 5
MAX_DEBUG_MESSAGES = 10

# Gemini Live low-latency audio send settings.
# 40 ms at 16 kHz mono int16 = 1280 bytes.
GEMINI_LIVE_AUDIO_CHUNK_BYTES = 1280
GEMINI_LIVE_AUDIO_FLUSH_SECONDS = 0.05

# Helper / correction AI
LLM_MODEL_DEFAULT = "gemini-3.1-flash-lite"
LLM_MODEL_BACKUP = "gemini-3.1-flash-lite"

# Translation model for Gemini Live Translate mode
GEMINI_LIVE_TRANSLATE_MODEL = "gemini-3.5-live-translate-preview"
GEMINI_LIVE_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

ENGINE_GEMINI_LIVE = "Gemini Mode - Gemini 3.5 Live Translate + Gemini 3.1 correction"
ENGINE_SONIOX = ENGINE_GEMINI_LIVE  # compatibility only; Soniox is disabled in this version

DEFAULT_LLM_HINT_INTERVAL = 45.0
MIN_LLM_CONTEXT_CHARS = 180
MAX_LLM_CONTEXT_CHUNKS = 6

# Helper AI safety net.
# Gemini 3.5 Live Translate can keep running, but Gemini 3.1 Flash-Lite
# helper calls are limited so daily quota is protected.
LLM_BUDGET_MODES = {
    "High Accuracy": {
        "interval": 20.0,
        "min_chars": 120,
        "session_limit": 120,
        "description": "More frequent helper AI checks. Use only for short important demos.",
    },
    "Balanced": {
        "interval": 45.0,
        "min_chars": 180,
        "session_limit": 80,
        "description": "Recommended default for normal classes.",
    },
    "Saver": {
        "interval": 90.0,
        "min_chars": 260,
        "session_limit": 40,
        "description": "Safer for multiple classes per day.",
    },
    "Emergency Rule-Based Only": {
        "interval": 999999.0,
        "min_chars": 999999,
        "session_limit": 0,
        "description": "No Gemini 3.1 helper calls. Built-in glossary cleanup only.",
    },
}


# Built-in non-technical / school event terms.
# These are always added even when technical_terms.csv does not include them.
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
        "jp": "CATIA",
        "reading": "きゃてぃあ",
        "en": "CATIA",
        "common_wrong": "キャティア;カティア;キャディア;カディア;勝ち方;書き方;キャリア;Catia;catia;CADIA;way to win",
        "notes": "CAD software used for product design and engineering",
    },
    {
        "domain": "cad",
        "jp": "CAD",
        "reading": "きゃど",
        "en": "CAD",
        "common_wrong": "キャド;cad;computer aided design;Computer Aided Design;Computer-Aided Design",
        "notes": "Computer-Aided Design",
    },
    {
        "domain": "cad",
        "jp": "スケッチャー",
        "reading": "すけっちゃー",
        "en": "Sketcher",
        "common_wrong": "スケッチ;Sketcher;sketcher;sketch",
        "notes": "CATIA sketch workspace",
    },
    {
        "domain": "cad",
        "jp": "寸法拘束",
        "reading": "すんぽうこうそく",
        "en": "dimensional constraint",
        "common_wrong": "寸法高速;寸法校則;寸法公則;dimension constraint;dimensional constraints",
        "notes": "Constraint that defines numerical dimensions",
    },
    {
        "domain": "cad",
        "jp": "幾何拘束",
        "reading": "きかこうそく",
        "en": "geometric constraint",
        "common_wrong": "幾何高速;記号拘束;幾何校則;geometry constraint;geometrical constraint",
        "notes": "Constraint that defines geometric relationships",
    },
    {
        "domain": "cad",
        "jp": "完全拘束",
        "reading": "かんぜんこうそく",
        "en": "fully constrained",
        "common_wrong": "完全高速;完全校則;full constraint;fully constraint",
        "notes": "Sketch condition where no degrees of freedom remain",
    },
    {
        "domain": "cad",
        "jp": "自由度",
        "reading": "じゆうど",
        "en": "degrees of freedom",
        "common_wrong": "自由道;degree of freedom;degrees of freedom",
        "notes": "Remaining movement/undetermined state in a sketch",
    },
    {
        "domain": "cad",
        "jp": "Pad",
        "reading": "ぱっど",
        "en": "Pad",
        "common_wrong": "パッド;pad;extrude;extrusion;押し出し",
        "notes": "CATIA function used to extrude a sketch",
    },
    {
        "domain": "cad",
        "jp": "押し出し",
        "reading": "おしだし",
        "en": "extrusion",
        "common_wrong": "押出し;押し出す;extrude;extrusion",
        "notes": "Creating 3D geometry by extruding a sketch",
    },
    {
        "domain": "cad",
        "jp": "フィレット",
        "reading": "ふぃれっと",
        "en": "fillet",
        "common_wrong": "フィレ;fillet;Fillet;filet",
        "notes": "Rounded edge feature",
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
        "domain": "cad",
        "jp": "面取り",
        "reading": "めんとり",
        "en": "chamfering",
        "common_wrong": "面取;面どり;chamfer;chamfering",
        "notes": "Removing or beveling a sharp edge",
    },
    {
        "domain": "cad",
        "jp": "設計意図",
        "reading": "せっけいいと",
        "en": "design intent",
        "common_wrong": "設計糸;設計意図;design intent",
        "notes": "Reasoning behind design dimensions and features",
    },
    {
        "domain": "cad",
        "jp": "加工性",
        "reading": "かこうせい",
        "en": "manufacturability",
        "common_wrong": "加工製;加工生;manufacturability;manufacturing feasibility",
        "notes": "How easy or realistic a part is to manufacture",
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
    ]

    is_cad_domain = domain in ["auto", "cad", "product design"]
    has_cad_context = any(word in combined for word in cad_context_words)

    if is_cad_domain and has_cad_context:
        original_replacements = {
            "勝ち方": "CATIA",
            "書き方": "CATIA",
            "キャリア": "CATIA",
            "カチア": "CATIA",
            "勝ティア": "CATIA",
            "勝ちア": "CATIA",
            "キャティア": "CATIA",
            "カティア": "CATIA",
            "キャディア": "CATIA",
            "カディア": "CATIA",
        }

        translation_replacements = {
                        "winning method": "CATIA",
            "So, the way to win": "So, in CATIA",
            "So the way to win": "So in CATIA",
            "the winning method": "CATIA",
            "career": "CATIA",
            "Carrier": "CATIA",
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
        "way to win": ("CATIA", "CATIA"),
        "how to win": ("CATIA", "CATIA"),
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

def downsample_pcm48_to_pcm16(pcm48_bytes):
    """
    Browser/WebRTC audio is resampled to 48 kHz in AudioProcessor.
    Gemini Live Translate expects raw 16-bit PCM mono at 16 kHz.
    This simple downsampler keeps every 3rd sample.

    It is not studio-quality resampling, but good enough for speech testing.
    """
    if not pcm48_bytes:
        return b""

    audio = np.frombuffer(pcm48_bytes, dtype=np.int16)

    if audio.size == 0:
        return b""

    audio16 = audio[::3].astype(np.int16)
    return audio16.tobytes()


def append_stream_text(old_text, new_text, max_chars=800):
    old_text = old_text or ""
    new_text = new_text or ""

    if not new_text:
        return old_text.strip()

    if old_text and old_text[-1] not in [" ", "\n", "。", "、", ".", "?", "!", "！", "？"]:
        combined = old_text + " " + new_text
    else:
        combined = old_text + new_text

    while "  " in combined:
        combined = combined.replace("  ", " ")

    if len(combined) > max_chars:
        combined = combined[-max_chars:]

    return combined.strip()


def make_gemini_live_setup(target_language_code="en"):
    return {
        "setup": {
            "model": f"models/{GEMINI_LIVE_TRANSLATE_MODEL}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "translationConfig": {
                    "targetLanguageCode": target_language_code,
                    "echoTargetLanguage": True,
                },
            },
        }
    }


def extract_live_text_from_response(data):
    """
    Raw Gemini Live websocket response uses lowerCamelCase.
    We only need inputTranscription and outputTranscription text.
    """
    input_text = ""
    output_text = ""

    server_content = data.get("serverContent") or data.get("server_content") or {}

    input_transcription = (
        server_content.get("inputTranscription")
        or server_content.get("input_transcription")
        or {}
    )

    output_transcription = (
        server_content.get("outputTranscription")
        or server_content.get("output_transcription")
        or {}
    )

    if isinstance(input_transcription, dict):
        input_text = input_transcription.get("text", "") or ""

    if isinstance(output_transcription, dict):
        output_text = output_transcription.get("text", "") or ""

    return input_text, output_text



def make_gemini_live_sdk_config(target_language_code="en"):
    """
    SDK config for Gemini 3.5 Live Translate.

    Correct model roles:
    - Translation engine: gemini-3.5-live-translate-preview
    - Helper/correction AI: gemini-3.1-flash-lite
    """
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=target_language_code,
            echo_target_language=True,
        ),
    )


def gemini_live_translate_worker(
    audio_queue,
    result_queue,
    stop_event,
    control_queue,
    api_key,
    target_language_code="en",
    caption_reset_seconds=DEFAULT_RESET_SECONDS,
):
    """
    Gemini 3.5 Live Translate worker using the official google-genai SDK.

    Translation engine:
        gemini-3.5-live-translate-preview

    Helper/correction AI:
        gemini-3.1-flash-lite, handled separately by llm_hint_worker.

    This keeps AudioProcessor and Soniox worker untouched.
    """

    async def run_live_session():
        client = genai.Client(api_key=api_key)
        config = make_gemini_live_sdk_config(target_language_code)

        result_queue.put({
            "type": "debug",
            "message": "Connecting with official Gemini Live SDK...",
        })

        live_original = ""
        live_translation = ""
        last_text_time = time.time()
        reset_sent = False
        first_input_seen = False
        first_output_seen = False

        async with client.aio.live.connect(
            model=GEMINI_LIVE_TRANSLATE_MODEL,
            config=config,
        ) as session:
            result_queue.put({
                "type": "debug",
                "message": "Gemini 3.5 Live Translate session started.",
            })

            async def send_audio_loop():
                nonlocal live_original
                nonlocal live_translation
                nonlocal last_text_time
                nonlocal reset_sent
                nonlocal first_input_seen
                nonlocal first_output_seen

                pcm16_buffer = bytearray()
                last_send_time = time.time()

                while not stop_event.is_set():
                    while control_queue is not None and not control_queue.empty():
                        try:
                            command = control_queue.get_nowait()

                            if command == "clear":
                                # Clear the Gemini worker's internal accumulated text too.
                                # Otherwise old text can return after pressing Clear Captions.
                                pcm16_buffer = bytearray()
                                live_original = ""
                                live_translation = ""
                                last_text_time = time.time()
                                reset_sent = False
                                first_input_seen = False
                                first_output_seen = False
                                result_queue.put({"type": "cleared"})

                            elif isinstance(command, dict) and command.get("type") == "set_base_caption":
                                # After AI correction is applied, do NOT clear the text.
                                # Use the corrected text as the new worker base, so
                                # the next live tokens continue from the fixed caption.
                                live_original = command.get("original", "") or live_original
                                live_translation = command.get("translation", "") or live_translation
                                pcm16_buffer = bytearray()
                                last_text_time = time.time()
                                reset_sent = False
                                first_input_seen = bool(live_original)
                                first_output_seen = bool(live_translation)
                                result_queue.put({
                                    "type": "debug",
                                    "message": "Gemini worker base updated after AI correction.",
                                })

                        except queue.Empty:
                            break

                    try:
                        pcm48 = await asyncio.to_thread(audio_queue.get, True, 0.05)

                        if pcm48:
                            pcm16 = downsample_pcm48_to_pcm16(pcm48)

                            if pcm16:
                                pcm16_buffer.extend(pcm16)

                    except queue.Empty:
                        pass

                    # Low latency:
                    # Send about 40 ms of 16 kHz mono int16 audio per packet.
                    # This helps Gemini receive speech earlier.
                    now = time.time()
                    should_send = (
                        len(pcm16_buffer) >= GEMINI_LIVE_AUDIO_CHUNK_BYTES
                        or (
                            pcm16_buffer
                            and now - last_send_time >= GEMINI_LIVE_AUDIO_FLUSH_SECONDS
                        )
                    )

                    if not should_send:
                        await asyncio.sleep(0.005)
                        continue

                    chunk = bytes(pcm16_buffer[:GEMINI_LIVE_AUDIO_CHUNK_BYTES])
                    pcm16_buffer = pcm16_buffer[GEMINI_LIVE_AUDIO_CHUNK_BYTES:]
                    last_send_time = now

                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )

                    await asyncio.sleep(0.003)

            async def receive_loop():
                nonlocal live_original
                nonlocal live_translation
                nonlocal last_text_time
                nonlocal reset_sent
                nonlocal first_input_seen
                nonlocal first_output_seen

                async for response in session.receive():
                    if stop_event.is_set():
                        break

                    server_content = getattr(response, "server_content", None)

                    if not server_content:
                        continue

                    input_transcription = getattr(
                        server_content,
                        "input_transcription",
                        None,
                    )

                    output_transcription = getattr(
                        server_content,
                        "output_transcription",
                        None,
                    )

                    input_text = ""
                    output_text = ""

                    if input_transcription:
                        input_text = getattr(input_transcription, "text", "") or ""

                    if output_transcription:
                        output_text = getattr(output_transcription, "text", "") or ""

                    if input_text or output_text:
                        last_text_time = time.time()
                        reset_sent = False

                    if input_text:
                        if not first_input_seen:
                            first_input_seen = True
                            result_queue.put({
                                "type": "debug",
                                "message": "Gemini Live input transcription started.",
                            })

                        live_original = append_stream_text(
                            live_original,
                            light_original_cleanup(input_text),
                            max_chars=MAX_ORIGINAL_CHARS * 2,
                        )

                        # Show Japanese immediately, even before English translation arrives.
                        result_queue.put({
                            "type": "tokens",
                            "original": live_original,
                            "translation": live_translation,
                            "endpoint": False,
                        })

                    if output_text:
                        if not first_output_seen:
                            first_output_seen = True
                            result_queue.put({
                                "type": "debug",
                                "message": "Gemini Live output translation started.",
                            })

                        live_translation = append_stream_text(
                            live_translation,
                            light_caption_cleanup(output_text),
                            max_chars=MAX_TRANSLATION_CHARS * 2,
                        )

                        # English appears when Gemini finishes/streams translation.
                        result_queue.put({
                            "type": "tokens",
                            "original": live_original,
                            "translation": live_translation,
                            "endpoint": False,
                        })

            async def reset_watchdog_loop():
                nonlocal live_original
                nonlocal live_translation
                nonlocal last_text_time
                nonlocal reset_sent

                while not stop_event.is_set():
                    await asyncio.sleep(0.25)

                    if not live_original and not live_translation:
                        continue

                    if time.time() - last_text_time >= float(caption_reset_seconds) and not reset_sent:
                        result_queue.put({"type": "page_reset"})
                        live_original = ""
                        live_translation = ""
                        reset_sent = True

            send_task = asyncio.create_task(send_audio_loop())
            receive_task = asyncio.create_task(receive_loop())
            reset_task = asyncio.create_task(reset_watchdog_loop())

            done, pending = await asyncio.wait(
                [send_task, receive_task, reset_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()

            for task in done:
                error = task.exception()
                if error:
                    raise error

    try:
        asyncio.run(run_live_session())

    except Exception as e:
        if not stop_event.is_set():
            result_queue.put({
                "type": "error",
                "message": f"Gemini Live SDK error: {e}",
            })

    finally:
        result_queue.put({
            "type": "stopped",
        })


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
            "corrected_japanese_original": "",
            "corrected_english_caption": "",
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
            "corrected_japanese_original": str(data.get("corrected_japanese_original", "")).strip(),
            "corrected_english_caption": str(data.get("corrected_english_caption", "")).strip(),
            "key_terms": data.get("key_terms", []),
            "corrections": data.get("corrections", []),
        }

    except Exception:
        return {
            "main_idea": cleaned[:220],
            "say_it_simply": "",
            "corrected_japanese_original": "",
            "corrected_english_caption": "",
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
- Do not summarize the speaker.
- Keep the speaker's perspective. If the speaker says "I" or "we", keep "I" or "we".
- Do not rewrite "I" as "the speaker" unless the original meaning is third-person.
- Use previous context only to repair unclear wording and technical terms.
- Repair obvious STT/translation mistakes in technical terms quickly.
- Repair awkward English sentence structure so the caption sounds natural.
- Return the corrected_japanese_original and corrected_english_caption even for short segments.
- Keep the corrected English caption close to the current English translation.
- Preserve technical terms from the glossary.
- If the transcript is unclear, make the safest minimal correction.
- If STT or translation uses a wrong technical term, add it to corrections.
- If the caption says ABC but the context means TTC / Time To Collision, correct ABC to TTC.
- Prefer corrected technical terms in key_terms.
- Also fix obvious Japanese Original mistakes in corrected_japanese_original.
- corrected_japanese_original must stay Japanese and close to the original.
- Do not invent missing Japanese. Only repair obvious wrong technical words.
- Example Japanese correction: 感性の影響 -> 慣性の影響.
- Example Japanese correction: 慣性補償性制御 -> 慣性補償制御.
- Example Japanese correction: 完成の駅 -> 慣性の影響.
- For key_terms, use the Japanese source technical term when possible, for example "慣性補償 = inertia compensation".
- Do not output English-to-English key terms like "inertia compensation = Control technique...".
- Also preserve school/event names:
  サマーコース = Summer Course
  ビヌス = BINUS
  ビヌスASO = BINUS ASO
  ビヌス大学 = BINUS University
  ARE = Automotive and Robotics Engineering
  PDE = Product Design Engineering
  BE = Business Engineering
  CATIA = CATIA
  CAD = Computer-Aided Design
  ロータリーエンジン = rotary engine
- CATIA / CAD correction rules:
  Preserve CATIA and CAD exactly as acronyms.
  If the CAD/Product Design topic is clearly about parts, sketches, constraints, Pad, or modeling,
  correct 勝ち方 / 書き方 / キャリア / "way to win" to CATIA.
  Do NOT correct 勝ち方 when the topic is truly about winning or competition.
  スケッチャー = Sketcher
  寸法拘束 = dimensional constraint
  幾何拘束 = geometric constraint
  完全拘束 = fully constrained
  自由度 = degrees of freedom
  Pad = Pad / extrusion
  フィレット = fillet
  Chamfer / 面取り = chamfer / chamfering
  設計意図 = design intent
  加工性 = manufacturability
- Rotary engine correction rules:
  ロータリーエンジン = rotary engine
  レシプロエンジン = reciprocating engine
  ローター = rotor
  アペックスシール = apex seal
- Major-name correction rules:
  If the topic is BINUS ASO majors or students, correct AROI / ARO to ARE when it means Automotive and Robotics Engineering.
  If the topic is BINUS ASO majors or students, correct PDA / ADC / PD to PDE when it means Product Design Engineering.
  If the topic is BINUS ASO majors or students, correct BA / B to BE only when it clearly means Business Engineering.
  Do not change normal words to ARE/PDE/BE unless the context is clearly about majors/programs.
- Context-only Summer Course rule:
  If the topic is clearly the ASO/BINUS Summer Course program, and the transcript says 様様, 様々, さまざま, or チーム where "course/program" makes more sense, correct it to サマーコース.
  Do NOT correct チーム when it really means team.
  Do NOT correct 様々 or さまざま when it really means various.
- If the transcript says ネウス大学, ビーナス大学, or ビナス大学 in this school context, correct it to ビヌス大学.
- Key terms may include important school/event names when relevant.
- Only output important technical words or important proper nouns, not normal words.
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
  "corrected_japanese_original": "corrected Japanese transcript, only fixing obvious technical recognition errors, otherwise copy the Japanese original",
  "corrected_english_caption": "corrected natural English version of the current English translation, keeping speaker perspective and not summarizing",
  "key_terms": [
    {{"term": "Japanese source technical term or important acronym", "meaning": "short English meaning"}}
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
                max_output_tokens=650,
            ),
        )

        parsed = parse_llm_json(response.text)

        result_queue.put({
            "type": "llm_hint",
            "main_idea": parsed.get("main_idea", ""),
            "say_it_simply": parsed.get("say_it_simply", ""),
            "corrected_japanese_original": parsed.get("corrected_japanese_original", ""),
            "corrected_english_caption": parsed.get("corrected_english_caption", ""),
            "source_text": context_text,
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
                "inertia compensation, classroom interpretation, technical terms, "
                "Summer Course, BINUS, BINUS ASO, BINUS University, ビヌス大学, "
                "ARE, Automotive and Robotics Engineering, PDE, Product Design Engineering, "
                "BE, Business Engineering, CATIA, CAD, Sketcher, dimensional constraint, "
                "geometric constraint, Pad, Fillet, Chamfer, rotary engine, apex seal"
            )

        elif domain_mode == "automotive":
            domain_text = (
                "Japanese automotive engineering class, vehicle systems, braking systems, "
                "drivetrain, suspension, steering, ADAS, AEB, TTC, Time To Collision, "
                "vehicle control, inertia compensation, rotary engine, rotor, apex seal, reciprocating engine"
            )

        elif domain_mode == "cad":
            domain_text = (
                "Japanese CAD class, CATIA, CAD, Sketcher, sketch constraints, dimensional constraints, "
                "geometric constraints, fully constrained sketch, degrees of freedom, Pad, extrusion, "
                "Hole, fillet, chamfering, technical drawing, projection drawing, product modeling"
            )

        elif domain_mode == "product design":
            domain_text = (
                "Japanese product design class, CATIA, CAD modeling, design process, design intent, "
                "dimensions, materials, usability, strength, manufacturability, cost, product development, prototyping"
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
                            "or completion compensation. "
                            "サマーコース means Summer Course. "
                            "In the ASO/BINUS Summer Course context only, if a phrase sounds like 様様, 様々, さまざま, or チーム but the sentence is clearly about the course/program, it may mean サマーコース. Do not force this correction when it really means various or team. "
                            "ビヌス means BINUS. ビヌスASO means BINUS ASO. "
                            "ビヌス大学 means BINUS University. "
                            "ARE means Automotive and Robotics Engineering. "
                            "PDE means Product Design Engineering. "
                            "BE means Business Engineering. "
                            "If speech sounds like AROI or ARO in the BINUS ASO major context, it is probably ARE. "
                            "If speech sounds like PDA or ADC in the BINUS ASO major context, it is probably PDE. "
                            "CATIA is CAD software. CAD means Computer-Aided Design. "
                            "In CATIA class, Sketcher, dimensional constraint, geometric constraint, Pad, Fillet, Chamfer, Hole, design intent, and manufacturability are important terms. "
                            "ロータリーエンジン means rotary engine. アペックスシール means apex seal. レシプロエンジン means reciprocating engine. "
                            "If Japanese sounds like ネウス大学, ビーナス大学, or ビナス大学, "
                            "it is probably ビヌス大学 / BINUS University."
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
                        # Clear worker-side accumulated text so old captions do not return.
                        final_original = ""
                        final_translation = ""
                        last_token_time = time.time()
                        result_queue.put({"type": "cleared"})

                    elif isinstance(command, dict) and command.get("type") == "set_base_caption":
                        # After AI correction is applied, do NOT clear the text.
                        # Use the corrected text as the new worker base, so
                        # the next live tokens continue from the fixed caption.
                        final_original = command.get("original", "") or final_original
                        final_translation = command.get("translation", "") or final_translation
                        last_token_time = time.time()
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
    "Japanese → English live captions using Gemini 3.5 Live Translate, "
    "with Gemini 3.1 Flash-Lite as the optional helper/correction AI."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Settings")

    translation_engine = ENGINE_GEMINI_LIVE
    st.info("Translation engine: Gemini 3.5 Live Translate")

    domain_mode = st.selectbox(
        "Technical domain",
        ["auto", "automotive", "cad", "product design"],
        index=0,
        help="For CATIA classes, choose 'cad' or 'product design' so CATIA is not misheard as 勝ち方 / way to win.",
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

    llm_model_name = st.selectbox(
        "LLM model",
        [
            LLM_MODEL_DEFAULT,
            LLM_MODEL_BACKUP,
        ],
        index=0,
    )

    llm_budget_mode = st.selectbox(
        "AI helper budget mode",
        list(LLM_BUDGET_MODES.keys()),
        index=1,
    )

    selected_budget = LLM_BUDGET_MODES[llm_budget_mode]
    llm_hint_interval = float(selected_budget["interval"])
    llm_min_context_chars = int(selected_budget["min_chars"])
    llm_session_limit = int(selected_budget["session_limit"])

    st.caption(selected_budget["description"])
    st.caption(
        f"Helper interval: {int(llm_hint_interval)} sec | "
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
    "llm_main_idea": "",
    "llm_say_it_simply": "",
    "llm_corrected_japanese_original": "",
    "llm_corrected_english_caption": "",
    "llm_corrected_source_text": "",
    "llm_key_terms": [],
    "llm_corrections": [],
    "llm_last_call_time": 0.0,
    "llm_last_source_text": "",
    "llm_context_chunks": [],

    # Restart reliability
    "restart_wait_until": 0.0,
    "last_worker_start_time": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_live_runtime_state(clear_captions=True, reset_budget=False):
    """Reset UI/worker state safely before a fresh Gemini Live session."""
    if clear_captions:
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

    st.session_state.llm_context_chunks = []
    st.session_state.llm_main_idea = ""
    st.session_state.llm_say_it_simply = ""
    st.session_state.llm_corrected_japanese_original = ""
    st.session_state.llm_corrected_english_caption = ""
    st.session_state.llm_corrected_source_text = ""
    st.session_state.llm_key_terms = []
    st.session_state.llm_corrections = []
    st.session_state.llm_error = ""
    st.session_state.llm_last_source_text = ""
    st.session_state.llm_running = False

    if reset_budget:
        st.session_state.llm_calls_this_session = 0
        st.session_state.llm_budget_reached = False


def reset_worker_channels():
    """Drop stale queue messages from old worker sessions."""
    st.session_state.soniox_result_queue = queue.Queue()
    st.session_state.soniox_control_queue = queue.Queue()
    st.session_state.soniox_stop_event = threading.Event()
    st.session_state.soniox_thread = None


def stop_current_session_for_restart():
    """Stop current Gemini/WebRTC session and prepare for clean restart."""
    try:
        st.session_state.soniox_stop_event.set()
    except Exception:
        pass

    st.session_state.app_active = False
    st.session_state.pending_start_translation = False
    st.session_state.soniox_running = False
    st.session_state.llm_running = False

    # Force a brand-new WebRTC component and give browser time to release mic.
    st.session_state.mic_instance_id += 1
    st.session_state.restart_wait_until = time.time() + 1.0

    # Replace queues so old "stopped"/"error" messages cannot affect next start.
    reset_worker_channels()


if st.session_state.current_engine and st.session_state.current_engine != translation_engine:
    stop_current_session_for_restart()
    st.session_state.current_engine = translation_engine
    st.rerun()

if not st.session_state.current_engine:
    st.session_state.current_engine = ENGINE_GEMINI_LIVE


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

api_key = None  # Soniox disabled in pure Gemini version.
gemini_api_key = safe_get_secret_or_env("GEMINI_API_KEY")

if not gemini_api_key:
    st.error(
        "GEMINI_API_KEY is not set.\n\n"
        "Gemini Mode needs Gemini 3.5 Live Translate. For Streamlit Cloud, add this in Secrets:\n\n"
        'GEMINI_API_KEY = "your_gemini_api_key_here"'
    )
    st.stop()

if use_llm_hints and not gemini_api_key:
    st.warning(
        "GEMINI_API_KEY is not set. Helper AI is disabled until you add it."
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
    key=f"gemini-live-caption-mic-{st.session_state.mic_instance_id}",
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

# If a worker died silently, do not leave the UI stuck in running state.
if (
    st.session_state.soniox_running
    and st.session_state.soniox_thread is not None
    and not st.session_state.soniox_thread.is_alive()
):
    st.session_state.soniox_running = False


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
        stop_current_session_for_restart()
        st.rerun()

    else:
        # Fresh start: remove stale worker messages and force a new mic object.
        reset_worker_channels()
        reset_live_runtime_state(clear_captions=True, reset_budget=False)

        st.session_state.app_active = True
        st.session_state.pending_start_translation = True
        st.session_state.soniox_running = False
        st.session_state.soniox_error = ""
        st.session_state.mic_instance_id += 1

        st.rerun()

if clear_clicked:
    reset_live_runtime_state(clear_captions=True, reset_budget=True)

    if st.session_state.soniox_running:
        st.session_state.soniox_control_queue.put("clear")
        st.session_state.debug_messages.append("Clear requested.")
        st.session_state.debug_messages = st.session_state.debug_messages[-MAX_DEBUG_MESSAGES:]


# ============================================================
# Auto-start Gemini after WebRTC mic is ready
# ============================================================

if (
    st.session_state.pending_start_translation
    and st.session_state.app_active
    and not st.session_state.soniox_running
    and webrtc_ctx.audio_processor
    and time.time() >= float(st.session_state.restart_wait_until)
):
    reset_worker_channels()
    reset_live_runtime_state(clear_captions=True, reset_budget=True)
    st.session_state.debug_messages = []
    st.session_state.llm_last_call_time = 0.0
    st.session_state.last_worker_start_time = time.time()

    processor = webrtc_ctx.audio_processor

    st.session_state.soniox_running = True
    st.session_state.pending_start_translation = False

    worker_target = gemini_live_translate_worker
    worker_args = (
        processor.audio_queue,
        st.session_state.soniox_result_queue,
        st.session_state.soniox_stop_event,
        st.session_state.soniox_control_queue,
        gemini_api_key,
        "en",
        float(reset_seconds),
    )

    st.session_state.soniox_thread = threading.Thread(
        target=worker_target,
        args=worker_args,
        daemon=True,
    )

    st.session_state.soniox_thread.start()


# If Start was clicked but WebRTC has not created the audio processor yet,
# keep rerunning briefly. Without this, Stop -> Start can get stuck waiting
# for a mic object that appears after the current rerun.
if (
    st.session_state.pending_start_translation
    and st.session_state.app_active
    and not st.session_state.soniox_running
):
    if time.time() < float(st.session_state.restart_wait_until):
        st.info("Releasing microphone. Starting again in a moment...")
    elif not webrtc_ctx.audio_processor:
        st.info("Waiting for microphone to become ready...")

    time.sleep(0.35)
    st.rerun()


# ============================================================
# Pull Gemini results into UI state
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

        if original or translation:
            st.session_state.live_token_version += 1
            prepare_next_ai_check_after_new_live_text()

        if st.session_state.pending_visual_reset and (original or translation):
            st.session_state.live_original = ""
            st.session_state.live_translation = ""
            st.session_state.caption_history = []
            st.session_state.llm_corrected_japanese_original = ""
            st.session_state.llm_corrected_english_caption = ""
            st.session_state.llm_corrected_source_text = ""
            st.session_state.llm_key_terms = []
            st.session_state.llm_corrections = []
            st.session_state.caption_stage = "raw_started"
            st.session_state.last_helper_fix_time = ""
            st.session_state.last_ai_check_time = ""
            st.session_state.correction_status = "pending"
            st.session_state.pending_visual_reset = False

        if original:
            st.session_state.live_original = original
            st.session_state.caption_stage = "raw_japanese"
            st.session_state.last_raw_input_time = time.strftime("%H:%M:%S")

            if not st.session_state.live_translation:
                st.session_state.correction_status = "waiting_for_english"

        if translation:
            st.session_state.live_translation = translation
            st.session_state.caption_stage = "raw_english"
            st.session_state.last_raw_translation_time = time.strftime("%H:%M:%S")

            if use_llm_hints and gemini_api_key:
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
        if st.session_state.live_translation and use_llm_hints and gemini_api_key:
            st.session_state.correction_status = "pending"

    elif item_type == "cleared":
        st.session_state.live_original = ""
        st.session_state.live_translation = ""
        st.session_state.caption_history = []
        st.session_state.last_update_time = ""
        st.session_state.llm_context_chunks = []
        st.session_state.llm_main_idea = ""
        st.session_state.llm_say_it_simply = ""
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
        st.session_state.soniox_error = item.get("message", "")
        st.session_state.soniox_running = False
        st.session_state.app_active = False
        st.session_state.pending_start_translation = False
        st.session_state.mic_instance_id += 1

    elif item_type == "stopped":
        # Ignore stale stopped messages while a new session is starting/running.
        if not st.session_state.app_active and not st.session_state.pending_start_translation:
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
        st.session_state.llm_corrected_japanese_original = item.get("corrected_japanese_original", "")
        st.session_state.llm_corrected_english_caption = item.get("corrected_english_caption", "")
        st.session_state.llm_corrected_source_text = item.get("source_text", "")
        st.session_state.llm_key_terms = item.get("key_terms", [])
        st.session_state.llm_corrections = item.get("corrections", [])
        st.session_state.llm_error = ""
        st.session_state.llm_running = False
        st.session_state.llm_calls_this_session += 1

        has_ai_update = (
            bool(st.session_state.llm_corrected_japanese_original)
            or bool(st.session_state.llm_corrected_english_caption)
            or bool(st.session_state.llm_key_terms)
            or bool(st.session_state.llm_corrections)
        )

        if has_ai_update:
            st.session_state.caption_stage = "ai_corrected"
            st.session_state.correction_status = "applied"

            # Important:
            # Keep the AI-corrected text visible AND continue from it.
            # The worker's internal base is replaced with the corrected text,
            # so the next live tokens append to the fixed caption instead of
            # returning old text or starting from empty.
            corrected_base_original = (
                st.session_state.llm_corrected_japanese_original
                or st.session_state.live_original
            )
            corrected_base_translation = (
                st.session_state.llm_corrected_english_caption
                or st.session_state.live_translation
            )

            corrected_base_original = light_original_cleanup(
                apply_llm_corrections(
                    corrected_base_original,
                    st.session_state.llm_corrections,
                )
            )
            corrected_base_translation = light_caption_cleanup(
                apply_llm_corrections(
                    corrected_base_translation,
                    st.session_state.llm_corrections,
                )
            )
            corrected_base_original, corrected_base_translation = light_domain_context_cleanup(
                corrected_base_original,
                corrected_base_translation,
                domain_mode,
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
            st.session_state.correction_status = "no_change"

        st.session_state.last_helper_fix_time = time.strftime("%H:%M:%S")

    elif item_type == "llm_error":
        st.session_state.llm_error = item.get("message", "")
        st.session_state.llm_running = False
        st.session_state.llm_calls_this_session += 1
        st.session_state.correction_status = "error"


# ============================================================
# Start LLM hint worker
# ============================================================

if use_llm_hints and gemini_api_key:
    source_text = build_llm_context(
        st.session_state.llm_context_chunks,
        st.session_state.live_original,
        st.session_state.live_translation,
    )

    enough_text = len(source_text) >= int(llm_min_context_chars)
    changed_text = source_text != st.session_state.llm_last_source_text
    interval_ready = (
        time.time() - float(st.session_state.llm_last_call_time)
        >= float(llm_hint_interval)
    )

    translated_text_ready = bool(st.session_state.live_translation.strip())
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

if st.session_state.soniox_running:
    st.success("Gemini 3.5 Live Translate running.")
elif st.session_state.app_active:
    st.info("Starting Gemini Live Translate...")
else:
    st.info("Live translation stopped.")

if st.session_state.soniox_error:
    st.error(st.session_state.soniox_error)

if use_llm_hints and st.session_state.llm_error:
    st.warning(f"LLM error: {st.session_state.llm_error}")

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
    and st.session_state.llm_corrected_source_text
    and source_text_matches_for_correction(
        current_source_for_display,
        st.session_state.llm_corrected_source_text,
    )
):
    if st.session_state.llm_corrected_japanese_original:
        corrected_original = apply_llm_corrections(
            st.session_state.llm_corrected_japanese_original,
            st.session_state.llm_corrections,
        )

    if st.session_state.llm_corrected_english_caption:
        corrected_translation = apply_llm_corrections(
            st.session_state.llm_corrected_english_caption,
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

if st.session_state.live_original and not display_english:
    display_english = "Waiting for English translation..."

source_is_corrected = (
    use_llm_hints
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
elif st.session_state.live_translation:
    jp_status_text = "Live Japanese"
    en_status_text = "Live English translation"
else:
    jp_status_text = "Waiting for Japanese speech..."
    en_status_text = "Waiting for English translation..."


if not use_llm_hints:
    correction_status_text = "AI correction off"
elif not gemini_api_key:
    correction_status_text = "AI correction unavailable: GEMINI_API_KEY missing"
elif st.session_state.llm_error:
    correction_status_text = f"AI correction error: {st.session_state.llm_error}"
elif st.session_state.llm_running:
    correction_status_text = "AI correction checking..."
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
    en_status_text = "Live English / AI correction pending"


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

    if st.session_state.llm_running:
        llm_terms_text = "AI checking key terms and caption corrections..."
    elif st.session_state.live_translation and not st.session_state.llm_key_terms and st.session_state.correction_status in ["pending", "checking"]:
        llm_terms_text = "AI correction pending..."
    elif st.session_state.llm_key_terms:
        llm_terms_lines = []

        for item in st.session_state.llm_key_terms[:8]:
            term = str(item.get("term", "")).strip()
            meaning = str(item.get("meaning", "")).strip()

            term = apply_llm_corrections(term, st.session_state.llm_corrections)
            meaning = apply_llm_corrections(meaning, st.session_state.llm_corrections)

            line = normalize_key_term_line(term, meaning)

            if line and line not in llm_terms_lines:
                llm_terms_lines.append(line)

            if len(llm_terms_lines) >= 5:
                break

        if llm_terms_lines:
            llm_terms_text = "\n".join(llm_terms_lines)
        else:
            llm_terms_text = "No Japanese technical key terms yet."
    else:
        llm_terms_text = "No LLM key terms yet."

else:
    simple_text = ""
    llm_terms_text = ""

safe_original = html.escape(display_japanese)
safe_caption_text = html.escape(display_english)
safe_llm_terms = html.escape(llm_terms_text)
safe_jp_status = html.escape(jp_status_text)
safe_en_status = html.escape(en_status_text)
safe_correction_status = html.escape(correction_status_text)

llm_html = ""

if use_llm_hints:
    llm_html = f"""
    <div>
        <div class="caption-label">Correction Status</div>
        <div class="correction-status-box">{safe_correction_status}</div>
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
        st.write("Engine:")
        st.code(translation_engine)

        st.write("Caption stage:")
        st.code(st.session_state.caption_stage)

        st.write("Live token version:")
        st.code(str(st.session_state.live_token_version))

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
            f"main_idea={st.session_state.llm_main_idea}\n"
            f"say_it_simply={st.session_state.llm_say_it_simply}\n"
            f"corrected_japanese_original={st.session_state.llm_corrected_japanese_original}\n"
            f"corrected_english_caption={st.session_state.llm_corrected_english_caption}\n"
            f"source_match_for_correction={source_text_matches_for_correction(current_source_for_display, st.session_state.llm_corrected_source_text)}\n"
            f"corrections={st.session_state.llm_corrections}\n"
            f"error={st.session_state.llm_error}"
        )

        st.write("Gemini audio chunk bytes:")
        st.code(str(GEMINI_LIVE_AUDIO_CHUNK_BYTES))

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
    max-height: 80px;
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
    max-height: 190px;
    overflow: hidden;
    white-space: pre-wrap;
    border: 1px solid #D1D5DB;
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
    min-height: 130px;
    max-height: 280px;
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
        min-height: 70px;
        max-height: 150px;
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
        line-height: 1.25;
        padding: 12px;
        min-height: 115px;
        max-height: 235px;
    }}
}}
</style>

<div class="caption-wrapper">
    <div>
        <div class="caption-label">Japanese Original <span class="caption-status">{safe_jp_status}</span></div>
        <div class="jp-caption-box">{safe_original}</div>
    </div>

    {llm_html}

    <div>
        <div class="caption-label">English Caption <span class="caption-status">{safe_en_status}</span></div>
        <div class="en-caption-box">{safe_caption_text}</div>
    </div>
</div>
"""

st.html(caption_html)


# ============================================================
# Live refresh
# ============================================================

if st.session_state.app_active or st.session_state.soniox_running:
    time.sleep(0.2)
    st.rerun()