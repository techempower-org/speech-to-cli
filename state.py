"""Shared mutable state for the Azure Speech MCP server.

Every module does `import state` and accesses reassignable globals as `state.X`.
Containers (CONFIG dict, _active_procs list) and Events/Locks can also be
imported by name since they are mutated in-place, not reassigned.
"""

import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import requests

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
    _whisper_model = None
except ImportError:
    HAS_WHISPER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULTS_PATH = os.path.expanduser("~/.config/speech-to-cli/config.json")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHIME_PATH = os.path.join(_SCRIPT_DIR, "chime_ready.wav")
CHIME_PROCESSING = os.path.join(_SCRIPT_DIR, "chime_processing.wav")
CHIME_SPEAK = os.path.join(_SCRIPT_DIR, "chime_speak.wav")
CHIME_DONE = os.path.join(_SCRIPT_DIR, "chime_done.wav")
CHIME_HUM = os.path.join(_SCRIPT_DIR, "chime_hum.wav")
CHIME_PAUSE = os.path.join(_SCRIPT_DIR, "chime_pause.wav")
CHIME_RESUME = os.path.join(_SCRIPT_DIR, "chime_resume.wav")

# Audio settings
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000  # 960 bytes per 30ms frame

# Echo cancellation (PipeWire)
EC_SOURCE = "echo_cancel_source"  # cleaned mic (AEC output)
EC_SINK = "echo_cancel_sink"      # TTS audio routes here so AEC can subtract it

# VAD + energy gate settings (these may be temporarily overridden by stt())
SILENCE_TIMEOUT = 3.0
NO_SPEECH_TIMEOUT = 7.0
MIN_SPEECH_DURATION = 0.15
# Live partials on the Wyoming VAD route (gnome-speaks live typing/subtitles):
# the LAN recognizer has no streaming protocol (wyoming-onnx-asr reports
# supports_transcript_streaming=False), so the utterance so far is re-sent
# every PARTIAL_INTERVAL_MS of speech and the answer is a hypothesis. Measured
# 2026-09-12 on Parakeet TDT 0.6B: 2.8 s of speech -> 0.15 s, 5.6 s -> 0.26 s;
# some short clips never answer (1.4 s hung past 20 s) while other requests
# are still served, hence the per-request timeout -- a hung partial is dropped,
# never the final transcription, which uses its own connection.
PARTIAL_INTERVAL_MS = 400
PARTIAL_TIMEOUT = 2.5
VAD_AGGRESSIVENESS = 3
ENERGY_CALIBRATION_FRAMES = 5
ENERGY_THRESHOLD_MULTIPLIER = 2.5
NOISE_CACHE_TTL = 120.0  # reuse ambient threshold for 2min (re-calibrates on fresh listen)

WS_IDLE_TIMEOUT = 540  # Azure closes at ~600s

_MAX_TTS_CHARS = 5000
_SSML_SAFE_RE = re.compile(r'^[a-zA-Z0-9\-_.:+%() ]+$')

# ---------------------------------------------------------------------------
# Mutable globals
# ---------------------------------------------------------------------------

_has_echo_cancel = None  # lazy-detected

_cached_noise_threshold = None
_cached_noise_time = 0.0
_http_session = None
_persistent_ws = None
_persistent_ws_time = 0.0
_hum_proc = None

# Cancellation support
_cancel_event = threading.Event()
_active_request_id = None
_active_procs = []
_active_procs_lock = threading.Lock()
_pause_event = threading.Event()

# Pre-warmed recorder
_prewarmed_rec = None
_prewarmed_rec_lock = threading.Lock()
_rec_idle_timer = None  # threading.Timer that kills idle recorder
_warmup_pending = False
_warmup_lock = threading.Lock()

# Pre-warmed player
_prewarmed_player = None
_prewarmed_player_rate = 0
_wyoming_tts_rate = 0  # rate of the last Wyoming audio-start; 0 = none seen yet (audio._prewarm_rate)
_prewarmed_player_lock = threading.Lock()
_player_idle_timer = None  # threading.Timer that reaps an idle prewarmed player

# TTY / terminal width caching
_tty_fd = None
_cached_tty_width = None
_cached_tty_width_time = 0.0

# ISO timestamp caching
_cached_iso_ts = ""
_cached_iso_ts_time = 0.0


def _get_iso_timestamp():
    """Cached ISO timestamp, refreshed every 500ms. Avoids 1000+ strftime calls per recording."""
    import state
    now = time.time()
    if now - state._cached_iso_ts_time > 0.5:
        state._cached_iso_ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(now))
        state._cached_iso_ts_time = now
    return state._cached_iso_ts


# STT lock
_stt_lock = threading.Lock()

# TTS timing
_last_tts_end = 0.0

# Audio detection
_half_duplex_setting = None  # set after load_config()
_last_detected_sink_id = None
_auto_detect_lock = threading.Lock()

# Agent hint tracking
_consecutive_no_speech = 0
_consecutive_short_response = 0  # responses under 5 words

# Stdout / request queue
_stdout_lock = threading.Lock()
_request_queue = []
_request_cond = threading.Condition()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _normalise_phrase_list(value):
    """phrase_list is a LIST of phrases. A prefs entry row stores it as ONE
    comma-separated string; iterating that would hand Azure one phrase per
    CHARACTER (stt.py builds the phrase list straight from CONFIG). Normalise
    here, at the single point every reader passes through."""
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(p).strip() for p in value if str(p).strip()]
    return []


def load_config():
    cfg = {}
    if os.path.exists(DEFAULTS_PATH):
        with open(DEFAULTS_PATH) as f:
            cfg = json.load(f)
    return {
        "key": os.environ.get("AZURE_SPEECH_KEY") or cfg.get("key"),
        "region": os.environ.get("AZURE_SPEECH_REGION") or cfg.get("region", "westus2"),
        "voice": os.environ.get("AZURE_SPEECH_VOICE") or cfg.get("voice", "en-US-Ava:DragonHDLatestNeural"),
        "fast_voice": cfg.get("fast_voice", "en-US-AvaNeural"),
        # Separate TTS region/key for DragonHD voices (only available in select regions)
        "tts_region": cfg.get("tts_region", None),  # None = use main region
        "tts_key": cfg.get("tts_key", None),          # None = use main key
        # Wyoming offline fallback (LAN STT/TTS when Azure is unreachable)
        "wyoming_host": cfg.get("wyoming_host", ""),  # "" = feature off
        "wyoming_tts_port": cfg.get("wyoming_tts_port", 10200),
        "wyoming_stt_port": cfg.get("wyoming_stt_port", 10300),
        "wyoming_tts_voice": cfg.get("wyoming_tts_voice", "en_GB-cori-high"),
        "wyoming_wake_port": cfg.get("wyoming_wake_port", 10400),
        "llm_thinking": cfg.get("llm_thinking", False),  # reasoning models: think before voice replies
        # Spiel speech provider (org.freedesktop.Speech.Provider)
        "spiel_provider": cfg.get("spiel_provider", False),
        "spiel_voices": cfg.get("spiel_voices", ["en_GB-cori-high"]),
        "spiel_expose_azure": cfg.get("spiel_expose_azure", False),
        "wake_word": cfg.get("wake_word", False),           # wake watcher on/off
        "wake_word_model": cfg.get("wake_word_model", ""),  # openwakeword model name
        # Audio device settings
        "player": cfg.get("player", "auto"),
        "recorder": cfg.get("recorder", "auto"),
        "mic_source": cfg.get("mic_source", None),
        "speaker_sink": cfg.get("speaker_sink", None),
        "silence_timeout": cfg.get("silence_timeout", SILENCE_TIMEOUT),
        "talk_silence_timeout": cfg.get("talk_silence_timeout", 4.0),
        "no_speech_timeout": cfg.get("no_speech_timeout", NO_SPEECH_TIMEOUT),
        "energy_multiplier": cfg.get("energy_multiplier", ENERGY_THRESHOLD_MULTIPLIER),
        # UI settings
        "chime_ready": cfg.get("chime_ready", True),
        "chime_processing": cfg.get("chime_processing", False),
        "chime_speak": cfg.get("chime_speak", False),
        "chime_done": cfg.get("chime_done", False),
        "chime_hum": cfg.get("chime_hum", False),
        "visual_indicator": cfg.get("visual_indicator", True),
        "live_subtitles": cfg.get("live_subtitles", True),
        "subtitle_color_user": cfg.get("subtitle_color_user", "light_green"),
        "subtitle_color_tts": cfg.get("subtitle_color_tts", "amber"),
        "vu_meter": cfg.get("vu_meter", True),
        "barge_in_frames": cfg.get("barge_in_frames", 3),
        "barge_in_silence": cfg.get("barge_in_silence", 1.0),
        "chime_barge_in": cfg.get("chime_barge_in", True),
        "enable_pause": cfg.get("enable_pause", True),
        "end_word": cfg.get("end_word", "over"),
        "max_record_seconds": cfg.get("max_record_seconds", 120),
        "enable_echo_cancel": cfg.get("enable_echo_cancel", False),
        # TTS prosody — passed through to tts() by consumers (gnome-speaks#17)
        "speed": cfg.get("speed", 1.0),
        "pitch": cfg.get("pitch", "default"),
        "volume": cfg.get("volume", "default"),
        # Spoken-word ledger kept by gnome-speaks (chronicle.jsonl + respeak)
        "chronicle": cfg.get("chronicle", True),
        # Wake-word dictation only types into fields IBus confirms non-secure
        "wake_word_secure_gate": cfg.get("wake_word_secure_gate", False),
        # User-authored word corrections — read by gnome-speaks
        # apply_auto_corrections(); was unwhitelisted, so every user's
        # corrections silently never applied (prefs-audit finding #1).
        "auto_corrections": cfg.get("auto_corrections", {}),
        "conversation_silence_timeout": cfg.get("conversation_silence_timeout", 4.0),
        "enable_barge_in": cfg.get("enable_barge_in", False),
        "debug": cfg.get("debug", False),
        "half_duplex": cfg.get("half_duplex", "auto"),
        # LLM / conversation mode settings (used by gnome-speaks)
        "llm_provider": cfg.get("llm_provider", "anthropic"),
        "llm_model": cfg.get("llm_model", "claude-opus-4.6"),  # dotted canonical (MODEL_MAP form)
        "llm_api_key": cfg.get("llm_api_key", ""),
        "llm_system_prompt": cfg.get("llm_system_prompt", ""),
        "conversation_mode": cfg.get("conversation_mode", False),
        "dictation_mode": cfg.get("dictation_mode", True),
        "skip_final_paste": cfg.get("skip_final_paste", True),
        "terminal_mode": cfg.get("terminal_mode", False),
        # Text-injection backend used by gnome-speaks: "ydotool" | "ibus" |
        # "auto". Read by the Python side, so it MUST live in this whitelist —
        # an unlisted key is silently dropped and the feature reads its
        # default forever.
        "injection_method": cfg.get("injection_method", "ydotool"),
        # Which speech provider is PRIMARY: "azure" (default) | "local" (the
        # Wyoming server, Azure only as fallback). Read by wyoming.prefer_local()
        # on the Python side, so it MUST be whitelisted here too.
        "speech_backend": cfg.get("speech_backend", "azure"),
        # Azure STT phrase hints (list, or the comma-separated string a prefs
        # entry row stores -- gnome-speaks normalises). Read on the Python
        # side, so it must be whitelisted or the prefs row is a no-op.
        "phrase_list": _normalise_phrase_list(cfg.get("phrase_list", [])),
        # STT recognition language (stt.py, and gnome-speaks get/set_language)
        # and the spoken-punctuation rewriter (gnome-speaks
        # apply_voice_commands). Both are read on the Python side, so both
        # MUST be whitelisted: unlisted, they reached CONFIG only through
        # gnome-speaks' _SYNC_FLAGS side door, which a wake-word-first
        # session (start_listening(quick=True)) never opens (gnome-speaks#127).
        "language": cfg.get("language", "en-US"),
        "voice_commands": cfg.get("voice_commands", True),
        "continuous_dictation": cfg.get("continuous_dictation", False),
        "loop_silence_timeout": cfg.get("loop_silence_timeout", 1.2),
        # Home Assistant token source for gnome-speaks' spellbook `assist`
        # action (gnome-speaks#129): the vault item name for `bw get password`
        # and a token cache file. Read on the Python side by _get_ha_token(),
        # so both MUST be whitelisted here; both default to "" = not used, so
        # a stock install never shells out to bw or opens a personal path.
        "ha_token_item": cfg.get("ha_token_item", ""),
        "ha_token_cache": cfg.get("ha_token_cache", ""),
        "read_notifications": cfg.get("read_notifications", False),
        # Audio file saving
        "save_audio_dir": cfg.get("save_audio_dir", None),  # auto-save all TTS to this dir
    }


def load_config_standalone(need_voice=False):
    """Load config for standalone scripts (speech.py, tts.py, voice_chat.py).

    Returns (key, region) or (key, region, voice) if need_voice=True.
    Exits with an error message if AZURE_SPEECH_KEY is not set.
    """
    cfg = load_config()
    if not cfg.get("key"):
        print("Error: No Azure Speech API key found.", file=sys.stderr)
        print("Set AZURE_SPEECH_KEY or create ~/.config/speech-to-cli/config.json",
              file=sys.stderr)
        sys.exit(1)
    if need_voice:
        return cfg["key"], cfg["region"], cfg["voice"]
    return cfg["key"], cfg["region"]


CONFIG = load_config()
_half_duplex_setting = CONFIG.get("half_duplex", "auto")


# ---------------------------------------------------------------------------
# Small cross-cutting helpers
# ---------------------------------------------------------------------------

def is_cancelled():
    """Check if the current operation has been cancelled."""
    return _cancel_event.is_set()


def register_proc(proc):
    """Track a subprocess so it can be killed on cancellation."""
    with _active_procs_lock:
        _active_procs.append(proc)


def unregister_proc(proc):
    """Stop tracking a subprocess."""
    with _active_procs_lock:
        try:
            _active_procs.remove(proc)
        except ValueError:
            pass


def cancel_active():
    """Kill all tracked subprocesses and signal cancellation."""
    _cancel_event.set()
    _pause_event.clear()  # unpause first so loops can exit
    with _active_procs_lock:
        for proc in _active_procs:
            try:
                proc.terminate()
            except Exception:
                pass


def pause_active():
    """Pause playback by setting the pause event (data feed stops, no SIGSTOP)."""
    if not CONFIG.get("enable_pause", True):
        return
    _pause_event.set()


def resume_active():
    """Resume playback by clearing the pause event."""
    if not CONFIG.get("enable_pause", True):
        return
    _pause_event.clear()


def get_http_session():
    """Reuse HTTP session for connection pooling (saves ~150ms per TTS call)."""
    import state
    if state._http_session is None:
        state._http_session = requests.Session()
    return state._http_session


def send_progress(token, progress, total=None, description=None):
    """Send an MCP progress notification if a token is provided."""
    if token is None:
        return
    msg = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": token,
            "progress": progress,
        }
    }
    if total is not None:
        msg["params"]["total"] = total
    if description:
        # Claude Code renders ANSI in 'description', Gemini CLI uses plain 'message'
        msg["params"]["description"] = description
        msg["params"]["message"] = re.sub(r'\033\[[0-9;]*m', '', description)
    line = json.dumps(msg) + "\n"
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()
    if CONFIG.get("debug"):
        with open("/tmp/speech-debug.log", "a") as _f:
            _f.write(f"PROGRESS: {line}")
