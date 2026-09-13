"""Audio I/O, device detection, chimes, UI helpers, VAD, and pre-warming.

Import pattern: `import state` at top. Reassignable globals accessed as `state.X`.
"""

import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time

import state
import wyoming
from state import (CONFIG, _cancel_event, _pause_event,
                   is_cancelled, register_proc, unregister_proc, get_http_session)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Pre-computed struct format for standard frame size (480 samples = 960 bytes)
_RMS_FMT = f'<{state.FRAME_BYTES // 2}h'


# ---------------------------------------------------------------------------
# Echo cancellation detection
# ---------------------------------------------------------------------------

def has_echo_cancel():
    """Check if PipeWire echo cancellation nodes exist (and enabled in config)."""
    if not CONFIG.get("enable_echo_cancel", True):
        return False
    if state._has_echo_cancel is not None:
        return state._has_echo_cancel
    try:
        result = subprocess.run(
            ["pw-cli", "list-objects"],
            capture_output=True, text=True, timeout=3,
        )
        state._has_echo_cancel = state.EC_SOURCE in result.stdout
    except Exception:
        state._has_echo_cancel = False
    return state._has_echo_cancel


# ---------------------------------------------------------------------------
# Audio output detection
# ---------------------------------------------------------------------------

def _get_default_sink_id():
    """Parse wpctl status to find the default sink ID (~12ms)."""
    try:
        status = subprocess.run(
            ["wpctl", "status"], capture_output=True, text=True, timeout=3)
        if status.returncode != 0:
            return None
    except Exception:
        return None
    in_sinks = False
    for line in status.stdout.splitlines():
        if "Sinks:" in line:
            in_sinks = True
            continue
        if in_sinks:
            stripped = line.strip()
            if stripped.startswith(("\u251c", "\u2514")) or not stripped:
                in_sinks = False
                continue
            if "*" in line:
                parts = line.split("*")[1].strip().split(".")
                return parts[0].strip()
    return None


def detect_audio_output(sink_id=None):
    """Detect default audio sink type via PipeWire.

    Returns (device_type, info_dict) where device_type is one of:
      'headphones' -- headset/headphone/earbuds (safe for full-duplex)
      'speakers'   -- built-in/HDMI/external speakers (needs half-duplex)
      'unknown'    -- detection failed
    """
    import json as _json
    if sink_id is None:
        sink_id = _get_default_sink_id()
    if not sink_id:
        return "unknown", {}

    try:
        dump = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=5)
        data = _json.loads(dump.stdout)
    except Exception:
        return "unknown", {}

    # Find the sink node
    sink_props = None
    for obj in data:
        if str(obj.get("id", "")) == sink_id:
            sink_props = obj.get("info", {}).get("props", {})
            break
    if not sink_props:
        return "unknown", {}

    node_name = sink_props.get("node.name", "")
    desc = sink_props.get("node.description", "")
    is_bluez = node_name.startswith("bluez_")

    # Get parent device form-factor
    device_id = sink_props.get("device.id")
    form_factor = ""
    device_bus = ""
    if device_id is not None:
        for obj in data:
            if obj.get("id") == device_id:
                dprops = obj.get("info", {}).get("props", {})
                form_factor = dprops.get("device.form-factor", "")
                device_bus = dprops.get("device.bus", "")
                break

    info = {
        "description": desc,
        "form_factor": form_factor,
        "bus": device_bus,
        "bluetooth": is_bluez,
    }

    if form_factor in ("headset", "headphone"):
        return "headphones", info
    if is_bluez:
        return "headphones", info
    if form_factor == "internal":
        return "speakers", info
    if "hdmi" in node_name.lower():
        return "speakers", info
    return "speakers", info


def _refresh_audio_detection():
    """Re-detect audio output if default sink changed. Non-blocking (~12ms).

    Only does full detection (~35ms) when the default sink ID changes.
    Called synchronously before talk/speak/converse.
    """
    if state._half_duplex_setting != "auto":
        return  # User forced a specific mode, don't override

    current_sink_id = _get_default_sink_id()
    if current_sink_id is None:
        return

    with state._auto_detect_lock:
        if current_sink_id == state._last_detected_sink_id:
            return  # Same sink, no change needed
        state._last_detected_sink_id = current_sink_id

    # Sink changed -- do full detection (~22ms for pw-dump, skip repeat wpctl)
    dev_type, dev_info = detect_audio_output(sink_id=current_sink_id)
    new_half_duplex = (dev_type == "speakers")
    old_half_duplex = CONFIG.get("half_duplex", False)
    CONFIG["half_duplex"] = new_half_duplex
    CONFIG["_detected_output"] = dev_type
    CONFIG["_detected_output_info"] = dev_info
    if new_half_duplex != old_half_duplex:
        try:
            _print_status(
                f"Audio: {dev_info.get('description', '?')} \u2192 {'half' if new_half_duplex else 'full'}-duplex",
                "93")  # Yellow
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def _build_player_cmd(tts_rate, target=None):
    """Build a player command list based on config. Returns list of fallback commands to try."""
    player = CONFIG.get("player", "auto")
    sink = target or CONFIG.get("speaker_sink")
    cmds = []

    if player in ("aplay", "auto"):
        cmd = ["aplay", "-f", "S16_LE", "-r", str(tts_rate), "-c", "1", "-t", "raw", "-q"]
        if sink:
            cmd = ["aplay", "-f", "S16_LE", "-r", str(tts_rate), "-c", "1", "-t", "raw",
                   "-D", f"pipewire:NODE={sink}", "-q"]
        cmds.append(cmd)
    if player in ("pw-play", "pw-cat"):
        cmd = ["pw-cat", "--playback", "--format", "s16", "--rate", str(tts_rate), "--channels", "1"]
        if sink:
            cmd += ["--target", sink]
        cmd.append("-")
        cmds.append(cmd)
    if player in ("ffplay", "auto"):
        cmds.append(["ffplay", "-f", "s16le", "-ar", str(tts_rate), "-ac", "1",
                      "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"])

    if not cmds:
        cmds.append(["aplay", "-f", "S16_LE", "-r", str(tts_rate), "-c", "1", "-t", "raw", "-q"])

    return cmds


def _build_rec_cmd(max_seconds=None, raw=True):
    """Build a recorder command list based on config. Returns a single command."""
    recorder = CONFIG.get("recorder", "auto")
    mic = CONFIG.get("mic_source")

    if recorder == "pw-record" or (recorder == "auto" and shutil.which("pw-record")):
        cmd = ["pw-record", "--format", "s16", "--rate", "16000", "--channels", "1"]
        if mic:
            cmd += ["--target", mic]
        cmd.append("-")
        return cmd

    # arecord fallback
    device = mic or "default"
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"]
    if max_seconds:
        cmd += ["-d", str(max_seconds)]
    return cmd


# ---------------------------------------------------------------------------
# Chime generation and playback
# ---------------------------------------------------------------------------

CHIME_RATE = 44100

# Struck-glass partial structure. Real bells and glass have INHARMONIC
# partials — ratios that aren't whole numbers — and that's the whole reason
# they read as a physical object rather than a synthesiser. Higher partials
# also decay faster on a real struck body, which `decay` encodes.
#
#            ratio   amp    decay (x faster than fundamental)
_PARTIALS = [(1.000, 1.000, 1.00),
             (2.000, 0.460, 1.45),
             (3.010, 0.250, 1.95),
             (4.170, 0.130, 2.60),
             (5.430, 0.062, 3.40),
             (6.790, 0.030, 4.30)]

# Warm F-pentatonic. Every pair is consonant, so the tones sound like one
# family instead of seven unrelated beeps.
F3, F4, G4, A4, C5, D5 = 174.61, 349.23, 392.00, 440.00, 523.25, 587.33


def _generate_chimes():
    """Generate all status chime WAV files if they don't exist.

    Replaces the original pure-sine generator, which sounded thin and clicky
    for two independent reasons:

      1. A single `math.sin` per note — no partials, so no timbre.
      2. Notes were hard-concatenated with a trapezoidal 20ms attack/release,
         so every note boundary was a step discontinuity. That broadband click
         was the actual harshness (a zero-crossing analysis of the old files
         reads ~7.4kHz dominant, which is the splice artefact, not the tone).

    This version uses struck-glass synthesis: inharmonic partials, exponential
    decay with a fast attack, notes that OVERLAP and ring into each other
    rather than butting together, and a faintly detuned twin voice for shimmer.
    44.1kHz so the upper partials aren't aliased.
    """
    import wave

    def _render(notes, tail=0.5):
        """notes: [(freq, start_s, amp, decay_s)] -> float list, summed.

        Notes are placed on a shared timeline by START TIME, so they overlap
        and decay through one another. Nothing is ever spliced.
        """
        total = max(s + d for _, s, _, d in notes) + tail
        n = int(CHIME_RATE * total)
        buf = [0.0] * n
        for freq, start, amp, decay in notes:
            i0 = int(start * CHIME_RATE)
            # Two voices detuned by 0.15% — slow beating reads as "glassy"
            # rather than as two notes.
            for detune, dweight in ((1.0, 1.0), (1.0015, 0.55)):
                for ratio, pamp, pdecay in _PARTIALS:
                    f = freq * ratio * detune
                    if f > CHIME_RATE * 0.45:      # keep well under Nyquist
                        continue
                    tau = decay / pdecay
                    a = amp * pamp * dweight
                    w = 2.0 * math.pi * f
                    for i in range(i0, n):
                        t = (i - i0) / CHIME_RATE
                        e = math.exp(-t / tau)
                        if e < 0.0005:             # stop when inaudible
                            break
                        # 4ms raised-cosine attack: fast enough to read as a
                        # strike, smooth enough to add no click.
                        if t < 0.004:
                            e *= 0.5 - 0.5 * math.cos(math.pi * t / 0.004)
                        buf[i] += a * e * math.sin(w * t)
        return buf

    def _write(path, buf, peak=0.42):
        if os.path.exists(path):
            return
        m = max((abs(v) for v in buf), default=0.0)
        g = (peak / m) if m > 0 else 0.0
        out = []
        for v in buf:
            x = v * g
            # Gentle tanh-ish soft clip; cannot ever hard-clip to a buzz.
            x = math.tanh(x * 1.35) / 1.35
            out.append(int(max(-1.0, min(1.0, x)) * 32767))
        try:
            with open(path, "wb") as f:
                w = wave.open(f, "wb")
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(CHIME_RATE)
                w.writeframes(struct.pack(f"<{len(out)}h", *out)); w.close()
        except OSError:
            pass

    # ready — an open rising fifth: an invitation to speak.
    _write(state.CHIME_PATH,
           _render([(A4, 0.000, 0.85, 0.55), (D5, 0.075, 0.95, 0.85)]))

    # processing — a single soft low tick. Deliberately the quietest of the
    # set: it can fire repeatedly and must never draw attention.
    _write(state.CHIME_PROCESSING,
           _render([(F4, 0.0, 0.50, 0.22)], tail=0.15), peak=0.20)

    # speak — one warm note, Ember about to talk.
    _write(state.CHIME_SPEAK,
           _render([(C5, 0.0, 0.95, 0.70)]), peak=0.38)

    # done — a falling fifth resolving down to the root: closure.
    _write(state.CHIME_DONE,
           _render([(C5, 0.000, 0.80, 0.50), (F4, 0.080, 0.90, 1.00)]), peak=0.34)

    # hum — sustained working drone, two octaves beating slowly. Very soft;
    # this one loops continuously while thinking.
    _write(state.CHIME_HUM,
           _render([(F3, 0.0, 0.85, 1.60), (F4, 0.0, 0.30, 1.40)], tail=0.2),
           peak=0.16)

    # pause — damped falling step, short decay: something set down.
    _write(state.CHIME_PAUSE,
           _render([(A4, 0.000, 0.75, 0.30), (F4, 0.055, 0.80, 0.40)]), peak=0.30)

    # resume — the mirror of pause, rising and opening back up.
    _write(state.CHIME_RESUME,
           _render([(F4, 0.000, 0.75, 0.35), (A4, 0.055, 0.85, 0.60)]), peak=0.32)


def _play_sound(path):
    """Play a WAV file non-blocking via aplay."""
    if os.path.exists(path):
        try:
            subprocess.Popen(
                ["aplay", "-D", "default", "-q", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# TTY / terminal helpers
# ---------------------------------------------------------------------------

def _get_tty():
    if state._tty_fd is None:
        paths = ["/dev/tty", "/dev/pts/0", "/dev/pts/1", "/dev/pts/2"]
        try:
            ppid = os.getppid()
            ptty = subprocess.check_output(["ps", "-o", "tty=", "-p", str(ppid)], text=True).strip()
            if ptty and not ptty.startswith("?"):
                paths.insert(0, f"/dev/{ptty}")
        except Exception:
            pass

        for path in paths:
            try:
                f = open(path, "w")
                if f.isatty():
                    state._tty_fd = f
                    break
                f.close()
            except Exception:
                continue

    return state._tty_fd


def _get_tty_width():
    """Get terminal width, cached for 2s to avoid repeated /dev/tty opens."""
    now = time.monotonic()
    if state._cached_tty_width is not None and (now - state._cached_tty_width_time) < 2.0:
        return state._cached_tty_width
    width = 120
    if "COLUMNS" in os.environ:
        try:
            width = int(os.environ["COLUMNS"])
            state._cached_tty_width = width
            state._cached_tty_width_time = now
            return width
        except ValueError:
            pass
    try:
        with open("/dev/tty", "r") as tty:
            width = os.get_terminal_size(tty.fileno()).columns
    except Exception:
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
    state._cached_tty_width = width
    state._cached_tty_width_time = now
    return width


_COLOR_MAP = {
    "default": None,
    "green": "32", "light_green": "92",
    "yellow": "33", "amber": "38;5;214", "rust": "38;5;166",
    "red": "31", "light_red": "91",
    "blue": "34", "light_blue": "94",
    "cyan": "36", "light_cyan": "96",
    "magenta": "35", "light_magenta": "95",
    "white": "97", "gray": "90",
}


def _colorize(text, color_name):
    """Wrap text in ANSI color codes. Returns plain text if color is 'default' or unknown."""
    if not color_name or color_name == "default":
        return text
    code = _COLOR_MAP.get(color_name)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_status(text, color_code="90"):
    """Print a status indicator to stderr."""
    if CONFIG.get("visual_indicator", True):
        sys.stderr.write(f"{text}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Chime convenience wrappers
# ---------------------------------------------------------------------------

def play_chime():
    """Ready chime -- ascending tone, signals 'speak now'."""
    if CONFIG.get("chime_ready", True):
        _play_sound(state.CHIME_PATH)
    _print_status("🎤 Listening...", "92")


def play_processing():
    """Processing blip -- signals STT is done, thinking."""
    if CONFIG.get("chime_processing", False):
        _play_sound(state.CHIME_PROCESSING)

    if CONFIG.get("chime_hum", False):
        stop_hum()  # Kill any existing hum before starting a new one
        _print_status("🧠 Thinking...", "94")
        try:
            state._hum_proc = subprocess.Popen(
                ["bash", "-c", "while true; do aplay -D default -q -- \"$1\"; done", "_", state.CHIME_HUM],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass


def stop_hum():
    """Stop the looping hum if active."""
    _print_status("")  # Clear indicator
    proc = state._hum_proc
    state._hum_proc = None
    if proc:
        try:
            import signal as _sig
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, _sig.SIGTERM)
            proc.wait(timeout=0.3)
        except Exception:
            # SIGTERM failed or timed out — force kill the process group
            try:
                os.killpg(pgid, _sig.SIGKILL)
                proc.wait(timeout=0.3)
            except Exception:
                pass


def play_speak():
    """Speak chime -- descending tone, signals TTS starting."""
    if CONFIG.get("chime_speak", False):
        _play_sound(state.CHIME_SPEAK)


def play_done():
    """Done tap -- signals TTS finished."""
    if CONFIG.get("chime_done", False):
        _play_sound(state.CHIME_DONE)


# ---------------------------------------------------------------------------
# Energy / VAD helpers
# ---------------------------------------------------------------------------

def rms_energy(frame_bytes):
    """Calculate RMS energy of a 16-bit PCM frame."""
    n = len(frame_bytes) // 2
    if n == 0:
        return 0.0
    # Fast path: numpy vectorized RMS (~5-10x faster, SIMD-optimized)
    if _HAS_NUMPY and len(frame_bytes) == state.FRAME_BYTES:
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    # Fallback: struct.unpack + Python generator
    if len(frame_bytes) == state.FRAME_BYTES:
        samples = struct.unpack(_RMS_FMT, frame_bytes)
    else:
        samples = struct.unpack(f'<{n}h', frame_bytes[:n * 2])
    return (sum(s * s for s in samples) / n) ** 0.5


def calibrate_noise(proc, n_frames=state.ENERGY_CALIBRATION_FRAMES):
    """Estimate ambient noise energy. Uses cached threshold if fresh (saves ~150ms)."""
    if state._cached_noise_threshold is not None and (time.time() - state._cached_noise_time) < state.NOISE_CACHE_TTL:
        chunk = proc.stdout.read(state.FRAME_BYTES)
        frames = [chunk] if chunk and len(chunk) == state.FRAME_BYTES else []
        return state._cached_noise_threshold, frames
    energies = []
    frames = []
    for _ in range(n_frames):
        chunk = proc.stdout.read(state.FRAME_BYTES)
        if not chunk or len(chunk) < state.FRAME_BYTES:
            break
        frames.append(chunk)
        energies.append(rms_energy(chunk))
    if not energies:
        return 500.0, frames
    ambient = sum(energies) / len(energies)
    threshold = max(ambient * state.ENERGY_THRESHOLD_MULTIPLIER, 300.0)
    state._cached_noise_threshold = threshold
    state._cached_noise_time = time.time()
    return threshold, frames


def is_speech_energy(chunk, vad, energy_threshold):
    """Combined VAD + energy gate: both must agree it's speech."""
    energy = rms_energy(chunk)
    if energy < energy_threshold:
        return False
    if vad:
        try:
            return vad.is_speech(chunk, state.SAMPLE_RATE)
        except Exception:
            return True
    return True


# ---------------------------------------------------------------------------
# Whisper model
# ---------------------------------------------------------------------------

def get_whisper_model():
    """Lazy-load whisper model (first call downloads ~150MB)."""
    if not state.HAS_WHISPER:
        return None
    if state._whisper_model is None:
        state._whisper_model = state.WhisperModel("base", device="cpu", compute_type="int8")
    return state._whisper_model


# ---------------------------------------------------------------------------
# WAV writing and quick STT helpers
# ---------------------------------------------------------------------------

def write_wav(path, raw_data):
    """Write raw PCM data as a WAV file."""
    with open(path, "wb") as f:
        f.write(struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + len(raw_data), b'WAVE', b'fmt ', 16, 1, 1,
            state.SAMPLE_RATE, state.SAMPLE_RATE * 2, 2, 16, b'data', len(raw_data)))
        f.write(raw_data)


def _quick_stt(raw_frames):
    """Fast Azure REST STT for short audio (barge-in keyword detection)."""
    import tempfile as _tempfile
    raw_data = b"".join(raw_frames)
    if not raw_data or len(raw_data) < state.FRAME_BYTES * 3:
        return ""
    _tmp = _tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = _tmp.name
    _tmp.close()
    try:
        write_wav(wav_path, raw_data)
        url = (f"https://{CONFIG['region']}.stt.speech.microsoft.com"
               f"/speech/recognition/conversation/cognitiveservices/v1")
        headers = {
            "Ocp-Apim-Subscription-Key": CONFIG["key"],
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        with open(wav_path, "rb") as f:
            resp = get_http_session().post(
                url, params={"language": "en-US"}, headers=headers, data=f, timeout=10,
            )
        if resp.status_code == 200:
            import json as _json
            result = _json.loads(resp.text)
            if result.get("RecognitionStatus") == "Success":
                return result.get("DisplayText", "").strip()
    except Exception:
        pass
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
    return ""


def _classify_voice_cmd(raw_frames):
    """Transcribe short barge-in audio via Azure STT and classify as voice command."""
    text = _quick_stt(raw_frames).lower()
    if not text:
        return "stop"
    for w in ("resume", "continue", "go on", "go ahead", "keep going"):
        if w in text:
            return "resume"
    for w in ("pause", "wait", "hold on", "hold"):
        if w in text:
            return "pause"
    for w in ("stop", "skip", "shut up", "quiet", "enough", "never mind"):
        if w in text:
            return "stop"
    for w in ("repeat", "again", "say again", "what", "say that again"):
        if w in text:
            return "repeat"
    return "reply"


# ---------------------------------------------------------------------------
# Recording with VAD
# ---------------------------------------------------------------------------

def record_with_vad(proc, max_seconds, stop_when=None, on_audio=None):
    """Record with energy-gated VAD, returning (raw_frames, energy_threshold).

    Two ways to end early, and they mean different things:
      is_cancelled()  the process-global cancel wire -- ABANDON (the caller
                      discards the frames);
      stop_when()     an optional per-call predicate -- FINISH NOW and KEEP
                      what was captured. This is the dictation hotkey / a badge
                      tap in loop mode: without it a batch (VAD) recording ran
                      on until silence or max_seconds, and a tap looked ignored
                      (gnome-speaks #110).

    on_audio(frames)  an optional observer of the utterance SO FAR: called on
                      the recording thread every PARTIAL_INTERVAL_MS once speech
                      has been heard, with the live frame list (read it, do not
                      keep it -- the recorder appends to it). It is how the
                      Wyoming route grows live partials (stt_vad partial_cb);
                      it must return quickly and never raise.
    """
    energy_threshold, calibration_frames = calibrate_noise(proc)
    vad = state.webrtcvad.Vad(state.VAD_AGGRESSIVENESS) if state.HAS_VAD else None
    frames = list(calibration_frames)
    silence_frames = 0
    speech_frames = 0
    max_silence = int(state.SILENCE_TIMEOUT * 1000 / state.FRAME_MS)
    max_no_speech = int(state.NO_SPEECH_TIMEOUT * 1000 / state.FRAME_MS)
    min_speech = int(state.MIN_SPEECH_DURATION * 1000 / state.FRAME_MS)
    max_frames = int(max_seconds * 1000 / state.FRAME_MS)
    partial_every = max(1, int(state.PARTIAL_INTERVAL_MS / state.FRAME_MS))
    since_partial = 0
    total_frames = 0

    for _ in range(max_frames):
        if is_cancelled():
            break
        if stop_when is not None and stop_when():
            break  # finish early; the frames so far are the utterance
        chunk = proc.stdout.read(state.FRAME_BYTES)
        if not chunk or len(chunk) < state.FRAME_BYTES:
            break
        frames.append(chunk)
        total_frames += 1
        if is_speech_energy(chunk, vad, energy_threshold):
            speech_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1
        if on_audio is not None and speech_frames > 0:
            since_partial += 1
            if since_partial >= partial_every:
                since_partial = 0
                on_audio(frames)
        if speech_frames >= min_speech and silence_frames >= max_silence:
            break
        if speech_frames == 0 and total_frames >= max_no_speech:
            break

    return frames, energy_threshold


# ---------------------------------------------------------------------------
# Pre-warming
# ---------------------------------------------------------------------------

_REC_IDLE_SECONDS = 30  # kill prewarmed recorder after this many seconds idle
_PLAYER_IDLE_SECONDS = 30  # reap prewarmed player after this many seconds idle (it holds the audio sink open on stdin)

def _prewarm_recorder():
    """Start a recorder process in background so next listen/converse is instant.
    Auto-releases after _REC_IDLE_SECONDS to avoid holding the mic device open."""
    with state._prewarmed_rec_lock:
        if state._prewarmed_rec is not None:
            return  # already pre-warmed
        try:
            proc = subprocess.Popen(
                _build_rec_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            state._prewarmed_rec = proc
            # Cancel any existing idle timer before starting a new one
            if state._rec_idle_timer is not None:
                state._rec_idle_timer.cancel()
            state._rec_idle_timer = threading.Timer(_REC_IDLE_SECONDS, _discard_prewarmed_rec)
            state._rec_idle_timer.daemon = True
            state._rec_idle_timer.start()
        except Exception:
            pass


def _prewarm_player(tts_rate=24000):
    """Start a player process in background so next speak is instant."""
    with state._prewarmed_player_lock:
        if state._prewarmed_player is not None:
            return
        for cmd in _build_player_cmd(tts_rate):
            try:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                state._prewarmed_player = proc
                state._prewarmed_player_rate = tts_rate
                # Auto-reap if never consumed: an idle prewarmed player holds
                # the audio sink open (blocked on stdin) and accumulates across
                # long-lived MCP servers, driving PipeWire xruns (audible crackle).
                if state._player_idle_timer is not None:
                    state._player_idle_timer.cancel()
                state._player_idle_timer = threading.Timer(_PLAYER_IDLE_SECONDS, _discard_prewarmed_player)
                state._player_idle_timer.daemon = True
                state._player_idle_timer.start()
                return
            except FileNotFoundError:
                continue


def _take_prewarmed_rec():
    """Take the pre-warmed recorder if available, or return None."""
    with state._prewarmed_rec_lock:
        proc = state._prewarmed_rec
        state._prewarmed_rec = None
        # Cancel idle timer — recorder is now in active use
        if state._rec_idle_timer is not None:
            state._rec_idle_timer.cancel()
            state._rec_idle_timer = None
        return proc


def _discard_prewarmed_rec():
    """Kill and discard any pre-warmed recorder (e.g. on shutdown or idle timeout)."""
    with state._prewarmed_rec_lock:
        proc = state._prewarmed_rec
        state._prewarmed_rec = None
        if state._rec_idle_timer is not None:
            state._rec_idle_timer.cancel()
            state._rec_idle_timer = None
    if proc:
        try:
            proc.kill()  # SIGKILL — pw-record ignores SIGTERM
            proc.wait(timeout=2)
        except Exception:
            pass


def _discard_prewarmed_player():
    """Close and reap any pre-warmed player (idle timeout or shutdown).

    An unused prewarmed player holds the audio sink open blocked on stdin;
    these accumulate across long-lived MCP servers and cause PipeWire
    xruns/crackle. Closing stdin sends EOF so aplay drains and exits."""
    with state._prewarmed_player_lock:
        proc = state._prewarmed_player
        state._prewarmed_player = None
        if state._player_idle_timer is not None:
            state._player_idle_timer.cancel()
            state._player_idle_timer = None
    if proc:
        try:
            if proc.stdin:
                proc.stdin.close()  # EOF -> aplay drains and exits cleanly
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass


def _take_prewarmed_player(tts_rate):
    """Take the pre-warmed player if rate matches, or return None."""
    with state._prewarmed_player_lock:
        # Player is being taken/replaced -- cancel the idle reaper.
        if state._player_idle_timer is not None:
            state._player_idle_timer.cancel()
            state._player_idle_timer = None
        if state._prewarmed_player is not None and state._prewarmed_player_rate == tts_rate:
            proc = state._prewarmed_player
            state._prewarmed_player = None
            return proc
        # Rate mismatch -- discard the stale player WITHOUT waiting for it.
        if state._prewarmed_player is not None:
            old = state._prewarmed_player
            state._prewarmed_player = None
            _reap_player_async(old)
        return None


def _reap_player_async(proc):
    """EOF a player now; wait for it (and kill it if needed) off-thread.

    Closing stdin returns at once (nothing was ever written to a prewarmed
    player). It is the wait() that costs 164-214 ms (measured, #20 review --
    aplay/pw-play tearing the sink down), and on the take path that wait sat
    between audio-start and the FIRST PCM WRITE of every offline utterance
    whenever the prewarm was at a different rate than the server's.
    """
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass

    def _wait():
        try:
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass

    threading.Thread(target=_wait, daemon=True).start()


def _prewarm_rate():
    """Rate the NEXT utterance is expected to play at.

    Azure streams at 24 kHz (fast quality). While Azure is being skipped --
    speech_backend=local, SPEECH_FORCE_OFFLINE, or Azure on cooldown -- the
    next utterance comes from Wyoming at the SERVER's rate (Piper: 22050),
    and a 24 kHz prewarm could only ever be discarded on take (#20 review).
    The server's rate is learned from its last audio-start (speech_tts sets
    state._wyoming_tts_rate); 22050 until one has been seen.
    """
    try:
        if wyoming.skip_azure():
            return int(state._wyoming_tts_rate or 22050)
    except Exception:
        pass
    return 24000


def _start_player(tts_rate, target=None):
    """Start a fresh player process, trying each configured command."""
    for cmd in _build_player_cmd(tts_rate, target=target):
        try:
            return subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            continue
    return None


def _prewarm_all():
    """Pre-warm recorder + player + refresh STT WebSocket + TTS connection."""
    _prewarm_recorder()
    _prewarm_player(_prewarm_rate())  # 24 kHz for Azure, the server's rate for Wyoming
    # Refresh STT WebSocket if it might have timed out
    try:
        if state.HAS_WS:
            from stt import _get_stt_ws
            _get_stt_ws()
    except Exception:
        pass
    # Keep TTS HTTP connection alive (use tts_region, not STT region)
    try:
        session = get_http_session()
        region = CONFIG.get("tts_region") or CONFIG.get("region")
        if region:
            session.head(f"https://{region}.tts.speech.microsoft.com", timeout=3)
    except Exception:
        pass
    with state._warmup_lock:
        state._warmup_pending = False


def _schedule_warmup():
    """Debounced warmup -- collapses rapid successive calls into one thread."""
    with state._warmup_lock:
        if state._warmup_pending:
            return
        state._warmup_pending = True
    threading.Thread(target=_prewarm_all, daemon=True).start()
