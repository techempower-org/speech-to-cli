"""Speech-to-text backends: streaming WebSocket, VAD+REST, Whisper, fixed-duration.

Import pattern: `import state` at top. Reassignable globals accessed as `state.X`.
"""

import json
import os
import struct
import subprocess
import tempfile
import threading
import time
import uuid

import state
import wyoming
from state import (CONFIG, HAS_VAD, HAS_WS, HAS_WHISPER,
                   is_cancelled, register_proc, unregister_proc,
                   get_http_session, send_progress, _get_iso_timestamp)
from audio import (play_chime, play_processing, calibrate_noise, is_speech_energy,
                   record_with_vad, write_wav, rms_energy, get_whisper_model,
                   _build_rec_cmd, _take_prewarmed_rec, _get_tty_width, _colorize)

if HAS_WS:
    import websocket
if HAS_VAD:
    import webrtcvad


# ---------------------------------------------------------------------------
# WebSocket helpers
# ---------------------------------------------------------------------------

def _make_ws_audio_msg(request_id, audio_data):
    """Build a binary WebSocket message for Azure STT audio streaming."""
    hdr = (
        f"Path: audio\r\n"
        f"X-RequestId: {request_id}\r\n"
        f"X-Timestamp: {_get_iso_timestamp()}\r\n"
        f"Content-Type: audio/x-wav\r\n"
    ).encode('ascii')
    return struct.pack('>H', len(hdr)) + hdr + audio_data


def _get_stt_ws():
    """Get or create persistent WebSocket to Azure STT (saves ~230ms on reuse).

    Returns (ws, is_fresh) where is_fresh=True if a new connection was just created.
    Callers should pass drain=not is_fresh to _init_stt_ws_session().
    """
    now = time.time()
    if state._persistent_ws is not None and (now - state._persistent_ws_time) < state.WS_IDLE_TIMEOUT:
        state._persistent_ws_time = now
        return state._persistent_ws, False
    if state._persistent_ws is not None:
        try:
            state._persistent_ws.close()
        except Exception:
            pass
        state._persistent_ws = None
    region = CONFIG["region"]
    key = CONFIG["key"]
    lang = CONFIG.get("language", "en-US")
    state._persistent_ws = websocket.create_connection(
        f"wss://{region}.stt.speech.microsoft.com/speech/recognition"
        f"/conversation/cognitiveservices/v1?language={lang}&format=detailed",
        header=[
            f"Ocp-Apim-Subscription-Key: {key}",
            f"X-ConnectionId: {uuid.uuid4().hex}",
        ],
        timeout=10,
    )
    state._persistent_ws_time = time.time()
    return state._persistent_ws, True


def _invalidate_stt_ws():
    """Close and discard the persistent WebSocket."""
    if state._persistent_ws is not None:
        try:
            state._persistent_ws.close()
        except Exception:
            pass
        state._persistent_ws = None


# ---------------------------------------------------------------------------
# Shared STT helpers (used by stt_streaming and talk_fullduplex)
# ---------------------------------------------------------------------------

def _make_logger(tag):
    """Create debug logger for STT/TTS operations."""
    dbg = "/tmp/speech-debug.log" if (os.environ.get("SPEECH_DEBUG") or CONFIG.get("debug")) else None
    def _log(msg):
        if dbg:
            with open(dbg, "a") as f:
                f.write(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}\n")
    return _log


def _check_end_word(text, end_word=None):
    """Check if text ends with the configured end word."""
    if end_word is None:
        end_word = CONFIG.get("end_word", "over")
    if not end_word:
        return False
    words = text.lower().strip().rstrip(".!?,").split()
    return len(words) > 0 and words[-1] == end_word.lower()


def _strip_end_word(text, end_word=None):
    """Remove trailing end word from transcription."""
    if end_word is None:
        end_word = CONFIG.get("end_word", "over")
    if not end_word or not text:
        return text
    import re as _re
    return _re.sub(r'\s*\b' + _re.escape(end_word) + r'[.!?,]*\s*$', '', text, flags=_re.IGNORECASE).strip()


def _silence_icon(ratio):
    """Return countdown icon based on silence ratio (0.0 to 1.0)."""
    if ratio < 0.25:
        return ""
    if ratio < 0.50:
        return " ◔"
    if ratio < 0.75:
        return " ◑"
    if ratio < 0.90:
        return " ◕"
    return " ●"


def _window_partial(text, term_width=None):
    """Truncate partial text to fit terminal, with '...' prefix if needed."""
    if term_width is None:
        term_width = _get_tty_width()
    window = max(40, term_width - 25)
    return text if len(text) < window else "..." + text[-(window - 3):]


# Pre-computed WAV header for STT init (never changes)
_STT_WAV_HEADER = struct.pack('<4sI4s4sIHHIIHH4sI',
    b'RIFF', 0, b'WAVE', b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, b'data', 0)

# Pre-serialized speech config JSON (never changes)
def _build_speech_config():
    """Build speech.config JSON, including custom phrase list if configured."""
    config = {
        "context": {
            "system": {"version": "1.0.00000"},
            "os": {"platform": "Linux", "name": "speech-to-cli"},
            "audio": {"source": {"connectivity": "Unknown", "manufacturer": "Unknown",
                                 "model": "Unknown", "type": "Unknown"}},
        }
    }
    # Custom phrase list boosts recognition of domain-specific terms
    phrases = CONFIG.get("phrase_list", [])
    if phrases:
        config["phraseDetection"] = {
            "enrichment": {
                "pronunciationAssessment": {},
                "phraseList": [{"phrase": p} for p in phrases],
            }
        }
    return json.dumps(config)


_STT_SPEECH_CONFIG = _build_speech_config()


def _init_stt_ws_session(ws, request_id, drain=True):
    """Initialize an STT WebSocket session: drain stale msgs, send config + WAV header.

    Set drain=False on freshly created connections (no stale messages to clear).
    """
    if drain:
        try:
            ws.settimeout(0.05)
            while True:
                ws.recv()
        except Exception:
            pass
    ws.send(
        f"Path: speech.config\r\nX-RequestId: {request_id}\r\n"
        f"X-Timestamp: {_get_iso_timestamp()}\r\n"
        f"Content-Type: application/json\r\n\r\n" + _STT_SPEECH_CONFIG
    )
    ws.send(_make_ws_audio_msg(request_id, _STT_WAV_HEADER), opcode=websocket.ABNF.OPCODE_BINARY)


def _parse_ws_msg(msg, phrases, partial_holder, end_word_event, end_word, _log,
                   raw_partial_holder=None, use_lexical=False):
    """Parse a WS STT message. Updates phrases/partial_holder in place.

    If raw_partial_holder is provided, also stores the unwindowed full text there
    (useful for live typing where terminal-width truncation would corrupt diffs).

    If use_lexical is True, uses the ITN field (lowercase, no punctuation,
    but with number/symbol normalization) instead of Display for phrase
    results. Useful for terminal/code input.

    Returns: 'hypothesis', 'phrase', 'turn_end', or None.
    """
    if not isinstance(msg, str):
        return None
    parts = msg.split("\r\n\r\n", 1)
    if len(parts) < 2:
        return None
    hdr, body = parts
    hdr_lower = hdr.lower()

    if "speech.hypothesis" in hdr_lower:
        try:
            partial = json.loads(body).get("Text", "")
            if partial:
                if _check_end_word(partial, end_word):
                    end_word_event.set()
                full = (" ".join(phrases) + " " + partial).strip() if phrases else partial
                partial_holder[0] = _window_partial(full)
                if raw_partial_holder is not None:
                    raw_partial_holder[0] = full
        except Exception:
            pass
        return "hypothesis"

    if "speech.phrase" in hdr_lower:
        try:
            data = json.loads(body)
            status = data.get("RecognitionStatus", "?")
            if status == "Success":
                nbest = data.get("NBest", [])
                if use_lexical and nbest:
                    phrase = nbest[0].get("ITN", nbest[0].get("Display", ""))
                else:
                    phrase = nbest[0]["Display"] if nbest else data.get("DisplayText", "")
                if phrase:
                    phrases.append(phrase)
                    if _check_end_word(phrase, end_word):
                        end_word_event.set()
                    full = " ".join(phrases)
                    partial_holder[0] = _window_partial(full)
                    if raw_partial_holder is not None:
                        raw_partial_holder[0] = full
            else:
                _log(f"phrase status={status}: {body[:150]}")
        except Exception:
            pass
        return "phrase"

    if "turn.end" in hdr_lower:
        return "turn_end"

    return None


def _wyoming_stt(raw_data, _log):
    """Offline STT via the configured Wyoming server. Returns text or ""."""
    host = CONFIG.get("wyoming_host", "")
    if not host:
        return ""
    if raw_data[:4] == b"RIFF":
        raw_data = raw_data[44:]  # strip our own standard WAV header
    try:
        text = wyoming.transcribe(
            host, int(CONFIG.get("wyoming_stt_port", 10300)),
            raw_data, rate=16000)
        _log(f"Wyoming STT recovered: {repr(text[:100])}")
        return text
    except wyoming.WyomingError as e:
        wyoming.mark_local_down()  # the LAN server failed, not the speaker
        _log(f"Wyoming STT failed: {e}")
        return ""


def _rest_stt_fallback(raw_frames, _log=None):
    """Fallback STT via REST API when WS fails. Returns text or empty string.

    Falls through to the LAN Wyoming server only on network-class Azure
    failures — an Azure-reachable "NoMatch" (silence) stays silent.
    """
    if _log is None:
        _log = lambda msg: None
    raw_data = b"".join(raw_frames) if isinstance(raw_frames, list) else raw_frames
    if not raw_data:
        return ""
    if wyoming.skip_azure():
        reason = wyoming.skip_reason()
        _log(f"Azure skipped ({reason}) — Wyoming STT direct")
        text = _wyoming_stt(raw_data, _log)
        if text or not wyoming.local_down() or not wyoming.azure_fallback_allowed() or reason == "azure_down":
            return text
        _log("Wyoming STT failed — falling back to Azure REST")
    azure_reachable = False
    _log(f"REST STT fallback: {len(raw_data)} bytes")
    # Build WAV in memory — no disk I/O
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(raw_data), b'WAVE', b'fmt ', 16, 1, 1,
        16000, 32000, 2, 16, b'data', len(raw_data))
    wav_data = wav_header + raw_data
    lang = CONFIG.get("language", "en-US")
    try:
        stt_url = (f"https://{CONFIG['region']}.stt.speech.microsoft.com"
                   f"/speech/recognition/conversation/cognitiveservices/v1")
        stt_headers = {
            "Ocp-Apim-Subscription-Key": CONFIG["key"],
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        resp = get_http_session().post(
            stt_url, params={"language": lang, "format": "detailed"},
            headers=stt_headers, data=wav_data, timeout=30,
        )
        if resp.status_code == 200:
            azure_reachable = True
            wyoming.mark_azure_up()
            result = resp.json()
            status = result.get("RecognitionStatus", "?")
            _log(f"REST STT status={status}")
            if status == "Success":
                nbest = result.get("NBest", [])
                text = nbest[0]["Display"] if nbest else result.get("DisplayText", "")
                _log(f"REST STT recovered: {repr(text[:100])}")
                return text
            _log(f"REST STT non-success: {status}")
        else:
            _log(f"REST STT HTTP error: {resp.status_code}")
            if resp.status_code >= 500:
                wyoming.mark_azure_down()
    except Exception as e:
        _log(f"REST STT exception: {e}")
        wyoming.mark_azure_down()
    if azure_reachable:
        return ""
    return _wyoming_stt(raw_data, _log)


# ---------------------------------------------------------------------------
# STT backends
# ---------------------------------------------------------------------------

def stt_streaming(max_seconds=30, progress_token=None):
    """Real-time STT via persistent Azure WebSocket + energy-gated VAD."""
    _log = _make_logger("stt-stream")
    _end_word = CONFIG.get("end_word", "over")

    def _do_streaming(ws, max_seconds, is_fresh=False):
        request_id = uuid.uuid4().hex

        proc = _take_prewarmed_rec()
        if proc is None:
            proc = subprocess.Popen(
                _build_rec_cmd(max_seconds=max_seconds),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )

        _init_stt_ws_session(ws, request_id, drain=not is_fresh)
        play_chime()
        send_progress(progress_token, 0, 100, "🎤 Listening...")
        register_proc(proc)

        phrases = []
        sender_done = threading.Event()
        end_word_detected = threading.Event()
        silence_ratio = [0.0]
        rec_frames = []
        partial_text = ["Listening..."]

        def send_audio():
            try:
                energy_threshold, calibration_frames = calibrate_noise(proc)
                _log(f"calibrated: threshold={energy_threshold:.0f}, cal_frames={len(calibration_frames)}")

                for frame in calibration_frames:
                    ws.send(_make_ws_audio_msg(request_id, frame), opcode=websocket.ABNF.OPCODE_BINARY)

                vad = webrtcvad.Vad(state.VAD_AGGRESSIVENESS) if HAS_VAD else None
                silence_frames = 0
                speech_frames = 0
                total_frames = 0
                max_silence = int(state.SILENCE_TIMEOUT * 1000 / state.FRAME_MS)
                max_no_speech = int(state.NO_SPEECH_TIMEOUT * 1000 / state.FRAME_MS)
                min_speech = int(state.MIN_SPEECH_DURATION * 1000 / state.FRAME_MS)
                max_frames = int(max_seconds * 1000 / state.FRAME_MS)
                last_progress_pct = 0
                bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
                _show_vu = CONFIG.get("vu_meter", True)
                _show_subs = CONFIG.get("live_subtitles", True)
                _user_color = CONFIG.get("subtitle_color_user")
                _log(f"limits: max_silence={max_silence} max_no_speech={max_no_speech} min_speech={min_speech}")

                while True:
                    if is_cancelled():
                        _log("cancelled")
                        break
                    chunk = proc.stdout.read(state.FRAME_BYTES)
                    if not chunk:
                        break
                    rec_frames.append(chunk)
                    try:
                        ws.send(_make_ws_audio_msg(request_id, chunk), opcode=websocket.ABNF.OPCODE_BINARY)
                    except Exception as _e:
                        _log(f"WS send error (frame {total_frames}): {_e}")
                        break
                    if len(chunk) == state.FRAME_BYTES:
                        total_frames += 1
                        current_pct = int((total_frames / max_frames) * 70)
                        energy = rms_energy(chunk)
                        idx = max(0, min(7, int((energy / 1200.0) * 8)))
                        vu = bars[idx]
                        timer_icon = _silence_icon(silence_ratio[0])

                        if total_frames % 3 == 0 or current_pct > last_progress_pct:
                            parts = ["🎤"]
                            if _show_subs:
                                parts.append(_colorize(partial_text[0], _user_color) + timer_icon)
                            else:
                                parts.append("Listening..." + timer_icon)
                            if _show_vu:
                                parts.append(vu)
                            send_progress(progress_token, current_pct, 100, " ".join(parts))
                            last_progress_pct = current_pct

                        if is_speech_energy(chunk, vad, energy_threshold):
                            speech_frames += 1
                            silence_frames = 0
                        else:
                            silence_frames += 1

                        if max_silence > 0:
                            silence_ratio[0] = silence_frames / max_silence

                        if total_frames % 33 == 0:
                            _log(f"f={total_frames} speech={speech_frames} silence={silence_frames} energy={energy:.0f} thresh={energy_threshold:.0f}")

                        if end_word_detected.is_set():
                            _log(f"STOP: end word '{_end_word}' detected. speech={speech_frames}")
                            break
                        if speech_frames >= min_speech and silence_frames >= max_silence:
                            _log(f"STOP: silence timeout. speech={speech_frames} silence={silence_frames}/{max_silence}")
                            send_progress(progress_token, 80, 100, "🧠 Transcribing...")
                            break
                        if speech_frames == 0 and total_frames >= max_no_speech:
                            _log(f"STOP: no speech timeout. total={total_frames}/{max_no_speech}")
                            break
                _log(f"REC END: speech={speech_frames} total={total_frames} phrases={len(phrases)}")
            except Exception as e:
                _log(f"exception: {e}")
            finally:
                proc.terminate()
                proc.wait()
                unregister_proc(proc)
                try:
                    ws.send(_make_ws_audio_msg(request_id, b""), opcode=websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    pass
                sender_done.set()

        sender = threading.Thread(target=send_audio, daemon=True)
        sender.start()

        # Read WS responses until turn.end
        deadline = time.time() + max_seconds + 5
        got_phrase = False
        while time.time() < deadline and not is_cancelled():
            try:
                ws.settimeout(1.0)
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                if sender_done.is_set():
                    if got_phrase:
                        break
                    try:
                        ws.settimeout(2.0)
                        msg = ws.recv()
                    except Exception:
                        break
                else:
                    continue
            except Exception:
                break

            mtype = _parse_ws_msg(msg, phrases, partial_text, end_word_detected, _end_word, _log)
            if mtype == "phrase":
                got_phrase = True
                _user_color = CONFIG.get("subtitle_color_user")
                send_progress(progress_token, 90, 100, f"🎤 {_colorize(' '.join(phrases), _user_color)}")
                if sender_done.is_set():
                    try:
                        ws.settimeout(0.5)
                        ws.recv()
                    except Exception:
                        pass
                    break
            elif mtype == "turn_end":
                _log(f"turn.end received (phrases={len(phrases)})")
                got_phrase = True  # WS analyzed audio, even if no speech found
                break

        sender.join(timeout=2)
        user_text = " ".join(phrases).strip()

        # Only fall back to REST if the WS never responded (connection failure).
        # If WS returned a phrase status (even InitialSilenceTimeout) or turn.end,
        # it already analyzed the audio — REST would just repeat the same result.
        if not user_text and rec_frames and not got_phrase:
            _log(f"WS returned nothing, falling back to REST STT (frames={len(rec_frames)})")
            user_text = _rest_stt_fallback(rec_frames, _log)
        elif not user_text and got_phrase:
            _log(f"WS analyzed audio but found no speech (skipping REST fallback)")

        user_text = _strip_end_word(user_text, _end_word)
        _log(f"FINAL: {repr(user_text[:100])}")

        _user_color = CONFIG.get("subtitle_color_user")
        if user_text:
            send_progress(progress_token, 100, 100, f"🎤 {_colorize(user_text, _user_color)}")
        else:
            send_progress(progress_token, 100, 100, "✅ Done")
        return user_text

    # Try persistent connection, retry once on failure
    for attempt in range(2):
        try:
            ws, is_fresh = _get_stt_ws()
            text = _do_streaming(ws, max_seconds, is_fresh)
            if is_cancelled():
                return {"text": "", "cancelled": True}
            return {"text": text}
        except Exception as e:
            _invalidate_stt_ws()
            if attempt == 0:
                continue
            return {"error": str(e)}


class _PartialTranscriber:
    """Live partials for the Wyoming route of stt_vad (gnome-speaks live typing).

    The LAN recognizer has no streaming protocol, so the utterance so far is
    re-transcribed from a side thread and each answer is a hypothesis. One
    request in flight, newest audio wins (a one-slot mailbox, the same shape
    as gnome-speaks' _LiveTyper): the recorder drops the live frame list here
    every PARTIAL_INTERVAL_MS, and if a request is still running the snapshot
    is simply replaced. Each request carries PARTIAL_TIMEOUT: measured on
    2026-09-12, some short clips never get an answer while the server keeps
    serving other connections, so a hung partial costs one timeout and
    nothing else -- it is never mark_local_down() (the server is not down)
    and it never delays the final transcription, which opens its own socket.
    partial_cb(text) runs on this thread; the caller makes it thread-safe.
    """

    def __init__(self, host, port, partial_cb, timeout):
        self._host, self._port, self._cb, self._timeout = host, port, partial_cb, timeout
        self._cv = threading.Condition()
        self._pending = None
        self._closed = False
        self._last = ""
        self.requests = 0          # observability for the repro
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="vad-partials")
        self._thread.start()

    def submit(self, frames):
        """record_with_vad's on_audio: snapshot the utterance so far."""
        pcm = b"".join(frames)
        with self._cv:
            self._pending = pcm
            self._cv.notify()

    def close(self, timeout=0.2):
        """Stop taking snapshots. Does not wait out a hung request: the thread
        is a daemon and the final transcription does not depend on it."""
        with self._cv:
            self._closed = True
            self._pending = None
            self._cv.notify()
        self._thread.join(timeout)

    def _run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                pcm, self._pending = self._pending, None
            self.requests += 1
            try:
                text = wyoming.transcribe(self._host, self._port, pcm,
                                          rate=16000, timeout=self._timeout)
            except Exception:
                continue           # a dropped hypothesis, nothing more
            if text and text != self._last:
                self._last = text
                try:
                    self._cb(text)
                except Exception:
                    pass


def stt_vad(max_seconds=30, progress_token=None, stop_when=None, partial_cb=None):
    """Record with energy-gated VAD, stop on silence (or stop_when), upload via REST.

    partial_cb(text): optional live hypotheses while recording -- only on the
    Wyoming route (skip_azure()), where they are re-transcriptions of the
    utterance so far (_PartialTranscriber). The Azure REST route has no live
    partials (streaming is the live Azure path) and never calls it.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    partials = None
    try:
        play_chime()
        send_progress(progress_token, 10, 100, "🎤 Listening...")
        proc = _take_prewarmed_rec()
        if proc is None:
            proc = subprocess.Popen(
                _build_rec_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        register_proc(proc)

        if partial_cb is not None and wyoming.skip_azure() and CONFIG.get("wyoming_host"):
            partials = _PartialTranscriber(
                CONFIG["wyoming_host"], int(CONFIG.get("wyoming_stt_port", 10300)),
                partial_cb, timeout=state.PARTIAL_TIMEOUT)
        frames, _ = record_with_vad(proc, max_seconds, stop_when=stop_when,
                                    on_audio=partials.submit if partials else None)
        if partials is not None:
            partials.close()
        proc.terminate()
        proc.wait()
        unregister_proc(proc)

        if is_cancelled():
            return {"text": "", "cancelled": True}

        if not frames:
            return {"text": "", "status": "NoAudio"}

        raw_data = b"".join(frames)
        write_wav(tmp_path, raw_data)

        send_progress(progress_token, 50, 100, "🧠 Transcribing...")
        if wyoming.skip_azure():
            reason = wyoming.skip_reason()
            with open(tmp_path, "rb") as f:
                text = _wyoming_stt(f.read(), lambda m: None)
            if text or not wyoming.local_down() or not wyoming.azure_fallback_allowed() or reason == "azure_down":
                send_progress(progress_token, 100, 100, "✅ Done (offline)")
                return {"text": text, "engine": "wyoming"}
            # the LAN server failed (not silence): fall through to Azure
        url = f"https://{CONFIG['region']}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": CONFIG["key"],
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        try:
            with open(tmp_path, "rb") as f:
                resp = get_http_session().post(url, params={"language": CONFIG.get("language", "en-US"), "format": "detailed"},
                                     headers=headers, data=f, timeout=30)
        except Exception as e:
            wyoming.mark_azure_down()
            with open(tmp_path, "rb") as f:
                text = _wyoming_stt(f.read(), lambda m: None)
            if text:
                return {"text": text, "engine": "wyoming"}
            return {"error": f"Azure STT unreachable ({e}); offline fallback failed"}
        if resp.status_code != 200:
            if resp.status_code >= 500:
                wyoming.mark_azure_down()
                with open(tmp_path, "rb") as f:
                    text = _wyoming_stt(f.read(), lambda m: None)
                if text:
                    return {"text": text, "engine": "wyoming"}
            return {"error": f"Azure STT error {resp.status_code}"}
        wyoming.mark_azure_up()
        result = resp.json()
        if result.get("RecognitionStatus") == "Success":
            nbest = result.get("NBest", [])
            text = nbest[0]["Display"] if nbest else result.get("DisplayText", "")
            send_progress(progress_token, 100, 100, "✅ Done")
            return {"text": text}
        return {"text": "", "status": result.get("RecognitionStatus", "Unknown")}
    finally:
        try:
            os.unlink(tmp_path)
            send_progress(progress_token, 100, 100, "✅ Done")
        except OSError:
            pass


def stt_whisper(max_seconds=30, progress_token=None):
    """Local STT via faster-whisper - no network needed."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        send_progress(progress_token, 0, 100, "🎤 Listening...")
        proc = _take_prewarmed_rec()
        if proc is None:
            proc = subprocess.Popen(
                _build_rec_cmd(),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        register_proc(proc)

        if HAS_VAD:
            frames, _ = record_with_vad(proc, max_seconds)
            proc.terminate()
            proc.wait()
            raw_data = b"".join(frames)
        else:
            raw_data = proc.stdout.read(state.SAMPLE_RATE * 2 * max_seconds)
            proc.terminate()
            proc.wait()
        unregister_proc(proc)

        if is_cancelled():
            return {"text": "", "cancelled": True}

        if not raw_data:
            return {"text": "", "status": "NoAudio"}

        write_wav(tmp_path, raw_data)

        send_progress(progress_token, 50, 100, "🧠 Transcribing...")
        model = get_whisper_model()
        segments, _ = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        send_progress(progress_token, 100, 100, "✅ Done")
        return {"text": text}
    finally:
        try:
            os.unlink(tmp_path)
            send_progress(progress_token, 100, 100, "✅ Done")
        except OSError:
            pass


def stt_fixed(seconds=5, progress_token=None):
    """Record for a fixed duration, then upload (fallback)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        play_chime()
        send_progress(progress_token, 0, 100, "🎤 Listening...")
        rec = CONFIG.get("recorder", "auto")
        mic = CONFIG.get("mic_source") or "default"
        if rec == "arecord" or rec == "auto":
            cmd = ["arecord", "-D", mic, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav",
                   "-d", str(seconds), tmp_path]
        else:
            cmd = ["pw-record", "--format", "s16", "--rate", "16000", "--channels", "1"]
            if CONFIG.get("mic_source"):
                cmd += ["--target", CONFIG["mic_source"]]
            cmd.append(tmp_path)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        register_proc(proc)
        try:
            proc.wait(timeout=seconds + 5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait()
        unregister_proc(proc)

        if is_cancelled():
            return {"text": "", "cancelled": True}
        if not os.path.exists(tmp_path):
            return {"error": "Recording failed"}

        send_progress(progress_token, 50, 100, "🧠 Transcribing...")
        if wyoming.skip_azure():
            reason = wyoming.skip_reason()
            with open(tmp_path, "rb") as f:
                text = _wyoming_stt(f.read(), lambda m: None)
            if text or not wyoming.local_down() or not wyoming.azure_fallback_allowed() or reason == "azure_down":
                send_progress(progress_token, 100, 100, "✅ Done (offline)")
                return {"text": text, "engine": "wyoming"}
            # the LAN server failed (not silence): fall through to Azure
        url = f"https://{CONFIG['region']}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": CONFIG["key"],
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        try:
            with open(tmp_path, "rb") as f:
                resp = get_http_session().post(url, params={"language": CONFIG.get("language", "en-US"), "format": "detailed"},
                                     headers=headers, data=f, timeout=30)
        except Exception as e:
            wyoming.mark_azure_down()
            with open(tmp_path, "rb") as f:
                text = _wyoming_stt(f.read(), lambda m: None)
            if text:
                return {"text": text, "engine": "wyoming"}
            return {"error": f"Azure STT unreachable ({e}); offline fallback failed"}
        if resp.status_code != 200:
            if resp.status_code >= 500:
                wyoming.mark_azure_down()
                with open(tmp_path, "rb") as f:
                    text = _wyoming_stt(f.read(), lambda m: None)
                if text:
                    return {"text": text, "engine": "wyoming"}
            return {"error": f"Azure STT error {resp.status_code}"}
        wyoming.mark_azure_up()
        result = resp.json()
        if result.get("RecognitionStatus") == "Success":
            nbest = result.get("NBest", [])
            text = nbest[0]["Display"] if nbest else result.get("DisplayText", "")
            send_progress(progress_token, 100, 100, "✅ Done")
            return {"text": text}
        return {"text": "", "status": result.get("RecognitionStatus", "Unknown")}
    finally:
        try:
            os.unlink(tmp_path)
            send_progress(progress_token, 100, 100, "✅ Done")
        except OSError:
            pass


def stt(seconds=None, mode=None, silence_timeout=None, vad_aggressiveness=None,
        energy_multiplier=None, progress_token=None, stop_when=None, partial_cb=None):
    """Speech-to-text with automatic mode selection.

    stop_when: optional predicate; when it turns true the batch (VAD)
    recorder finishes EARLY and keeps what it captured -- the dictation
    hotkey / loop-mode badge tap. Distinct from the cancel wire, which
    abandons. Honoured by the vad mode; streaming has its own stop path and
    fixed-length recording cannot be cut short without corrupting the WAV.

    partial_cb(text): live hypotheses while recording, vad mode on the
    Wyoming route only (see stt_vad). Called from a worker thread.
    """
    max_seconds = max(1, min(int(seconds or 30), 30))

    with state._stt_lock:
        old_silence = state.SILENCE_TIMEOUT
        old_vad = state.VAD_AGGRESSIVENESS
        old_energy = state.ENERGY_THRESHOLD_MULTIPLIER

        if silence_timeout is not None:
            state.SILENCE_TIMEOUT = max(0.1, min(float(silence_timeout), 10.0))
        if vad_aggressiveness is not None:
            state.VAD_AGGRESSIVENESS = max(0, min(int(vad_aggressiveness), 3))
        if energy_multiplier is not None:
            state.ENERGY_THRESHOLD_MULTIPLIER = max(0.5, min(float(energy_multiplier), 20.0))

        try:
            if mode is None:
                if HAS_WS and HAS_VAD:
                    mode = "streaming"
                elif HAS_VAD:
                    mode = "vad"
                else:
                    mode = "fixed"

            if mode == "streaming" and HAS_WS:
                result = stt_streaming(max_seconds, progress_token=progress_token)
            elif mode == "whisper" and HAS_WHISPER:
                result = stt_whisper(max_seconds, progress_token=progress_token)
            elif mode == "vad" and HAS_VAD:
                result = stt_vad(max_seconds, progress_token=progress_token,
                                 stop_when=stop_when, partial_cb=partial_cb)
            else:
                result = stt_fixed(max_seconds, progress_token=progress_token)

            if result.get("text"):
                play_processing()
            return result
        finally:
            state.SILENCE_TIMEOUT = old_silence
            state.VAD_AGGRESSIVENESS = old_vad
            state.ENERGY_THRESHOLD_MULTIPLIER = old_energy
