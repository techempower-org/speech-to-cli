<!-- claude-md-version: a4f6ebd | updated: 2026-03-22 -->
# Azure Speech MCP Server (speech-to-cli)

MCP server providing voice I/O (STT + TTS) for AI CLI agents via Azure Speech Services.
Version 4.5.0. Python 3, threading-based, no Azure SDK — plain REST/WebSocket.
License: GPL-3.0.

## Tech Stack

- **Python 3** (no venv required — uses system python3)
- **Azure Speech Services** — REST API for TTS and STT, WebSocket for streaming STT
- **PipeWire / ALSA** — audio recording and playback
- **webrtcvad** — voice activity detection
- **websocket-client** — persistent WebSocket for streaming STT
- **faster-whisper** — optional local STT (superseded as the offline path by the Wyoming LAN fallback below; remains available as the explicit `whisper` mode if installed)
- **Wyoming fallback** — STT/TTS automatically fall back to a LAN Wyoming server (Piper TTS + ONNX-ASR STT) on network-class Azure failures. Config: `wyoming_host` ("" = off), `wyoming_tts_port`, `wyoming_stt_port`, `wyoming_tts_voice` in config.json. A 60s circuit breaker skips Azure while it's marked down; `speech_backend: "local"` makes Wyoming PRIMARY with Azure as the fallback (symmetric 60s local breaker, `wyoming.prefer_local()` / `skip_reason()`); `SPEECH_FORCE_OFFLINE=1` forces offline with NO Azure fallback (env override for tests, not a setting). Degradation: single Piper voice. Live partials on the Wyoming STT route are SYNTHESIZED (2026-09-12): the server has no streaming protocol (`supports_transcript_streaming: False`), so `stt(mode="vad", partial_cb=)` → `stt_vad(partial_cb=)` runs `_PartialTranscriber` — `record_with_vad(on_audio=)` hands it the live frame list every `state.PARTIAL_INTERVAL_MS` (400) of speech, one `wyoming.transcribe()` in flight (newest snapshot wins), each with `state.PARTIAL_TIMEOUT` (2.5 s). A hung partial is dropped — never `mark_local_down()`, never awaited by the final, which opens its own socket (measured: some short clips never answer while the server keeps serving other connections). Azure REST route: no partials (streaming is Azure's live path). Repro: `tests/repros/vad_partials.py` (red on the pre-seam tree). TTS streams: `wyoming.synthesize_stream()` yields the format at `audio-start` and PCM chunks as they arrive, and `_play_wyoming` plays them as they land (#19). **`tts()` is `tts_play(tts_prepare(...))`** (gnome-speaks#134): `tts_prepare()` is the network half -- route selection, both breakers, the Azure POST or the Wyoming stream up to `audio-start` -- returning a `PreparedTTS` that never holds a player and never raises (an error handle plays as the error dict); `tts_play()` starts the player and runs the loops; `PreparedTTS.close()` releases a handle that will never be played. A prefetching caller prepares N+1 while N plays. Repro: `tests/repros/tts_prepare_play.py` (`STC_BASELINE_TTS=<old speech_tts.py>` makes it two-sided against the pre-seam `tts()`; it reuses `wyoming_stream.py`'s fakes, whose suite is `__main__`-guarded for that reason) -- the collect-all `wyoming.synthesize()` is a wrapper for WAV callers (`tts.py`, `speak.py`); repro in `tests/repros/wyoming_stream.py` (private per-PID scratch under `tmp/repros/`; `wyoming_stream_canary.py` proves its cleanup cannot touch a preset `XDG_STATE_HOME`). Every socket error inside `synthesize_stream()` is a `WyomingError`, and `_tts_wyoming` releases its player in a `finally`, so a mid-stream RST can neither read as `ok` nor orphan the player. `audio._prewarm_rate()` prewarms at the server's learned rate while Azure is skipped, and a rate-mismatched prewarm is discarded without a synchronous `wait()` (#20 review).
- **requests** — HTTP client with connection pooling
- **numpy** — optional, accelerates RMS energy calculation (~5-10x via SIMD)

## Quick Start

```bash
./install.sh                    # or: sudo apt install alsa-utils xclip python3 python3-pip
pip install -r requirements.txt # requests, webrtcvad, websocket-client, faster-whisper
export AZURE_SPEECH_KEY="your-key"
export AZURE_SPEECH_REGION="westus2"
python3 mcp_speech.py           # MCP server
python3 speech.py               # Standalone: mic → clipboard
python3 tts.py "hello"          # Standalone: text → speech
python3 voice_chat.py           # Standalone: voice chat
python3 listen.py --markers     # Standalone: one-shot listen → transcript on stdout (agent loops)
python3 speak.py -v en-US-AndrewNeural "hi"  # Standalone: TTS w/ per-call voice + tts_region
SPEECH_DEBUG=1 python3 mcp_speech.py && tail -f /tmp/speech-debug.log
```

### MCP Integration

**Claude Code** (`.mcp.json`):
```json
{"mcpServers": {"azure-speech": {"command": "python3", "args": ["${HOME}/Projects/speech-to-cli/mcp_speech.py"]}}}
```

**Gemini CLI**: Uses `gemini-extension.json`.

## Architecture

### Module Structure

```
state.py (318L) — constants, CONFIG dict, mutable globals, small helpers
  ↑ imported by all modules
audio.py (734L) — audio I/O, device detection, chimes, VAD, pre-warming
  ↑ imports state
stt.py (683L) — STT backends (streaming WS, VAD+REST, Whisper, fixed)
  ↑ imports state, audio
speech_tts.py (1058L) — TTS, multi_speak, talk_fullduplex, get_voices
  ↑ imports state, audio, stt
mcp_speech.py (997L) — MCP tool schemas, request routing, stdio transport
  ↑ imports state, audio, stt, speech_tts
```

Standalone utilities (`speech.py`, `tts.py`, `voice_chat.py`) import `load_config_standalone` from `state.py`. Agent-loop standalones: `listen.py` (wraps `stt.stt()` — one-shot record-until-silence, transcript on stdout, `--markers` + exit-code contract) and `speak.py` (reuses `tts.py` synth with `--voice` override, prefers `tts_region`) pair as a Bash speak→listen loop when the MCP server isn't connected in a session.

**Import pattern**: `import state` then `state.X = val`. `CONFIG` dict imported by name (mutated in-place).

### STT Pipeline

1. **Streaming** (default): Persistent WebSocket, real-time VAD + energy gating. Falls back to REST.
2. **VAD+REST**: Record locally, upload WAV to Azure REST API.
3. **Whisper**: Local faster-whisper model (offline, ~150MB download).
4. **Fixed**: Fixed duration recording (last-resort).

### TTS Pipeline

Text → SSML with prosody → Azure REST (streaming) → player process (aplay/pw-cat/ffplay) → live subtitles + VU meter. Pre-warmed player eliminates startup latency.

### Full-Duplex Talk

Overlaps TTS playback and STT recording. Resets energy threshold to 300 post-TTS. Falls back to half-duplex for speakers.

### Audio Routing

- **Full-duplex** (headphones): Simultaneous record + play, auto-detected via PipeWire.
- **Half-duplex** (speakers): Sequential TTS then STT.
- **Echo cancellation** (experimental): PipeWire AEC nodes.

## Source Files

| File | Lines | Purpose |
|------|------:|---------|
| `state.py` | 318 | Constants, CONFIG dict, mutable globals |
| `audio.py` | 734 | Audio I/O, PipeWire, chimes, VAD, pre-warming |
| `stt.py` | 683 | STT backends (streaming WS, VAD+REST, Whisper, fixed) |
| `speech_tts.py` | 1058 | TTS, multi_speak, talk_fullduplex, get_voices |
| `mcp_speech.py` | 997 | MCP tool schemas, JSON-RPC, stdio transport |
| `speech.py` | 104 | Standalone: mic → transcribe → clipboard |
| `tts.py` | 82 | Standalone: text → Azure TTS → speaker |
| `voice_chat.py` | 155 | Standalone: voice chat companion |

## MCP Tools

### `listen` — Record + transcribe (STT only)
`seconds`, `mode`, `silence_timeout`, `vad_aggressiveness`, `energy_multiplier`

### `speak` — One-way TTS
`text` (required), `quality`, `voice`, `speed`, `pitch`, `volume`, `subtitle_color`, `output_file`

### `talk` — Speak then listen (primary conversation tool)
`text` (required), `quality`, `voice`, `speed`, `pitch`, `volume`, `seconds`, `mode`, `silence_timeout`, `subtitle_color`

### `converse` — Listen-only with context
`seconds`, `mode`, `silence_timeout`

### `multi_speak` — Multi-voice TTS (parallel requests, sequential playback)
`segments` (required: `[{text, voice, subtitle_color}]`), `quality`, `output_file`

### `multi_speak_stream` — Multi-voice in single SSML request
`segments` (required: `[{text, voice}]`), `quality`, `output_file`

### Audio File Saving
All TTS tools (`speak`, `multi_speak`, `multi_speak_stream`) accept an `output_file` parameter. When set, audio plays AND saves to disk. Supports `.mp3` (via ffmpeg) and `.wav`. The standalone `tts.py` also supports `-o file.mp3` and `-s` (silent/save-only).

### `configure` — View/change settings at runtime
### `get_voices` — List Azure voices
### `pause` / `resume` — Pause/resume audio

## Configuration

Config loaded from env vars → `~/.config/speech-to-cli/config.json`. All changeable at runtime via `configure`.

**Credentials**: `key`/`AZURE_SPEECH_KEY`, `region`/`AZURE_SPEECH_REGION`, `tts_region`, `tts_key`
**Voice**: `voice` (HD default: `en-US-Ava:DragonHDLatestNeural`), `fast_voice` (`en-US-AvaNeural`)
**Audio**: `player`, `recorder`, `mic_source`, `speaker_sink`
**Timing**: `silence_timeout` (3.0s), `talk_silence_timeout` (4.0s), `no_speech_timeout` (7.0s), `max_record_seconds` (120), `energy_multiplier` (2.5), `end_word` ("over")
**Mode**: `half_duplex` (auto), `enable_pause` (true)
**Chimes**: `chime_ready` (true), `chime_processing`, `chime_speak`, `chime_done`, `chime_hum`, `chime_barge_in` (true)
**UI**: `visual_indicator`, `live_subtitles`, `subtitle_color_user`, `subtitle_color_tts`, `vu_meter`
**Experimental**: `enable_echo_cancel`, `enable_barge_in`, `barge_in_frames` (3), `barge_in_silence` (1.0), `debug`
**Standalone/LLM**: `llm_provider`, `llm_model`, `llm_api_key`, `conversation_mode`, `dictation_mode`, `continuous_dictation`, `read_notifications`

## Key Implementation Details

- TTS max 5000 chars/segment. WS cached 540s. Noise floor cached 120s.
- Post-TTS threshold reset to 300. Pre-warmed recorder/player/WS/HTTP.
- End word "over" stops recording immediately. SSML safety via `_sanitize_ssml_attr()`.
- Connection pooling saves ~150ms/call. numpy RMS fast path ~5-10x.

## Version Management

Version in 3 files: `mcp_speech.py`, `gemini-extension.json`, `CLAUDE.md`. Use `./bump-version.sh X.Y.Z` or `/bump-version` skill.

## Hooks

- **PreToolUse**: Blocks direct edits to config.json (use `configure` tool).
- **PostToolUse**: Warns about version sync when editing version files.
