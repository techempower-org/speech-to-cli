# Backends: speech-to-cli (the local speech stack)

Standing rule (JP 2026-10-27, "Backends grow continually"): every project that talks to outside sources keeps this file as a tiered backlog: per row the backend, the account or key it needs, its terms caveat and its status. Lanes add one adapter per idle slot through the normal gates; every sign-up or plan change is JP's one click, batched through the ask queue. Started 2026-11-11 (money-e9).

Status words: **live** (in code and serving) · **candidate** (not started) · **evaluated** (tried, verdict recorded).

## Tier A: real API or model, in code

| Backend | What it does here | Key / account | Terms caveat | Status |
|---|---|---|---|---|
| Azure Speech Services (REST TTS/STT, WebSocket streaming STT; Dragon HD voices) | Primary cloud voice I/O; recorded output per the fleet's Dragon HD rule | Azure Speech key (vault), `config.json` | Azure Sponsorship credit covers first-party Speech; per-character / per-second billing | live (CLAUDE.md) |
| Wyoming server on familiar: Piper TTS + ONNX-ASR STT (Parakeet) | LAN fallback, or primary with `speech_backend: "local"`; `SPEECH_FORCE_OFFLINE=1` forces it | none | model licences per voice/model; no streaming partials on the server (synthesized client-side) | live (CLAUDE.md "Wyoming fallback") |
| Whisper (local) | STT backend in `stt.py` | none | model licence | live (`stt.py`) |

## Tier C: candidates

| Backend | What it would do here | Key / account | Terms caveat | Status |
|---|---|---|---|---|
| DrivenData **pasketti-word** (Qwen3-ASR-1.7B fine-tune, [HF](https://huggingface.co/drivendata/pasketti-word)) and **pasketti-phonetic** (WavLM-large fine-tune, [HF](https://huggingface.co/drivendata/pasketti-phonetic)), children's speech recognition, open weights (2026-10-23) | A children's-speech STT route for the Wyoming/local stack: word transcription for a child's reading or question, phone-level output for early-reader scoring; consumers: storyvox read-along, the kids' side of the local phone pipeline, Familiar's household voice | none (open weights) | Licence **bigcode-openrail-m** on both (use restrictions; read before any shipped feature); children's audio stays on the LAN; base-model licences (Qwen3-ASR, WavLM) apply | candidate (2026-11-11; from DrivenData's 2026-11-11 newsletter, JP-starred). First step: an eval harness on familiar comparing word error rate against Parakeet and Whisper on a small set of child read-alouds; then a `stt.py` backend behind the existing `speech_backend` switch |
