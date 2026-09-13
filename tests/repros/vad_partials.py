#!/usr/bin/env python3
"""Live partials on the Wyoming VAD route (gnome-speaks live typing, 2026-09-12).

With speech_backend=local every gnome-speaks dictation is routed to stt_vad(),
which recorded the whole utterance and asked the LAN recognizer once -- so
the badge's live transcript and live typing into the focused field, both
streaming-path features, silently did not exist on the local route. The
server has no streaming protocol (supports_transcript_streaming=False), so
stt_vad(partial_cb=...) re-transcribes the utterance so far from a side
thread every PARTIAL_INTERVAL_MS of speech, one request in flight, each with
PARTIAL_TIMEOUT.

Fake recorder (speech frames then silence), fake wyoming.transcribe (text
grows with the audio length; one length hangs past the timeout), Azure
route faked as unreachable-by-design (never contacted). No audio, no network,
no user config written.

  P1  partial_cb fires >= 2 times while recording, each hypothesis longer than
      the last, each BEFORE the final result returns
  P2  the final result is the full transcription and is unaffected by a
      partial request that hangs: total wall time < PARTIAL_TIMEOUT + slack
      (the hung request is abandoned, not awaited)
  P3  one request in flight: partial requests <= snapshots submitted, and no
      partial is delivered after stt_vad returned
  P4  partial_cb=None (the old contract): exactly one transcribe call, the
      final -- no side thread, no extra requests
  P5  Azure route (skip_azure False): partial_cb never called and the fake
      Azure session is used for the final -- partials are Wyoming-only
  P6  a partial failure never marks the local server down

exit 0 = all hold; 1 = a check failed; 2 = setup failure.
"""
import io
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("SPEECH_FORCE_OFFLINE", "0")

import state      # noqa: E402
import audio      # noqa: E402
import stt        # noqa: E402
import wyoming    # noqa: E402

FAILS = []


def check(label, ok, msg):
    print(f"  {'OK  ' if ok else 'FAIL'} ({label}) {msg}")
    if not ok:
        FAILS.append(label)


class FakeRecorder:
    """stdout yields `speech` frames of loud audio, then silence forever."""

    def __init__(self, speech_frames, fb):
        self._n = speech_frames
        self._fb = fb
        self.reads = 0
        self.stdout = self

    def read(self, n):
        self.reads += 1
        time.sleep(0.002)          # a little real time, so partials interleave
        if self.reads <= self._n:
            return (b"\x00\x40\x00\xc0" * (self._fb // 4))[:n]
        return b"\x00" * n

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


def setup():
    state.HAS_VAD = False            # energy-only gate, deterministic
    state.FRAME_MS = 20
    state.FRAME_BYTES = 640
    state.SILENCE_TIMEOUT = 0.2
    state.NO_SPEECH_TIMEOUT = 2.0
    state.MIN_SPEECH_DURATION = 0.05
    state.PARTIAL_INTERVAL_MS = 100  # 5 frames
    state.PARTIAL_TIMEOUT = 0.4
    audio.calibrate_noise = lambda proc, n_frames=0: (1.0, [])
    stt.calibrate_noise = audio.calibrate_noise
    stt.play_chime = lambda *a, **k: None
    stt.play_processing = lambda *a, **k: None
    stt.send_progress = lambda *a, **k: None
    stt.register_proc = lambda p: None
    stt.unregister_proc = lambda p: None
    state.CONFIG.update({"wyoming_host": "fake.test", "wyoming_stt_port": 1,
                         "speech_backend": "local", "key": "test-key",
                         "region": "westus", "language": "en-US"})
    wyoming.mark_local_up()
    wyoming.mark_azure_up()


WORDS = "the quick brown fox jumps over the lazy dog".split()


def make_fake_transcribe(log, hang_at=None):
    lock = threading.Lock()
    inflight = [0]
    peak = [0]

    def fake(host, port, pcm, rate=16000, width=2, channels=1, timeout=20.0):
        with lock:
            inflight[0] += 1
            peak[0] = max(peak[0], inflight[0])
        try:
            n = len(pcm) // 640
            if hang_at is not None and hang_at[0] <= n < hang_at[1]:
                time.sleep(timeout + 0.05)      # the server never answers
                raise TimeoutError("timed out")
            words = WORDS[:max(1, min(len(WORDS), n // 4))]
            text = " ".join(words)
            log.append((time.monotonic(), n, text))
            return text
        finally:
            with lock:
                inflight[0] -= 1
    return fake, peak


def run(partial_cb, fake, proc):
    stt._take_prewarmed_rec = lambda: proc
    wyoming.transcribe = fake
    stt.wyoming.transcribe = fake
    t0 = time.monotonic()
    result = stt.stt_vad(max_seconds=5, stop_when=None, partial_cb=partial_cb)
    return result, time.monotonic() - t0


def main():
    setup()
    if "partial_cb" not in stt.stt_vad.__code__.co_varnames:
        print("FAIL: stt_vad has no partial_cb seam -- live partials do not exist on the local route")
        return 1
    if wyoming.skip_azure() is not True:
        print("!! SETUP FAILURE: skip_azure() is not True under speech_backend=local"); return 2

    # ---- P1-P3, P6: Wyoming route, partials on, one length hangs -------------
    calls, partials = [], []
    fake, peak = make_fake_transcribe(calls, hang_at=(20, 30))
    proc = FakeRecorder(speech_frames=60, fb=640)        # 1.2 s of speech
    returned = [None]
    def cb(text):
        partials.append((time.monotonic(), text, returned[0]))
    result, wall = run(cb, fake, proc)
    returned[0] = time.monotonic()
    time.sleep(0.5)                                        # anything late?
    texts = [p[1] for p in partials]
    growing = all(len(texts[i]) < len(texts[i + 1]) for i in range(len(texts) - 1))
    before_final = all(p[2] is None for p in partials)
    check("P1", len(partials) >= 2 and growing and before_final,
          f"partials={texts} (>=2, growing, all before the final returned)")
    check("P2", result.get("text") == " ".join(WORDS) and result.get("engine") == "wyoming"
          and wall < state.PARTIAL_TIMEOUT + 1.5,
          f"final={result.get('text')!r} engine={result.get('engine')} wall={wall:.2f}s "
          f"(hung partial abandoned, not awaited)")
    late = [p for p in partials if p[2] is not None]
    check("P3", peak[0] <= 2 and not late,
          f"peak concurrent requests={peak[0]} (partial + final at most), late partials={len(late)}")
    check("P6", not wyoming.local_down(), "a hung partial did not mark the local server down")

    # ---- P4: old contract, no partials ------------------------------------------
    calls, partials = [], []
    fake, _ = make_fake_transcribe(calls)
    result, _ = run(None, fake, FakeRecorder(speech_frames=40, fb=640))
    check("P4", len(calls) == 1 and result.get("text"),
          f"partial_cb=None -> {len(calls)} transcribe call(s), final={result.get('text')!r}")

    # ---- P5: Azure route -> no partials, Azure fake used --------------------------
    state.CONFIG["speech_backend"] = "azure"
    wyoming.mark_azure_up()
    if wyoming.skip_azure():
        print("!! SETUP FAILURE: skip_azure() still True with speech_backend=azure"); return 2
    posted = []
    class Resp:
        status_code = 200
        def json(self): return {"RecognitionStatus": "Success", "DisplayText": "azure final"}
    class Sess:
        def post(self, url, **kw): posted.append(url); return Resp()
    stt.get_http_session = lambda: Sess()
    calls, partials = [], []
    fake, _ = make_fake_transcribe(calls)
    result, _ = run(lambda t: partials.append(t), fake, FakeRecorder(speech_frames=40, fb=640))
    check("P5", partials == [] and len(calls) == 0 and len(posted) == 1
          and result.get("text") == "azure final",
          f"azure route: partials={partials} wyoming calls={len(calls)} azure posts={len(posted)} "
          f"final={result.get('text')!r}")

    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("PASS: live partials grow during a Wyoming VAD recording, a hung partial is dropped, "
          "the final is unaffected, and the Azure route is untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
