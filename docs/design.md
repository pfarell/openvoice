# OpenVoice — local push-to-talk dictation for Windows

Status: approved 2026-09-26. Fully local: faster-whisper on this machine, no cloud, no API keys, no paid services.

## Purpose

Talk instead of typing, into whatever window has focus. WisprFlow's interaction model, rebuilt
offline: hold a key, speak, release, text appears. Optional auto-Enter. A small floating pill shows
what the app is doing.

## Architecture

```
app.py          entry point: QApplication, Bus, wiring, single-instance lock, record lifecycle
bus.py          Bus(QObject) — the only cross-module channel (Qt signals)
config.py       Config — JSON at %APPDATA%\OpenVoice\config.json
audio.py        Recorder (sounddevice) + trim_silence() pure function
transcribe.py   Transcriber — faster-whisper, blocking transcribe() called off-thread
hotkeys.py      HotkeyStateMachine (pure) + HotkeyManager (pynput listener -> bus)
injector.py     insert_text() — clipboard+Ctrl+V into the focused window, optional Enter
indicator.py    Indicator(QWidget) — the floating pill, QPainter at 60 Hz
tray.py         Tray(QSystemTrayIcon) — settings surface
tests/          pytest, negative controls, SAPI end-to-end STT harness
```

Dependency direction is one-way: `app.py` imports everything; every module imports only `bus`,
`config` and stdlib/third-party. `indicator.py` and `tray.py` never import each other.

## Bus contract (bus.py)

```python
class Bus(QObject):
    state_changed       = Signal(str)   # "idle" | "recording" | "transcribing" | "error"
    audio_level         = Signal(float) # 0.0..1.0, emitted ~60 Hz while recording
    recording_started   = Signal()
    recording_stopped   = Signal()
    transcription_done  = Signal(str)   # final text, "" when nothing was heard
    transcription_error = Signal(str)   # human-readable
    config_changed      = Signal(str)   # config key name
```

`Bus` is instantiated once in `app.py` and passed by constructor injection. No module-level global.

## Config keys (defaults in config.py)

| key | default | meaning |
|---|---|---|
| `mode` | `"both"` | `"hold"`, `"toggle"`, `"both"` |
| `hold_key` | `"alt_r"` | hold-to-talk key |
| `toggle_key` | `"f9"` | tap on / tap off |
| `cancel_key` | `"esc"` | discard the current recording |
| `auto_enter` | `false` | press Enter after inserting |
| `auto_enter_delay_ms` | `120` | wait before that Enter |
| `trailing_space` | `false` | append a space to the transcript |
| `model_name` | `"small.en"` | faster-whisper model id |
| `compute_type` | `"int8"` | CPU first; `int8_float16` after the GPU phase |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `mic_device` | `null` | sounddevice index or name substring; null = system default |
| `custom_vocabulary` | `""` | names/jargon fed as whisper `initial_prompt` |
| `mirror_input` | `false` | type as you speak (live partials) — phase 3, off by default |
| `pill_enabled` | `true` | show the floating indicator |
| `run_at_login` | `false` | shortcut in the user's Startup folder |
| `max_record_seconds` | `300` | hard stop |
| `language` | `"en"` | whisper language |

## Behaviour

1. **Start recording** — hold key down, or toggle key tap. Stream opens, level starts flowing to the pill.
2. **Record** — everything is buffered; the pill animates from the live RMS level. Key auto-repeat is ignored.
3. **Stop** — key release (hold mode) or second tap (toggle mode). Stream closes, silence is trimmed.
4. **Transcribe** — on a worker thread, so the UI never blocks. Pill shows the transcribing state.
5. **Insert** — clipboard is set to the transcript, Ctrl+V is sent, clipboard is restored ~300 ms later
   only if it still holds the transcript. Optional Enter after a configurable delay.
6. **Cancel** — Esc while recording discards the audio and inserts nothing.
7. Silence / no speech detected -> nothing is inserted, pill flashes a dim "no speech" state.

## Pill states

| state | appearance |
|---|---|
| idle | hidden |
| recording | pill fades + scales in (150 ms ease-out), 7 bars driven by the smoothed level, fast attack / slow release with per-bar phase offset |
| transcribing | bars collapse to 3 travelling dots, dimmed |
| no speech / error | brief dim red-tinted flash, then fade out |

A small muted `↵` glyph sits on the pill's right edge whenever `auto_enter` is on. Click-through:
`Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus`, always-on-top, no taskbar entry.
Positioned bottom-centre of the primary screen's available geometry.

## Verification plan (what proves what)

| claim | check | can it fail? |
|---|---|---|
| silence trimming is correct | `tests/test_vad.py`: digital silence -> empty, speech in noise -> preserved and trimmed | yes; negative control mutates the threshold |
| hold/toggle/cancel/key-repeat logic | `tests/test_hotkeys.py` on the pure state machine | yes; negative control mutates the repeat guard |
| text reaches a real focused window | inject into a real third-party window and read the value back | yes |
| speech -> correct text | `tests/test_e2e_stt.py`: Windows SAPI speaks known sentences to WAV, real pipeline transcribes, WER computed | yes; WER must be ~0 |
| the pill looks right | render every state, screenshot, inspect the pixels | yes; eyes |
| config survives restarts | round-trip + corrupted-file recovery test | yes |

Structural checks (no exceptions, non-empty output, files exist) are reported separately and are
never presented as evidence of behaviour.

## Known limits, stated up front

- `small.en` on CPU does not match WisprFlow's cloud accuracy. Expect solid results on clear speech,
  weaker on names and jargon — partially mitigated by `custom_vocabulary`.
- CPU latency is roughly 0.3-0.5x realtime: a 10 s dictation is a 3-5 s wait. GPU phase 2 targets this.
- Synthetic input cannot reach windows running elevated while this app runs unelevated.
- Apps that deliberately ignore synthetic keyboard input (some anti-cheat) will receive nothing.
- Auto-enter on is risky in chat apps — it can send a message early. Default is off.

## Build order

Serial spine (`config.py`, `bus.py`, this document) then a four-way parallel fan-out over disjoint
files (`audio.py`, `indicator.py`, `hotkeys.py`+`injector.py`, `tray.py`+launcher), then serial
integration (`app.py`), then serial verification.
