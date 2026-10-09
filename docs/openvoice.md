# OpenVoice — local push-to-talk dictation for Windows

> Talk instead of typing. Hold a key or tap a key, speak, release — the text lands in
> whatever window has focus, with optional auto-Enter. Fully local: the speech model runs
> on this machine. No cloud, no API key, no subscription.
>
> Product name: **OpenVoice**. Data lives in `%APPDATA%\OpenVoice`; a pre-rebrand
> `%APPDATA%\RockyDictate` directory keeps being used when it already exists, so an
> upgrade never orphans existing settings.

---

## 1. What OpenVoice is

OpenVoice is a Windows tray application that turns speech into text in any focused
application. It is modelled on WisprFlow's interaction — a global hold-to-talk key, a
small floating indicator, and automatic insertion — but rebuilt entirely offline.

Three properties define it:

| property | how it is delivered |
|---|---|
| **Always available** | Global hotkeys work from any application; a tray icon is the only chrome. |
| **Non-disruptive** | The floating pill is click-through and never takes focus; the clipboard is restored after every paste. |
| **Fully local** | faster-whisper runs on the local CPU or GPU. Audio never leaves the machine. |

---

## 2. At a glance

| | |
|---|---|
| **Speech engine** | faster-whisper (CTranslate2), auto-selected: `large-v3` on CUDA; multilingual `small` on CPU (`small.en` when English is forced) |
| **Languages** | Automatic detection per dictation (~99 languages); per-language post-processing for en, zh, ja, th, vi, id, fr, es and more |
| **Interaction** | Hold `Right Alt` to talk, or tap `F9` to toggle; `Esc` cancels |
| **Indicator** | A 200×44 px hand-painted dark capsule with 7 voice-reactive bars, 3 travelling dots, and a red error flash |
| **Insertion** | Clipboard paste (Ctrl+V) into the focused window, clipboard restored ~300 ms later if unchanged |
| **Text post-processing** | Rule-based inverse text normalisation (numbers, currency, percent, ranges, spoken punctuation) |
| **Personal vocabulary** | Prompt biasing with your own term list, plus a CorrectionLearner that learns from your own edits |
| **Storage** | `%APPDATA%\OpenVoice\` — `config.json`, `learned_terms.json`, `history\` (WAV + text), and a bounded log |
| **UI framework** | PySide6 6.x (Qt 6) — widget-free pill painted with `QPainter` at 60 Hz; native tray menu |
| **Platform** | Windows (selective hotkey suppression and the Win32 clipboard path are Windows-only by design) |

---

## 3. Quick start

```
cd openvoice          # the cloned repository
run_dictate.bat
```

`run_dictate.bat` launches with `pythonw.exe`, so there is no console window. To launch
from a terminal with visible output:

```
.venv\Scripts\python.exe app.py
```

| action | default key |
|---|---|
| Hold to talk (release to insert) | `Right Alt` |
| Tap once to start, tap again to stop | `F9` |
| Throw the recording away | `Esc` while recording |
| Toggle recording | left-click the tray icon |

---

## 4. How it works

### 4.1 Architecture

```
app.py          entry point: QApplication, Bus, wiring, single-instance lock, record lifecycle
bus.py          Bus(QObject) — the only cross-module channel (Qt signals)
config.py       Config — JSON at %APPDATA%\OpenVoice\config.json
audio.py        Recorder (sounddevice) + signal conditioning (pure functions)
transcribe.py   Transcriber — faster-whisper, blocking transcribe() called off-thread
hotkeys.py      HotkeyStateMachine (pure) + HotkeyManager (pynput listener -> bus)
injector.py     insert_text() — clipboard+Ctrl+V into the focused window, optional Enter
formatting.py   normalise() — inverse text normalisation
vocabulary.py   hotword shaping, term mining, CorrectionLearner
history.py      last N dictations as WAV + text
indicator.py    Indicator(QWidget) — the floating pill, QPainter at 60 Hz
tray.py         Tray(QSystemTrayIcon) — settings surface
tests/          pytest suite, negative controls, SAPI end-to-end STT harness
tools/          mic_check, e2e_stt, benchmark, shoot_indicator, make_icon, setup_login
```

Dependency direction is one-way: `app.py` imports everything; every module imports only
`bus`, `config` and stdlib/third-party. `indicator.py` and `tray.py` never import each
other.

### 4.2 The bus contract (`bus.py`)

Every cross-module message travels over one `Bus(QObject)`, instantiated once in `app.py`
and passed by constructor injection. There is no module-level global.

```python
class Bus(QObject):
    state_changed       = Signal(str)   # "idle" | "recording" | "transcribing" | "error"
    audio_level         = Signal(float) # 0.0 .. 1.0, emitted ~60 Hz while recording
    recording_started   = Signal()
    recording_stopped   = Signal()
    transcription_done  = Signal(str)   # final text; "" when nothing was heard
    transcription_error = Signal(str)   # human-readable
    transcription_rejected = Signal(str)# the engine refused to guess (hallucination gate)
    config_changed      = Signal(str)   # config key name
```

### 4.3 The dictation lifecycle

1. **Start** — hold key down, or toggle key tap. The state machine emits `"start"`; the
   recorder opens its stream and begins buffering; `bus.set_state("recording")` fades the
   pill in.
2. **Record** — PortAudio callbacks append copied blocks; each block is sanitised (NaN/inf
   replaced, clipped to [-1, 1]); a throttled RMS level (≈60 Hz) drives the pill. Windows
   key auto-repeat is filtered by the hotkey state machine, so holding the key does not
   re-trigger.
3. **Stop** — key release (hold mode) or second tap (toggle mode). The stream stops, the
   captured audio is high-passed at 80 Hz, resampled to 16 kHz, silence-trimmed with a
   self-calibrating threshold, and level-normalised to ≈ −20 dBFS RMS.
4. **Transcribe** — on a `dictate-decode` worker thread so the UI never blocks. The pill
   switches to its travelling-dots state.
5. **Gate** — the engine's own confidence decides whether the result is trustworthy
   (see §8.5). A rejected transcript becomes a brief red flash instead of pasted text.
6. **Insert** — the transcript is normalised, then pasted into the focused window through
   the clipboard. The previous clipboard contents are restored ~300 ms later, and only if
   the clipboard still holds our payload. Optional Enter follows a configurable delay.
7. **Idle** — the pill fades out; the app returns to `idle`.

### 4.4 Threading model

| thread | owns |
|---|---|
| Qt main thread | UI, tray, indicator, clipboard insertion (Qt's clipboard has main-thread affinity, so insertion deliberately happens here) |
| `dictate-decode` worker | one per utterance; runs faster-whisper |
| `dictate-warmup` worker | loads the model once at startup and decodes one silent second so the first dictation is not slow |
| pynput listener thread | global hotkeys, owned by `HotkeyManager` |

Recording is refused while a previous utterance is still transcribing (`_busy` guard), so
two decodes never overlap.

### 4.5 Single instance

A `QLocalServer` named `openvoice.single_instance` is the lock. A second launch probes
the socket, finds it taken, and exits with a message pointing at the tray icon. There is
no dialog and no focus stealing.

### 4.6 Logging

`pythonw` has no console, so everything goes to
`%APPDATA%\OpenVoice\openvoice.log`. The file is deleted and restarted once it
exceeds 1 MB, so it can never grow without bound.

---

## 5. The user interface

OpenVoice has exactly two surfaces: the **tray menu** and the **floating pill**.

### 5.1 Tray menu

The tray icon (a white microphone on a dark rounded square) opens a menu whose check
states are refreshed every time it is shown, so values changed elsewhere are reflected
without rebuilding the menu.

```
Idle                          (disabled status line)
─────────────────────────────
Recording mode  ▸             Hold to talk / Tap to toggle / Both
  [ ] Auto-enter after paste
  [ ] Append a space after text
─────────────────────────────
Language        ▸             Auto-detect / English / Chinese (Mandarin) / … (24 more)
Microphone      ▸             System default / each input device
Custom vocabulary…            multi-line editor
─────────────────────────────
  [x] Show the indicator
  [ ] Start with Windows
─────────────────────────────
Quit
```

| item | effect |
|---|---|
| Status line | Mirrors the bus state: `Idle`, `Recording…`, `Transcribing…`, `Error`. Also drives the tooltip, `OpenVoice — <state>`, which gains ` · <language>` once a language has been detected. |
| Recording mode | Exclusive choice: `hold`, `toggle`, or `both`. |
| Language | Exclusive choice: `Auto-detect` (default) or a forced language code from a 25-entry list. Auto-detection is two-pass (see §8.1) and per dictation. |
| Auto-enter after paste | Press Enter after inserting; the pill shows a `↵` glyph while armed. Risky in chat apps — off by default. |
| Append a space after text | Adds a trailing space, for dictating consecutive sentences. |
| Microphone | Enumerated live from `sounddevice`; selecting one reopens the stream. |
| Custom vocabulary… | Edits the prompt-biasing term list. Empty means no biasing prompt at all — no vocabulary ships with the app. |
| Show the indicator | Hides the pill entirely when unchecked. |
| Start with Windows | Creates/removes a Startup-folder shortcut; the shortcut on disk is the source of truth, the config key is its cached mirror. |
| Quit | Cancels any recording, stops the hotkey listener, closes the IPC server, exits. |

Left-clicking the tray icon toggles recording without opening the menu.

### 5.2 The pill — anatomy

The indicator is a frameless, always-on-top, click-through capsule
(`WindowTransparentForInput | WindowDoesNotAcceptFocus | Tool`), painted entirely by hand
in `paintEvent` — no widget styling and no child widgets.

| element | value (logical px) |
|---|---|
| Pill | 200 × 44, corner radius 22 (full capsule) |
| Window | 208 × 52 (4 px margin for anti-aliasing and scale headroom) |
| Bars | 7, each 4 wide, 4 apart; group width 52 |
| Bar heights | 4 (silent) → 24 (loud) |
| Dots (transcribing) | 3 circles, radius 2.6, spanning 76 px |
| Placement | Bottom-centre of the primary screen's available geometry, 72 px above its bottom edge |
| Auto-enter glyph | `↵` at the pill's right edge, 18 × 20 box, 13 px DemiBold |

The pill is positioned once when it appears; it does not follow the cursor.

### 5.3 The pill — states

| state | appearance |
|---|---|
| **idle** | Hidden. The 60 Hz timer is stopped, so an idle pill costs zero frames. |
| **recording** | Capsule fades + scales in (150 ms, ease-out cubic, scale 0.96 → 1.0). 7 rounded bars ride the smoothed voice level. While the mic is quiet, a subtle breathing pulse keeps the row alive. |
| **transcribing** | Bars crossfade into 3 dim travelling dots over 160 ms; the dots drift across a 76 px span, brightest at the centre of their travel. |
| **error / no speech** | Red-tinted capsule and border: fade in 60 ms (smoothstep), hold 300 ms, fade out 180 ms — a 540 ms flash, then hide. The bars collapse to minimum so a soft red pulse reads as "nothing was caught". |

Verified renders from `tools/shoot_indicator.py` live in `tools/shots/`
(`pill_recording.png`, `pill_transcribing.png`, `pill_error.png`, and their
`_on_background` variants for contrast checking).

---

## 6. The animation system

All motion is hand-rolled in `Indicator._tick()`, driven by a `Qt.PreciseTimer` at
`FRAME_MS = 1000/60 ≈ 16.67 ms`. Frame deltas are clamped (a delta > 100 ms counts as
100 ms) so a stalled frame cannot jump the animation.

### 6.1 Transition timings

| transition | duration | easing | transform |
|---|---|---|---|
| Show | 150 ms | ease-out cubic `1 − (1−p)³` | alpha 0 → 1, scale 0.96 → 1.0 |
| Hide | 120 ms | linear | alpha 1 → 0, scale 1.0 → 0.985 |
| Error in | 60 ms | smoothstep `p²(3−2p)` | alpha 0 → 1, red tint 0 → 1 |
| Error hold | 300 ms | — | full red |
| Error out | 180 ms | linear | alpha → 0 |
| Bars → dots | 160 ms | linear blend | bars fade out as dots fade in |
| Dots → bars | 120 ms | linear blend | reverse |

The hide shrink (0.985) is deliberately smaller than the show grow (0.96): the pill
settles out gently instead of shrinking away.

### 6.2 The level model (why the bars feel alive)

Raw RMS arrives at ~60 Hz. Every smoothing decision lives in the indicator:

1. **One envelope, fast attack / slow release.** `ENV_ATTACK = 0.60` per frame makes a
   spoken syllable snap up almost instantly; `ENV_RELEASE = 0.06` gives a ~270 ms tail so
   it dies away instead of flickering.
2. **A 48-frame history ring.** Each envelope value is pushed into a fixed ring buffer.
3. **Per-bar delay.** Bar *i* reads the ring `DELAYS = (0, 2, 4, 6, 8, 10, 12)` frames
   behind the head. A change in level therefore *travels across* the pill left-to-right
   rather than moving all seven bars in lockstep.
4. **Per-bar gain arch.** `GAINS = (0.80, 0.92, 1.04, 1.12, 1.04, 0.90, 0.78)` — the
   middle bar is the tallest for a given level, so the group reads as a single organic
   shape.
5. **Per-bar attack/release.** Each bar has its own coefficients
   (attack 0.66–0.72, release 0.34–0.40), so the row never looks mechanically rigid.
6. **Breathing.** When recording and a bar's value is below 0.10, a small sine offset
   (`BREATH_DEPTH = 0.035`, rate 0.0032 rad/ms ≈ 2 s period, 0.85 rad phase per bar) keeps
   a silent microphone visibly alive — the user can see the app is listening.
7. **Visual mapping.** Bar height is `4 + 20 · v^0.75`; the power curve expands quiet
   speech and compresses loud speech so the range stays expressive.

### 6.3 Cost

Each painted frame draws a constant amount of geometry — one rounded-rect fill, one
rounded-rect stroke, 7 bars (or 3 dots), and at most one glyph. Every bar does O(1)
arithmetic against the fixed ring, so a frame is **O(1)** time and allocates nothing per
bar. The level update is likewise O(7). When the pill hides, the timer stops.

---

## 7. The color system

### 7.1 Pill palette (exact values from `indicator.py`)

| token | value | usage |
|---|---|---|
| `FILL` | `#0B0B0F` @ alpha 205 | Capsule body — near-black, slightly translucent so the desktop shows through |
| `BORDER` | `#FFFFFF` @ alpha 40 | 1 px hairline rim. The original spec called for alpha 28; measured on a near-black background that stepped the rim to only ~(36,36,39) against a (20,20,24) field, so it was raised to 40 — a hairline that still separates on dark. |
| `BAR_LIGHT` | `#F2F3F5` | Voice bars while recording |
| `BAR_DIM` | `#A8ADB5` @ alpha 200 | The 3 travelling dots while transcribing |
| `BAR_ERROR` | `#E47070` | Bars tinted red in the error flash |
| `FILL_ERROR` | `#1A0A0C` @ alpha 205 | Error capsule body — the dark fill pulled toward red |
| `BORDER_ERROR` | `#E86060` @ alpha 90 | Error rim |
| `GLYPH` | `#F2F3F5` @ alpha 110 | The `↵` auto-enter glyph — present but muted, never shouting |

Error tinting is a per-frame linear RGBA **mix** between the normal and error tokens
(`_mix`), driven by the 0→1 error envelope, so the flash is a smooth interpolation rather
than a hard colour swap. The same envelope drives body, border and bars in phase.

### 7.2 Icon palette (`tools/make_icon.py`)

| element | value |
|---|---|
| Background | Rounded square (radius 23.5 % of side), vertical linear gradient `#2B3444` → `#171C24` (top → bottom) |
| Hairline highlight | `#FFFFFF` @ alpha 24, 0.6 % of the icon size |
| Glyph | White (`#FFFFFF`) microphone: capsule 25.5 % wide × 40 % tall, U-shaped cradle (arc radius 19.5 %, stroke 7.2 %), stem (5.2 %), base bar (5.8 %), all round-capped |

Exported as PNG 64/128/256 and a manually assembled multi-size ICO containing
16, 24, 32, 48, 64, 128 and 256 px images (Qt's ICO writer emits one size per file, so the
script renders each size and merges the blobs into one container, then reads it back to
confirm every size is present). If the tray icon file is missing, a minimal drawn
microphone is used as a fallback — never a blank icon.

### 7.3 Design language

- **Neutral, low-contrast chrome.** The pill is near-black with an almost invisible rim;
  it should read as a shadow, not a UI panel.
- **One accent, used sparingly.** White-light bars when things are working; red only when
  the engine refused to guess. No third hue anywhere in the product.
- **Motion is the feedback.** State is communicated by shape and movement (bars → dots →
  red pulse) rather than by text, icons or badges.
- **No focus, no chrome.** No buttons, no close control, no drag handle — the pill can
  never interrupt what the user is doing.

---

## 8. The accuracy pipeline

### 8.1 Engine selection (`transcribe.resolve_engine`)

`model_name`, `device` and `compute_type` all default to `"auto"`:

| setting | auto behaviour |
|---|---|
| `device` | CUDA when `ctranslate2.get_cuda_device_count() > 0`, else CPU |
| `model_name` | `Systran/faster-whisper-large-v3` on CUDA; `Systran/faster-whisper-small` (multilingual) on CPU, or `small.en` when the language is exactly English |
| `compute_type` | `float16` on CUDA; `int8` on CPU |

The choice of `large-v3` over the turbo variant is deliberate: on the HuggingFace Open ASR
Leaderboard English short-form set the average WER is turbo 7.83 vs large-v3 7.44, and the
gap widens on noisy/accented speech (VoxPopuli: 11.87 vs 9.54). The GPU is what makes a
model this size practical at all — turbo at beam 5 measured RTF 0.07 on CUDA against 2.28
on CPU. If the preferred engine cannot load,
`load()` falls back to the multilingual `small` model on CPU, then to `small.en`, and logs
the original failure rather than dying. Models load with `local_files_only=True` so a cached
model never hits the network; a model that is genuinely absent is downloaded only through
`tools/fetch_model.py`, deliberately and by hand.

**Language selection is a pre-pass, not a guess after the fact.** When `language` is `auto`,
the transcriber first calls `WhisperModel.detect_language()` (one encoder pass, its own
confidence and probability), then decodes with that language pinned. This ordering matters:
faster-whisper passes `hotwords` as prior decoder context, and an English prompt demonstrably
steers a non-English dictation into English *even when detection is correct*. Knowing the
language first lets the prompt, the hallucination list and the post-processing all be chosen
for the right language. A low-confidence detection is logged and still used; a forced language
skips detection entirely; an unsupported code degrades to auto rather than raising.

CUDA DLLs shipped via pip (`nvidia-cublas`, `nvidia-cudnn`) are added to the DLL search
path at load time, with the handles kept referenced for the process lifetime — otherwise
Windows unloads the directories again.

### 8.2 Audio conditioning (`audio.py`)

Order of operations on stop, all pure functions except the recorder:

1. **Sanitise** — every captured block is NaN/inf-replaced and clipped to [-1, 1] in the
   audio callback. (This machine's driver emitted samples around 10²⁸ under loud input.)
2. **High-pass at 80 Hz** — a 2nd-order Butterworth biquad removes DC and room rumble.
   This is the *only* processing allowed before the recogniser: speech enhancement was
   explicitly rejected because MetricGAN+ raised WER in all 40 tested Whisper
   configurations, and Whisper's mel front end is trained on unprocessed audio.
3. **Resample to 16 kHz** — captured at the device's native rate (PortAudio's resampler is
   not band-limited and can lose the top of the speech band), then band-limited polyphase
   resampling with scipy when available and a numpy polyphase fallback otherwise.
4. **Trim silence** — 20 ms frames / 10 ms hop; the kept span runs from the first to the
   last frame above threshold, plus 150 ms padding. With `adaptive_vad` (default), the
   threshold is derived from the clip's own 20th-percentile noise floor and can only ever
   be *more permissive* than the −40 dBFS absolute default, never more aggressive.
5. **Normalise level** — makeup gain toward −20 dBFS RMS, capped by a −1 dBFS peak ceiling
   and +30 dB maximum. This is what recovers the first word on a low-gain array.

### 8.3 Pre-flight microphone health

`tools/mic_check.py` measures peak/RMS level, SNR, clipping, and band levels relative to
the 300–1000 Hz band, then prints a verdict with reasons and the exact Windows settings
path. The in-app health check (`app._check_microphone_health`) samples the first three
dictations and warns once via a tray notification if the median 2–4 kHz roll-off is
abnormally steep or clipping is present.

The critical lesson encoded here: a *share* of total energy must never be used for this
verdict. Room rumble and DC dominate the total, so energy share collapses toward zero for
perfectly healthy speech. The check uses relative band levels and an SNR estimate instead.

### 8.4 Prompt biasing (`vocabulary.py`)

`DEFAULT_VOCABULARY` is empty on purpose: the words worth rescuing are the speaker's own,
so someone else's project names would only be noise in the prompt. Terms come from the
tray editor or from the CorrectionLearner. Keep any list short: a 29-term version measured
48.1 % WER on a real dictation while a 10-term list scored 37.0 % on the same audio — a
long prompt skews the language model and damages *ordinary* words.

For **English** the list is wrapped in ordinary prose rather than ending as a bare list:

> `The speaker may mention <terms> while dictating ordinary sentences about work, budgets and projects.`

Measured: the prose framing scored 80.1 % vs 78.9 % for a list-ending prompt, and was the
only form that improved the hard technical sentence.

For **every other language** the prompt is a bare comma-separated list with no English prose:
the framing sentence itself is English prior context, and the reproduction in the research
showed exactly that pulling a correctly-detected non-English dictation back into English.
Tail ordering and the token budget are unchanged.

**Ordering matters more than membership.** faster-whisper's prompt carries roughly its
tail; anything past the cap is silently dropped. The terms are therefore sorted
least-to-most important and trimmed from the front until they fit a 150-token budget, so
the most important terms *survive at the tail*. Short, ordinary-looking terms rank highest
because they are the ones the language model substitutes with common words — a short
project name came back as an ordinary word while the long distinctive names survived.

Fuzzy post-hoc correction was measured and rejected: a threshold low enough to fix
mishearings also corrupts correct words. Biasing at decode time is safe; string munging
after the fact is not.

### 8.5 The hallucination gate

Whisper hallucinates on non-speech (Barański 2025 measured 40.3 % of 301,317 non-speech
clips producing text). OpenVoice rejects rather than guesses. A transcript is refused when:

| check | threshold |
|---|---|
| silent / empty audio | peak < 1e-4, or clip shorter than 0.30 s |
| known filler phrases | whole-transcript match against the English list plus per-language lists (Chinese `谢谢观看`, Japanese `ご視聴ありがとうございました`, Thai `ขอบคุณที่รับชม`, Vietnamese `cảm ơn các bạn đã xem video này`, Indonesian `terima kasih telah menonton`, French `merci d'avoir regardé cette vidéo`, Spanish `gracias por ver el video`, plus the known Chinese phantom sentence), matched after case-folding, whitespace removal and punctuation stripping |
| repetitive output | compression ratio ≥ 2.2 |
| low confidence | average logprob < −1.0 |
| no speech | no-speech probability ≥ 0.5 **and** avg logprob < −0.5 |
| guessed decode | temperature > 0 **and** avg logprob < −0.5 |

A rejected transcript produces the red pill flash and inserts nothing. Every rejection
reason is a human-readable string that reaches the log.

### 8.6 Inverse text normalisation (`formatting.py`)

Rule-based on purpose: NeMo's ITN needs Pynini (conda-only on Windows), and neural ITN has
an extremely low tolerance to unrecoverable errors. Two rules govern everything:

1. **Never invent content.** If a pattern is not clearly present, return the text
   unchanged.
2. **Be conservative with small counts in prose.** "I have two cats" stays words; counts
   become digits only with a unit, currency, percent, decimal, ordinal, fraction, math
   operator, range, or a value ≥ 10000.

It handles: spoken punctuation ("comma", "new line", …) with de-duplication against
whisper's own punctuation; numbers and ordinals; fractions; percentages; currency;
units (mm, kg, GB, kHz, °C…); ranges ("from 20 to 30" → "20-30"); math ("plus", "squared"
→ `²`). Prefixing a sentence with `literal ` bypasses normalisation entirely. The function
is idempotent: running it twice changes nothing the second time.

**Language routing is mandatory, not an optimisation.** Three branches exist:

* **Chinese / Japanese / Thai / Cantonese** are written without spaces, so the space-token
  pipeline cannot see their word boundaries — and worse, it would insert or delete spaces the
  writing system does not use. These take a character-level branch only: whitespace around
  CJK punctuation and between CJK characters is removed, and a half-width `, . ! ?` directly
  after a CJK character becomes its full-width form. No number or currency rules run.
* **English** keeps the full original pipeline, unchanged and byte-identical.
* **French / Spanish / Indonesian / Vietnamese** get their own spoken punctuation words,
  digit-based decimal handling (`3 virgule 14` → `3,14`, `3 coma 14` → `3,14`, `3 koma 14`,
  `3 phẩy 14`), percent phrases (`12 pour cent` → `12 %`, `12 por ciento` → `12%`,
  `12 persen` → `12%`, `12 phần trăm` → `12%`) and currency words (`25 euros` → `25 €`,
  `25000 rupiah` → `25000 Rp`, `25 đồng` → `25 ₫`). The **English word tables never run** on
  these languages — `million` is an English scale word and would rewrite French
  "deux million d'euros" as "deux 1,000,000 d'euros". A test asserts exactly that pair, in
  both directions, so removing the gate fails the suite.
* Any other language gets whitespace tidy-up only, the safest possible pass.

Full spoken-number conversion for the non-English Latin languages is deliberately **not**
implemented yet (French `quatre-vingt-dix` = 90 is its own project); the digit-based rules
above cover the common cases without risking prose.

### 8.7 Learning from corrections (`vocabulary.CorrectionLearner`)

`CorrectionLearner.observe(inserted, final)` diffs the inserted transcript against what the
user kept, considers only `replace` opcodes, and records a term only when:

- the corrected word looks like a term (one word, alphabetic/alphanumeric, not a common
  English stopword), **and**
- the misheard word is a *near miss* (shared prefix ≥ 3 characters, or a case-sensitive
  similarity ratio ≥ 0.5).

Terms become active after `learn_threshold` (default 2) sightings, so a one-off typo can
never rewrite text. Storage is `learned_terms.json` next to the config file
(`terms` counts + `wrong → right` replacements), and `apply()` rewrites text at word
boundaries, longest key first, matching the casing of the matched text.

`suggest_terms()` can mine a folder tree (breadth-first, depth ≤ 3, ≤ 300 directories,
noise directories skipped) for candidate proper nouns to add by hand.

---

## 9. Settings reference

All values live in `%APPDATA%\OpenVoice\config.json`; the tray exposes the common ones.
The file is written atomically (temp file + `os.replace`), and an unreadable file is
quarantined as `config.json.bak-<timestamp>` and replaced with defaults — the app never
dies on a corrupt config.

| key | default | meaning |
|---|---|---|
| `mode` | `"both"` | `"hold"`, `"toggle"`, or `"both"` |
| `hold_key` | `"alt_r"` | hold-to-talk key |
| `toggle_key` | `"f9"` | tap on / tap off |
| `cancel_key` | `"esc"` | discard the current recording |
| `auto_enter` | `false` | press Enter after inserting (the pill shows `↵`) |
| `auto_enter_delay_ms` | `120` | pause between paste and Enter |
| `trailing_space` | `false` | append a space to the transcript |
| `model_name` | `"auto"` | `auto` = `large-v3` on CUDA, multilingual `small` (`small.en` for English) on CPU; any faster-whisper repo id works |
| `device` | `"auto"` | `auto` prefers CUDA, falls back to CPU; `cpu`/`cuda` to force |
| `compute_type` | `"auto"` | `float16` on GPU, `int8` on CPU |
| `beam_size` | `5` | beam-search width |
| `vad_filter` | `false` | whisper's own VAD (measured to matter not at all here; the adaptive trim does the work) |
| `condition_on_previous` | `false` | condition decoding on previous text |
| `language` | `"auto"` | `auto` detects per dictation via a two-pass pre-check (§8.1); a code (`en`, `zh`, `ja`, `th`, `vi`, `id`, `fr`, `es`, …) forces one |
| `detect_per_segment` | `false` | pass `multilingual=True` (re-detect every segment) for code-switching |
| `adaptive_vad` | `true` | derive the silence threshold from the clip's own noise floor |
| `normalize_audio` | `true` | makeup gain so the recogniser sees a consistent level |
| `keep_mic_warm` | `true` | hold the input stream open so a sleeping mic array is awake when you start talking |
| `mic_device` | `null` | sounddevice index or name substring; null = system default |
| `custom_vocabulary` | `""` | terms fed as the biasing prompt; empty = no biasing |
| `initial_prompt` | `""` | extra text prepended to whisper's decode prompt (style / domain priming). It shifts phrasing style (measured: number words come back spelled out), so the plain decode stays the default; set it only when you want that |
| `learn_threshold` | `2` | sightings before a learned term becomes active |
| `spoken_punctuation` | `true` | convert spoken punctuation words |
| `save_history` | `true` | keep the last `history_limit` dictations, audio included |
| `history_limit` | `200` | history entries kept (raised from 20 so benchmark runs can be re-scored) |
| `pill_enabled` | `true` | show the floating indicator |
| `run_at_login` | `false` | shortcut in the Windows Startup folder |
| `max_record_seconds` | `300` | hard stop |
| `mirror_input` | `false` | reserved for live partial typing (not implemented) |

`capture_native_rate` (default true) is read defensively but intentionally absent from the
defaults dict: a missing key means enabled.

---

## 10. Recent dictations

The last `history_limit` dictations are stored in `%APPDATA%\OpenVoice\history\` as a
16-bit mono WAV of *exactly* the audio the recogniser received, plus a `.txt` with what
came back. When a transcript is wrong, the pair shows whether the fault was the audio or
the model. Entries are pruned newest-first.

---

## 11. Verification

The project's evidence standard: structural checks (no exceptions, non-empty output, files
exist) are reported separately and never presented as evidence of behaviour.

| claim | check | can it fail? |
|---|---|---|
| silence trimming is correct | `tests/test_vad.py` — digital silence → empty, speech in noise → preserved and trimmed | yes; negative control mutates the threshold |
| hold/toggle/cancel logic | `tests/test_hotkeys.py` on the pure state machine (incl. the auto-repeat guard) | yes; negative control mutates the guard |
| text reaches a real window | `tools/inject_e2e.py` — inject into a real third-party window and read it back | yes |
| speech → correct text | `tools/e2e_stt.py` — SAPI speaks known sentences, real pipeline transcribes, WER computed | yes; WER must be ~0 |
| the pill looks right | `tools/shoot_indicator.py` — render every state, screenshot, inspect pixels | yes; eyes |
| config survives restarts | round-trip + corrupt-file recovery tests | yes |
| multilingual accuracy + detection | `tools/lang_bench.py` — real public speech clips per language, WER/CER vs the reference, detected-language check | yes; a known-mismatched row is scored SUSPECT and printed, not averaged away |

```
.venv\Scripts\python.exe -m pytest -q                       # unit + integration suite
.venv\Scripts\python.exe tests\negative_control_vad.py      # proves the VAD tests can fail
.venv\Scripts\python.exe tools\e2e_stt.py --vocab "Zenor"   # synthetic speech -> text, WER
.venv\Scripts\python.exe tools\mic_check.py --seconds 5     # YOUR microphone, spectrum included
.venv\Scripts\python.exe tools\test_my_voice.py             # YOUR voice, saved to tools/voice_samples
.venv\Scripts\python.exe tools\benchmark.py                 # 8-probe diagnostic benchmark
.venv\Scripts\python.exe tools\inject_e2e.py                # takes the desktop for ~3 s
.venv\Scripts\python.exe tools\shoot_indicator.py           # render the pill, look at it
.venv\Scripts\python.exe tools\lang_bench.py --fetch        # ~14 MB of public speech samples, once
.venv\Scripts\python.exe tools\lang_bench.py                # per-language WER/CER + detected language
.venv\Scripts\python.exe tools\fetch_model.py --list        # which models are cached locally
```

Measured on this machine, 2026-09-26: synthetic speech to text 7/7 sentences exact
(0.00 % WER); eight-probe benchmark on the user's own voice averaged 95.0 % across the
seven scorable probes (target 89–90 %). The real cross-application injection check is a
deliberate manual tool rather than part of the suite — it needs the OS foreground, and
anything typed while the suite runs would land inside the test window.

Measured on this machine, 2026-09-28, three real human-speech clips per language from public
research datasets (FLEURS / LibriSpeech / JSUT), through the real pipeline, `large-v3` on CUDA
with automatic detection: English 0.0 %, French 5.9 %, Spanish 0.0 %, Indonesian 7.9 %, Chinese
10.9 % CER, Japanese 3.3 % CER, Thai 0.7 % CER, Vietnamese 8.5 % WER. **24/24 clips detected
correctly; zero wrongful rejections.** Three clips per language describes the sample, not the
language — and the tool now flags the artifacts it found (a duplicate reference, a
reference-longer-than-audio row) instead of hiding them in the mean.

---

## 12. Known limits, stated honestly

- **Microphone, not model, is the usual bottleneck.** This machine's Intel array loses
  consonant energy (1–8 kHz) only while speaking — the signature of driver-level voice
  processing. With the `s` gone, "send the invoice" is heard as "and the invoice"; no model
  fixes that. The README documents the exact Windows settings to disable.
- **A cold mic costs the first syllable.** The array takes up to a second to reach full
  gain after sound starts; `keep_mic_warm` hides the ramp, but starting your first word a
  beat after pressing the key materially improves results.
- **CPU latency is ~0.3× realtime** with `small.en` (the multilingual `small` is comparable and
  runs slower than realtime on a large model); a 10 s dictation is a 3–4 s wait plus the
  detection pre-pass. The GPU path (`large-v3`, which only the GPU makes practical — turbo
  measured RTF 0.07 there) is the fix and is used automatically when CUDA is available.
- **Synthetic input cannot reach elevated windows** while the app runs unelevated, and
  apps that deliberately ignore synthetic keyboard input receive nothing.
- **Multi-monitor placement and HiDPI > 100 %** are untested; the pill uses the primary
  screen.
- **Multilingual accuracy varies by language and by register.** On the measured sample, Chinese
  was the weakest (its main failure mode is code-switching into English words, e.g.
  "photosphere" → "Potos Bay"), and Vietnamese/Indonesian names and numbers still slip. Thai
  looked excellent on clean read speech; spontaneous Thai is harder and is not in the sample.
- **Spoken-number conversion is English-only for now**; the other languages get punctuation,
  percent, currency and decimal handling only. Spoken punctuation command words exist for
  French, Spanish, Indonesian and Vietnamese.
- **`mirror_input`** (live partial text while speaking) is a reserved setting, not a
  feature.

---

## 13. File map and paths

```
<repo>\                      app.py, bus.py, config.py, audio.py, transcribe.py,
                             hotkeys.py, injector.py, formatting.py, vocabulary.py,
                             history.py, indicator.py, tray.py
<repo>\assets\               icon.ico, icon_64/128/256.png
<repo>\docs\                 design.md (original approved design), openvoice.md (this doc)
<repo>\tests\                pytest suite + negative controls
<repo>\tools\                mic_check, e2e_stt, benchmark, accuracy_bench, inject_e2e,
                             test_my_voice, vocab_test, shoot_indicator, make_icon,
                             setup_login, lang_bench (per-language accuracy on public
                             speech), fetch_model (explicit model download),
                             voice_samples\, lang_samples\, shots\
%APPDATA%\OpenVoice\         config.json, learned_terms.json, history\, openvoice.log
                             (a legacy RockyDictate directory keeps being used if present)
```

Launch: `run_dictate.bat` (no console) or `.venv\Scripts\python.exe app.py` (console).
