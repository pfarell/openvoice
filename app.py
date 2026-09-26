"""OpenVoice — local push-to-talk dictation for Windows.

Wiring only: this module owns the recording lifecycle and the threads, and it is
the single place where the other modules meet. Per-dictation work is O(1) state
transitions plus one worker thread per utterance; nothing here loops over a
collection.

Threading model
---------------
* Qt main thread: UI, tray, indicator, clipboard insertion (Qt's clipboard has
  main-thread affinity, so insertion deliberately happens there).
* `dictate-decode` worker: one per utterance, runs faster-whisper.
* `dictate-warmup` worker: loads the model once at startup.
* pynput listener thread: hotkeys, owned by HotkeyManager.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from audio import Recorder, measure_quality
from formatting import normalise
from vocabulary import CorrectionLearner
from bus import Bus
from config import Config, default_config_path
from history import save as save_history
from hotkeys import HotkeyManager
from indicator import Indicator
from injector import insert_text
from transcribe import Transcriber
from tray import Tray

IPC_NAME = "openvoice.single_instance"
LOG_NAME = "openvoice.log"
NO_SPEECH_MS = 650
ERROR_MS = 1600


def setup_logging(log_path: Path) -> None:
    """pythonw has no console, so keep a bounded log file or startup failures vanish."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size > 1_000_000:
            log_path.unlink()
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except Exception as exc:
        print(f"[app] file logging unavailable ({exc}); stderr only", file=sys.stderr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def single_instance_taken(name: str) -> bool:
    """True when another copy already owns the IPC name. Keeps the server alive on self."""
    probe = QLocalSocket()
    probe.connectToServer(name)
    if probe.waitForConnected(300):
        probe.disconnectFromServer()
        return True
    QLocalServer.removeServer(name)
    return False


class DictationApp:
    def __init__(self, qapp: QApplication) -> None:
        self.qapp = qapp
        self.log = logging.getLogger("app")
        self.config = Config()
        self.bus = Bus()
        self.recorder = Recorder(self.bus, self.config)
        self.transcriber = Transcriber(self.bus, self.config)
        self.hotkeys = HotkeyManager(self.bus, self.config)
        self.indicator = Indicator(self.bus, self.config)
        self.tray = Tray(
            self.bus,
            self.config,
            {
                "toggle_recording": self.toggle_recording,
                "quit": self.quit,
                "reload_mic": self.reload_mic,
            },
        )

        self._busy = False
        self._last_audio = None
        self._quality_samples: list = []
        self._warned_about_mic = False
        self._server = QLocalServer(qapp)
        self._server.listen(IPC_NAME)

        self._idle_timer = QTimer(qapp)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(lambda: self.bus.set_state("idle"))

        self._max_timer = QTimer(qapp)
        self._max_timer.setSingleShot(True)
        self._max_timer.timeout.connect(self._on_max_duration)

        self.hotkeys.action.connect(self._on_action)
        self.bus.transcription_done.connect(self._on_transcription_done)
        self.bus.transcription_error.connect(self._on_error)
        self.bus.transcription_rejected.connect(self._on_rejected)
        sys.excepthook = self._excepthook

    # ------------------------------------------------------------- lifecycle

    def run(self) -> int:
        self.hotkeys.start()
        self.recorder.prime()
        self.bus.set_state("idle")
        threading.Thread(
            target=self.transcriber.warm_up, daemon=True, name="dictate-warmup"
        ).start()
        self.log.info(
            "ready: mode=%s hold=%s toggle=%s auto_enter=%s model=%s",
            self.config.get("mode"),
            self.config.get("hold_key"),
            self.config.get("toggle_key"),
            self.config.get("auto_enter"),
            self.config.get("model_name"),
        )
        return self.qapp.exec()

    def quit(self) -> None:
        try:
            self.recorder.cancel()
            self.hotkeys.stop()
            self._server.close()
        except Exception:
            self.log.exception("shutdown cleanup failed")
        self.qapp.quit()

    def _excepthook(self, exc_type, exc, tb) -> None:
        self.log.error("unhandled exception", exc_info=(exc_type, exc, tb))
        try:
            self.tray.notify("OpenVoice", f"Unhandled error: {exc}")
        except Exception:
            pass

    # -------------------------------------------------------------- commands

    def _on_action(self, action: str) -> None:
        if action == "start":
            self.start_recording()
        elif action == "stop":
            self.stop_recording()
        elif action == "cancel":
            self.cancel_recording()

    def toggle_recording(self) -> None:
        if self.bus.state == "recording":
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.bus.state == "recording":
            return
        if self._busy:
            self.log.info("ignoring start: still transcribing the previous utterance")
            return
        self._idle_timer.stop()
        self.recorder.start()
        if not self.recorder.is_recording:
            self.log.warning("microphone did not start")
            return
        self.bus.recording_started.emit()
        self.bus.set_state("recording")
        seconds = int(self.config.get("max_record_seconds") or 300)
        self._max_timer.start(seconds * 1000)

    def stop_recording(self) -> None:
        if self.bus.state != "recording":
            return
        self._max_timer.stop()
        audio = self.recorder.stop()
        self.bus.recording_stopped.emit()
        if audio is None or audio.size == 0:
            self.log.info("nothing above the silence threshold; nothing to transcribe")
            self._flash("no speech")
            return
        self._busy = True
        self._last_audio = audio
        self.bus.set_state("transcribing")
        threading.Thread(
            target=self._decode, args=(audio,), daemon=True, name="dictate-decode"
        ).start()

    def cancel_recording(self) -> None:
        if self.bus.state != "recording":
            return
        self._max_timer.stop()
        self.recorder.cancel()
        self.bus.recording_stopped.emit()
        self.log.info("recording cancelled")
        self.bus.set_state("idle")

    def reload_mic(self) -> None:
        if self.recorder.is_recording:
            self.cancel_recording()
        self.recorder.reopen()
        self.log.info("microphone set to %r", self.config.get("mic_device"))

    def _on_max_duration(self) -> None:
        self.log.warning("hit the %s s limit; stopping", self.config.get("max_record_seconds"))
        self.stop_recording()

    # --------------------------------------------------------------- worker

    def _decode(self, audio) -> None:
        try:
            result = self.transcriber.transcribe_audio(audio)
        except Exception as exc:
            self.log.exception("transcription failed")
            self.bus.transcription_error.emit(str(exc))
            return
        if not result.accepted:
            self.bus.transcription_rejected.emit(result.reject_reason or "unusable audio")
            return
        self.log.info(
            "decode: language=%s p=%.2f",
            result.language,
            result.language_probability or 0.0,
        )
        self.bus.transcription_done.emit(result.text, result.language)

    def _on_transcription_done(self, text: str, language: str) -> None:
        self._busy = False
        text = (text or "").strip()
        self._check_microphone_health(self._last_audio)
        save_history(
            self._last_audio,
            text,
            self.config,
            meta={"language": language, "engine": self.transcriber.engine_name},
        )
        self._last_audio = None
        if not text:
            self._flash("no speech recognised")
            return
        try:
            text = normalise(text, self.config, language=language)
        except Exception:
            self.log.exception("text normalisation failed; inserting the raw transcript")
        ok = insert_text(
            text,
            auto_enter=bool(self.config.get("auto_enter")),
            enter_delay_ms=int(self.config.get("auto_enter_delay_ms") or 120),
            trailing_space=bool(self.config.get("trailing_space")),
        )
        if not ok:
            self.log.error("could not insert text into the focused window")
            self.tray.notify(
                "OpenVoice",
                "Could not insert the text. Check the log for the reason.",
            )
        else:
            self.tray.set_last_language(language)
        self.log.info(
            "inserted %d characters%s (language=%s)",
            len(text),
            " + Enter" if self.config.get("auto_enter") else "",
            language,
        )
        self.bus.set_state("idle")

    def _check_microphone_health(self, audio) -> None:
        """Warn once when the captured audio itself is the problem.

        The previous version of this check used the share of energy above 1 kHz
        and wrongly declared a healthy microphone unusable, because a relative
        share is swamped by low-frequency rumble and normal speech has a
        -6 dB/octave spectral tilt. This uses absolute band levels, an SNR
        estimate and a clipping fraction instead, and it only fires on evidence
        that actually predicts poor transcription.
        """
        if audio is None or getattr(audio, "size", 0) == 0:
            return
        try:
            quality = measure_quality(audio)
        except Exception as exc:
            self.log.warning("microphone check failed: %s", exc)
            return
        self._quality_samples.append(quality)
        if len(self._quality_samples) < 3 or self._warned_about_mic:
            return
        tilts = sorted(q["tilt_db"] for q in self._quality_samples)
        median_tilt = tilts[len(tilts) // 2]
        worst_clipping = max(q["clipping_fraction"] for q in self._quality_samples)
        if median_tilt > -30.0 and worst_clipping < 0.001:
            return
        self._warned_about_mic = True
        self.log.warning(
            "capture quality looks poor: median tilt %.1f dB, clipping %.3f%%",
            median_tilt, worst_clipping * 100.0,
        )
        self.tray.notify(
            "OpenVoice: capture quality",
            f"Speech roll-off is {median_tilt:.0f} dB and clipping is "
            f"{worst_clipping * 100:.1f}%. Run tools/mic_check.py for the full report.",
        )

    def _on_rejected(self, reason: str) -> None:
        """The engine refused to guess. Better to say so than to insert a guess."""
        self._busy = False
        self._last_audio = None
        self.log.info("transcript rejected: %s", reason)
        self.bus.set_state("error")
        self._idle_timer.start(NO_SPEECH_MS)

    def _on_error(self, message: str) -> None:
        self._busy = False
        self.log.error("engine error: %s", message)
        self.tray.notify("OpenVoice", message)
        self.bus.set_state("error")
        self._idle_timer.start(ERROR_MS)

    def _flash(self, reason: str) -> None:
        self.log.info("%s", reason)
        self.bus.set_state("error")
        self._idle_timer.start(NO_SPEECH_MS)


def main() -> int:
    setup_logging(default_config_path().with_name(LOG_NAME))
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("OpenVoice")
    qapp.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        logging.getLogger("app").warning("no system tray available; the app still runs")

    if single_instance_taken(IPC_NAME):
        print("OpenVoice is already running — look for the tray icon.")
        return 0

    app = DictationApp(qapp)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
