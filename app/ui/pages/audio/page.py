"""Audio lab: make, open or pick a sound; see it as a waveform, spectrogram, mel spectrogram and MFCCs;
filter it, estimate its pitch, play it in your audio player, transcribe it with a speech model;
and train spoken-digit classifiers on spectrograms, then look inside them on Model Testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QPushButton, QWidget

from core.common.vocab import Modality, Task
from labs.audio import dsp
from labs.audio.presets import AUDIO_PRESETS

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, FlowGrid, KeyValues, Page, label
from ..lab_parts import DownloadedModels, LabDatasets, PresetGrid, PretrainedModels
from .views import TimeFrequency, Waveform

__all__ = ["AudioPage"]


def _spin(low: float, high: float, value: float, suffix: str, decimals: int = 0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setMaximumWidth(140)
    return box


class AudioPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Audio", "sounds as pictures, filters, speech recognition, spoken-digit models",
                         parent)
        self.ctx = ctx
        self.original = np.zeros(0)
        self.signal, self.rate = np.zeros(0), 16_000
        testing = QPushButton(icon("test-tube"), "Model Testing")
        testing.clicked.connect(self.test_on_runs)
        self.head.add_action(testing)

        source = Card("Sound", "generate a test signal, open a WAV file or take a spoken digit")
        self.kind = QComboBox()
        for key, title in dsp.GENERATORS.items():
            self.kind.addItem(title, key)
        self.seconds = _spin(0.1, 30, 2, " s", 1)
        self.frequency = _spin(20, 8000, 440, " Hz")
        make = QPushButton(icon("wand-sparkles"), "Generate")
        make.clicked.connect(self.generate)
        open_button = QPushButton(icon("folder-open"), "Open WAV…")
        open_button.clicked.connect(self.choose)
        self.digit = QPushButton(icon("mic"), "A spoken digit")
        self.digit.setToolTip("A random recording from the Free Spoken Digit Dataset (download it in the "
                              "Dataset Hub)")
        self.digit.clicked.connect(self.spoken_digit)
        source.add(FlowGrid([self.kind, self.seconds, self.frequency, make, open_button, self.digit], 150,
                            fewest=2))
        self.values = KeyValues((("Source", ""), ("Length", ""), ("Sample rate", ""), ("Loudness (RMS)", ""),
                                 ("Pitch", "")))
        source.add(self.values)
        self.body.addWidget(source)

        views = Card("Pictures", "time → right; low frequencies at the bottom")
        self.wave = Waveform()
        self.spectrum = TimeFrequency("spectrogram (dB)")
        self.mel = TimeFrequency("mel spectrogram, 64 bands (dB)")
        self.mfcc = TimeFrequency("MFCCs, 13 coefficients")
        for widget in (self.wave, self.spectrum, self.mel, self.mfcc):
            views.add(widget)
        self.body.addWidget(views)

        tools = Card("Filter, play, save",
                     "Butterworth filters run forwards and backwards, so nothing shifts in time")
        self.low_cut = _spin(0, 8000, 0, " Hz")
        self.low_cut.setPrefix("low cut ")
        self.high_cut = _spin(0, 8000, 0, " Hz")
        self.high_cut.setPrefix("high cut ")
        apply = QPushButton("Apply filter")
        apply.clicked.connect(self.apply_filter)
        reset = QPushButton("Undo filters")
        reset.clicked.connect(lambda: self.show_signal(self.original, self.rate, keep=True))
        play = QPushButton(icon("volume-2"), "Play")
        play.setToolTip("Opens the sound in your audio player")
        play.clicked.connect(self.play)
        save = QPushButton(icon("download"), "Save WAV…")
        save.clicked.connect(self.save)
        tools.add(FlowGrid([self.low_cut, self.high_cut, apply, reset, play, save], 150, fewest=2))
        tools.add(label("0 Hz = no cut. Low cut removes rumble; high cut removes hiss.", "CardCaption",
                        wrap=True))
        self.body.addWidget(tools)

        speech = Card("Speech recognition", "Whisper or wav2vec 2.0 on this computer")
        row = QHBoxLayout()
        self.speech_model = DownloadedModels(ctx, {Task.SPEECH_RECOGNITION})
        transcribe = QPushButton(icon("captions"), "Transcribe")
        transcribe.clicked.connect(self.transcribe)
        row.addWidget(self.speech_model)
        row.addWidget(transcribe)
        row.addStretch(1)
        speech.add(row)
        self.transcript = label("", "Body", wrap=True, selectable=True)
        speech.add(self.transcript)
        self.body.addWidget(speech)

        self.presets = PresetGrid(ctx, AUDIO_PRESETS, "spoken digits as 64×64 log mel spectrograms")
        self.body.addWidget(self.presets)
        self.body.addWidget(LabDatasets(ctx, lambda card: card.modality == Modality.AUDIO))
        self.models = PretrainedModels(ctx, {Task.SPEECH_RECOGNITION})
        self.models.changed.connect(self.speech_model.refresh)
        self.body.addWidget(self.models)
        self.body.addStretch(1)
        ctx.downloads.finished.connect(lambda *_: self.refresh())
        self.refresh()
        self.generate()

    def refresh(self) -> None:
        self.presets.refresh()
        self.digit.setEnabled(self.ctx.catalog.datasets.status("spoken_digits") == "ready")

    # Sources -----------------------------------------------------------------------------------------
    def generate(self) -> None:
        signal = dsp.generate(self.kind.currentData(), self.seconds.value(), 16_000, self.frequency.value())
        self.show_signal(signal, 16_000, source=f"{self.kind.currentText()}, {self.frequency.value():g} Hz")

    def choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a WAV file", str(self.ctx.paths.workspace),
                                              "WAV audio (*.wav)")
        if path:
            self.open_wav(Path(path))

    def open_wav(self, path: Path) -> None:
        try:
            signal, rate = dsp.load_wav(path)
        except (OSError, ValueError) as exc:
            self.transcript.setText(f"Could not read {path.name}: {exc}")
            return
        self.show_signal(signal, rate, source=path.name)

    def spoken_digit(self) -> None:
        folder = self.ctx.paths.datasets / "spoken_digits" / "recordings"
        files = sorted(folder.glob("*.wav"))
        if files:
            self.open_wav(files[np.random.default_rng().integers(len(files))])

    # Showing ------------------------------------------------------------------------------------------
    def show_signal(self, signal: np.ndarray, rate: int, *, source: str = "", keep: bool = False) -> None:
        self.signal, self.rate = np.asarray(signal, dtype=float), int(rate)
        if not keep:
            self.original = self.signal.copy()
        if source:
            self.values.set_value("Source", source)
        self.values.set_value("Length", f"{len(signal) / rate:.2f} s")
        self.values.set_value("Sample rate", f"{rate:,} Hz")
        self.values.set_value("Loudness (RMS)",
                              f"{float(np.sqrt(np.mean(self.signal ** 2))) if signal.size else 0:.3f}")
        pitch = dsp.pitch(self.signal, rate)
        self.values.set_value("Pitch", f"{pitch:.1f} Hz" if pitch else "none found (not tonal)")
        self.wave.set_signal(self.signal, rate)
        power, _times, freqs = dsp.spectrogram(self.signal, rate)
        self.spectrum.set_values(power, low_label="0 Hz", high_label=f"{freqs[-1] / 1000:.1f} kHz")
        mel, _times, centres = dsp.mel_spectrogram(self.signal, rate, 64)
        self.mel.set_values(mel, low_label=f"{centres[0]:.0f} Hz", high_label=f"{centres[-1] / 1000:.1f} kHz")
        self.mfcc.set_values(dsp.mfcc(self.signal, rate)[:, 1:], low_label="c1", high_label="c12",
                             diverging=True)

    def apply_filter(self) -> None:
        nyquist = self.rate / 2
        low = self.low_cut.value() or None
        high = self.high_cut.value() if 0 < self.high_cut.value() < nyquist else None
        if low and high and low >= high:
            self.transcript.setText("The low cut must be below the high cut.")
            return
        self.show_signal(dsp.bandpass(self.signal, self.rate, low, high), self.rate, keep=True)

    def current_file(self) -> Path:
        return dsp.save_wav(self.ctx.inference.workspace_file("audio-lab.wav"), self.signal, self.rate)

    def play(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_file())))

    def save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save the sound",
                                              str(self.ctx.paths.results / "sound.wav"),
                                              "WAV audio (*.wav)")
        if path:
            dsp.save_wav(Path(path), self.signal, self.rate)

    def transcribe(self) -> None:
        if self.speech_model.repo is None:
            self.transcript.setText("Download a speech model below first.")
            return
        self.transcript.setText("Listening…")
        pieces: list[str] = []

        def partial(data: dict) -> None:
            pieces.append(data.get("text", ""))
            self.transcript.setText(" ".join(pieces) + " …")

        def done(result: dict) -> None:
            self.transcript.setText(result["text"] or "(nothing heard)")

        audio = str(self.current_file())
        self.ctx.inference.request("transcribe", {"model": self.speech_model.repo, "audio": audio},
                                   on_partial=partial, on_done=done,
                                   on_error=lambda m: self.transcript.setText(f"Failed: {m}"))

    def test_on_runs(self) -> None:
        """Model Testing with this sound as the input for a spoken-digit run."""
        page = self.ctx.show_page("testing")
        audio_runs = [item for item in page.items if item.kind == "audio"]
        if audio_runs:
            page.open_run(audio_runs[0].run_id)
        page.inputs.set_path(self.current_file())
