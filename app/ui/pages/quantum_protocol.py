"""The evaluation protocol of a quantum run on the Quantum ML page: simulated, not hardware; the
simulator and its noise; the resources it took; the classical baselines on the same split; and how
the trained model's accuracy changes as the noise is scaled."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QWidget

from ..widgets import Card, DataTable, KeyValues, Pill, ResponsiveRow, label
from ..widgets.plots import BarList

__all__ = ["ProtocolCard"]


class ProtocolCard(Card):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Evaluation protocol", "simulated on a classical computer", parent=parent)
        self.pill = Pill("Simulator results · not hardware", "warn")
        self.add_head_widget(self.pill)
        self.claim = label("", "Body", wrap=True)
        self.add(self.claim)
        self.values = KeyValues((("Simulator", ""), ("Noise", ""), ("Qubits · depth", ""), ("Gates", ""),
                                 ("Evaluations · shots", ""), ("Memory per state", ""), ("Wall time", "")))
        baselines = Card("Classical baselines", "same split, same features", padded=False)
        self.baselines = DataTable(("Model", "Test accuracy", "Macro F1", "Fit s"), mono_columns=(1, 2, 3),
                                   stretch_column=0)
        baselines.add(self.baselines)
        noise = Card("Noise sensitivity", "test accuracy as the noise is scaled")
        self.noise = BarList()
        noise.add(self.noise)
        self.noise_note = label("", "CardCaption", wrap=True)
        noise.add(self.noise_note)
        self.add(self.values)
        self.add(ResponsiveRow([(baselines, 1), (noise, 1)], breakpoint=900))
        self.setVisible(False)

    def show_protocol(self, protocol: dict[str, Any] | None, test_acc: float | None) -> None:
        self.setVisible(bool(protocol))
        if not protocol:
            return
        sim, res, claims = protocol["simulator"], protocol["resources"], protocol["claims"]
        self.claim.setText(f"{claims['comparison']} Quantum advantage {claims['quantum_advantage']}.")
        noise = sim.get("noise") or {}
        shots = sim.get("shots") or "exact"
        self.values.set_value("Simulator", f"{sim['mode']} · {sim['kind']} · shots {shots}")
        self.values.set_value("Noise", ", ".join(f"{k} {v}" for k, v in noise.items()) or "none")
        self.values.set_value("Qubits · depth", f"{res['qubits']} · {res['depth']}")
        self.values.set_value("Gates", f"{res['gates_1q']} one-qubit · {res['gates_2q']} two-qubit · "
                                       f"{res['parameters']} parameters")
        self.values.set_value("Evaluations · shots", f"{res['circuit_evaluations']:,} · {res['shots']:,}")
        self.values.set_value("Memory per state", f"{res['state_memory_bytes']:,} bytes")
        self.values.set_value("Wall time", f"{res['wall_time_s']:.1f} s")
        self.baselines.clear_rows()
        if test_acc is not None:
            self.baselines.add_row(("This quantum model", f"{test_acc:.4f}", "—", "—"))
        for row in protocol["baselines"]:
            self.baselines.add_row((row["title"], f"{row['test_acc']:.4f}", f"{row['test_f1']:.4f}",
                                    f"{row['fit_time_s']:.2f}"))
        comparison = protocol["noise_comparison"]
        self.noise.set_rows([(f"noise × {row['factor']:g}", row["test_acc"]) for row in comparison["rows"]])
        self.noise_note.setText(f"Scaled from {comparison['reference']}; × 0 is the ideal simulation.")
