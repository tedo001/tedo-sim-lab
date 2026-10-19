"""Any simulation as a Python script that runs it again with the lab's own modules — a starting
point to change freely in the Code page (which puts the lab's code on ``PYTHONPATH``)."""

from __future__ import annotations

import json
import pprint

__all__ = ["script_for"]

_HEAD = '''"""{title}: written by the TEDO AI Research Lab's Simulation Lab.

Run it from the Code page (F5), or with the experiment Python: the lab's code must be on
PYTHONPATH (the Code page does that). Change anything below and run it again.
"""

import json

import numpy as np

'''


def script_for(kind: str, settings: dict) -> str:
    builders = {"circuit": _circuit, "waves": _waves, "lattice": _lattice, "image": _image, "scenes": _scenes,
                "camera": _camera}
    if kind not in builders:
        raise ValueError(f"no script for {kind!r}")
    return builders[kind](settings)


def _literal(value) -> str:
    """``value`` as Python source (True, None and tuples, not JSON's true and null)."""
    return pprint.pformat(value, indent=4, width=96, sort_dicts=False)


def _circuit(settings: dict) -> str:
    head = _HEAD.format(title="Quantum circuit")
    return head + f'''from labs.simulation.quantum import Circuit, sample, simulate

CIRCUIT = json.loads("""{json.dumps(settings["circuit"])}""")
SHOTS = {int(settings.get("shots", 1024))}
SEED = {int(settings.get("seed", 0))}

circuit = Circuit.from_dict(CIRCUIT)
result = simulate(circuit)
print("Amplitudes (basis, amplitude, probability):")
for label, amplitude, probability in result.amplitudes():
    print(f"  |{{label}}>  {{amplitude.real:+.4f}}{{amplitude.imag:+.4f}}i   {{probability:.4f}}")
for qubit in range(circuit.qubits):
    x, y, z = result.bloch(qubit)
    print(f"Bloch vector of qubit {{qubit}}: ({{x:+.3f}}, {{y:+.3f}}, {{z:+.3f}})")
print(f"{{SHOTS}} shots:", sample(result.state, SHOTS, SEED))
print()
print(circuit.to_qasm())
'''


def _waves(settings: dict) -> str:
    head = _HEAD.format(title="Wave mechanics")
    return head + f'''from labs.simulation.quantum.waves import WaveSetup, WaveSimulation, eigenstates

SETUP = WaveSetup.from_dict(json.loads("""{json.dumps(settings["setup"])}"""))
DURATION = {float(settings.get("duration", 30.0))}

simulation = WaveSimulation(SETUP)
steps = int(DURATION / SETUP.dt)
for chunk in range(10):
    simulation.step(max(1, steps // 10))
    print(f"t = {{simulation.time:7.2f}}  norm = {{simulation.norm():.5f}}  "
          f"<x> = {{simulation.mean_x():+8.3f}}  <E> = {{simulation.energy():.5f}}  "
          f"right of 0: {{simulation.beyond(0.0):.4f}}")

energies, x, states = eigenstates(SETUP, 6)
print("Lowest energies:", np.round(energies, 5))
'''


def _lattice(settings: dict) -> str:
    head = _HEAD.format(title="Electrons in a lattice (Kohn-Sham)")
    return head + f'''from labs.simulation.quantum.kohn_sham import ENERGY_TERMS, solve
from labs.simulation.quantum.lattice import LatticeSetup

SETUP = LatticeSetup.from_dict({_literal(settings["setup"])})

result = solve(SETUP, progress=lambda fraction, text: print(text))
print("Converged" if result.converged else "Not converged", f"in {{result.seconds:.2f}} s")
for key, title in ENERGY_TERMS.items():
    print(f"{{title:22s}} {{result.energies[key]:+.6f}} Ha")
gap = result.gap()
print(f"Fermi level {{result.fermi:.5f}} Ha;", "a metal" if gap is None else f"gap {{gap:.5f}} Ha")
print("Lowest levels (Ha) and electrons in them:")
for n, (energy, filled) in enumerate(zip(result.eigenvalues[:12], result.occupations[:12])):
    print(f"  {{n:3d}} {{energy:+.5f}} {{filled:.3f}}")
np.savetxt("density.csv", result.density, delimiter=",")
print("Wrote density.csv (rows along y)")

try:  # draw it in 3-D if matplotlib is installed (it is in the experiment Python)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    x, y = np.meshgrid(result.x, result.y)
    figure = plt.figure(figsize=(11, 5))
    for index, (field, title) in enumerate(((result.v_ion, "Lattice potential V_ion (Ha)"),
                                            (result.density, "Electron density n (bohr^-2)"))):
        axes = figure.add_subplot(1, 2, index + 1, projection="3d")
        axes.plot_surface(x, y, field, cmap="Blues_r" if index == 0 else "Reds", linewidth=0.2)
        axes.set_title(title)
    plt.show()
'''


def _image(settings: dict) -> str:
    source = settings.get("source", {"sample": "Shapes"})
    load = (f'image = sample({source["sample"]!r})' if "sample" in source else
            f'image = read_png({source.get("file", "input.png")!r}).astype(np.float32) / 255.0')
    head = _HEAD.format(title="Image pipeline")
    return head + f'''from labs.simulation.vision import Step, run_pipeline, sample
from labs.simulation.vision.png import read_png, write_png

PIPELINE = {_literal(settings["pipeline"])}

{load}
steps = [Step.from_dict(item) for item in PIPELINE]
result, timings = run_pipeline(image, steps)
for step, ms in zip(steps, timings):
    print(f"{{step.op:14s}} {{ms:7.1f}} ms {{'' if step.enabled else '(off)'}}")
write_png("result.png", result)
print("Wrote result.png", result.shape)
'''


def _scenes(settings: dict) -> str:
    return _HEAD.format(title="Synthetic scenes") + f'''from pathlib import Path

from labs.simulation.vision import SceneSpec, generate_dataset

SPEC = SceneSpec.from_dict({_literal(settings["spec"])})
COUNT = {int(settings.get("count", 100))}

annotations = generate_dataset(SPEC, COUNT, Path("synthetic-shapes"), progress=None)
print("Wrote", annotations)
'''


def _camera(settings: dict) -> str:
    head = _HEAD.format(title="Camera model")
    return head + f'''from labs.simulation.vision import Camera, project, projection_matrix

CAMERA = Camera.from_dict({_literal(settings["camera"])})

np.set_printoptions(precision=3, suppress=True)
print("P = K [R | t] =")
print(projection_matrix(CAMERA))
corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
uv, visible = project(CAMERA, corners)
for corner, point, seen in zip(corners, uv, visible):
    print(corner, "->", point, "visible" if seen else "outside")
'''
