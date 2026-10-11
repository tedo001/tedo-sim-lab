# Third-party notices — TEDO AI Research Lab (installed app)

The lab's own code is by tedo001 <durgamani.d.e.c.e.50@gmail.com>. The installed app bundles
the components below; each keeps its own licence. Every Python package's licence is listed in
`lab-code/configs/dependency_licences.yaml`; all are permissive (MIT, BSD, Apache-2.0, ISC, PSF)
except Qt for Python, below.

## Qt for Python (PySide6, Shiboken6) and Qt — LGPL-3.0

The interface uses Qt through PySide6 under the GNU Lesser General Public License v3.0
(`LGPL-3.0.txt`, which incorporates `GPL-3.0.txt`).

- The Qt and PySide6 libraries are separate, unmodified shared libraries inside the
  installation folder (`lib/PySide6/`). You may replace them with your own builds of the
  same versions; the application loads whatever libraries are there.
- Source code: <https://code.qt.io/cgit/pyside/pyside-setup.git/> and
  <https://download.qt.io/official_releases/QtForPython/> (Qt for Python), and
  <https://download.qt.io/official_releases/qt/> (Qt).
- The installed version is shown in the app's status bar ("Qt x.y.z").

## The launcher and the Python runtime

- `TEDO-AI-Lab.exe`, the small program that starts the app, comes from freeze-core (MIT) through
  cx_Freeze (PSF-2.0); its notice is `frozen_application_license.txt` in the installation
  folder. cx_Freeze and python-msilib (PSF-2.0), which make the app folder and the `.msi`, are
  build tools and are not distributed.
- The Python runtime and standard library: Python Software Foundation License 2.0
  (<https://docs.python.org/3/license.html>).

## Bundled assets

- Inter and JetBrains Mono fonts — SIL Open Font License 1.1 (`app/resources/fonts/OFL-*.txt`)
- Lucide icons — ISC (`app/resources/icons/LICENSE-lucide.txt`)
- CNN Explainer (ported) — MIT (`labs/computer_vision/explainer/LICENSE-cnn-explainer.txt`)
- ImageNet-1k class list (names only, no images) — Apache-2.0 via timm; WordNet 3.0 licence
  (`labs/ontology/data/`)
