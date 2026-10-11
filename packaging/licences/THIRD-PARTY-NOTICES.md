# Third-party notices — TEDO AI Research Lab (installed app)

The lab's own code is by tedo001 <durgamani.d.e.c.e.50@gmail.com>. The installed app bundles
the components below; each keeps its own licence. Every Python package's licence is listed in
`lab-code/configs/dependency_licences.yaml`; all are permissive (MIT, BSD, Apache-2.0, ISC, PSF)
except Qt for Python, below.

## Qt for Python (PySide6, Shiboken6) and Qt — LGPL-3.0

The interface uses Qt through PySide6 under the GNU Lesser General Public License v3.0
(`LGPL-3.0.txt`, which incorporates `GPL-3.0.txt`).

- The Qt and PySide6 libraries are separate, unmodified shared libraries inside the
  installation folder (`_internal/PySide6/`). You may replace them with your own builds of the
  same versions; the application loads whatever libraries are there.
- Source code: <https://code.qt.io/cgit/pyside/pyside-setup.git/> and
  <https://download.qt.io/official_releases/QtForPython/> (Qt for Python), and
  <https://download.qt.io/official_releases/qt/> (Qt).
- The installed version is shown in the app's status bar ("Qt x.y.z").

## PyInstaller bootloader — GPL-2.0-or-later with the bootloader exception

The small program that starts the bundle is PyInstaller's bootloader. Its licence includes an
exception that allows distributing it with programs under any licence; PyInstaller itself is a
build tool and is not distributed. <https://pyinstaller.org/en/stable/license.html>

## Bundled assets

- Inter and JetBrains Mono fonts — SIL Open Font License 1.1 (`app/resources/fonts/OFL-*.txt`)
- Lucide icons — ISC (`app/resources/icons/LICENSE-lucide.txt`)
- CNN Explainer (ported) — MIT (`labs/computer_vision/explainer/LICENSE-cnn-explainer.txt`)
- ImageNet-1k class list (names only, no images) — Apache-2.0 via timm; WordNet 3.0 licence
  (`labs/ontology/data/`)
