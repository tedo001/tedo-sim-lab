# Building the Windows installer

```
 the code ──PyInstaller──▶ dist\TEDO-AI-Lab\TEDO-AI-Lab.exe ──Inno Setup──▶ dist\installer\TEDO-AI-Lab-<version>-setup.exe
            (the window)      (a folder that runs)                           (the one file you hand out)
```

PyInstaller builds for the system it runs on, so the `.exe` is built on Windows: on your
PC, or by GitHub Actions (below).

## The easy way: GitHub Actions

*Actions → Windows installer → Run workflow* (branch `tedo`). After about 15 minutes the run's
**Artifacts** section has `TEDO-AI-Lab-setup` (the setup `.exe`). The workflow:

1. builds the app;
2. checks it opens every page;
3. compiles the setup;
4. installs it silently;
5. checks the installed copy as well.

Pushing a tag `v0.1.0` does the same and attaches the setup file to that GitHub release.
Changes under `packaging/` also start a build.

## On your PC

Once:

| Tool | How | Check |
| --- | --- | --- |
| Python 3.11 (64-bit) | <https://www.python.org/downloads/>, tick **Add python.exe to PATH** | `python --version` |
| Inno Setup 6 | `winget install JRSoftware.InnoSetup` | `& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" /?` |

Then, in the repository folder:

```powershell
git pull origin tedo
.\packaging\build_installer.bat
```

The script:

1. makes `.venv-build` with the window's libraries (`packaging\requirements-window.txt`) and PyInstaller;
2. builds `dist\TEDO-AI-Lab\` from `packaging\tedo_lab.spec`;
3. runs the built `.exe` with `--smoke-test` and stops if any page fails to open;
4. compiles `packaging\installer.iss` into `dist\installer\TEDO-AI-Lab-<version>-setup.exe`.

The version comes from `app/__init__.py`.

## What is in it

| Part | Contents | Where it goes |
| --- | --- | --- |
| The window | PySide6, numpy, pandas, scikit-learn, MLflow, nltk, pycocotools, keyring (about 520 MB unpacked) | Program Files (or the user's programs folder without administrator rights) |
| `_internal\lab-code\` | The lab's plain sources (`core`, `labs`, `plugins`), cards in `configs`, and the documents | inside the install folder |
| The experiment Python | PyTorch (CUDA build with an NVIDIA GPU), XGBoost, OpenCV and the rest of `configs\experiment-requirements.txt` | `%LOCALAPPDATA%\TEDO AI Lab\experiment-python`, made from inside the app (Settings → Experiment Python) |
| Your work | datasets, runs, notebooks, reports, the database | the workspace: `%USERPROFILE%\TEDO-AI-Lab` unless `--workspace` says otherwise |

Training never runs in the window process. Workers start under the experiment Python with
`lab-code` on `PYTHONPATH`, which holds no compiled files. So any Python 3.11+ works there,
whatever version built the window.

Updating or uninstalling never touches the workspace or the experiment Python.

## Licences

Everything bundled is permissively licensed, with two exceptions:

- **Qt for Python (PySide6):** LGPL-3.0. It stays a set of separate, replaceable libraries in a one-folder build.
- **PyInstaller's bootloader:** GPL with an exception that allows any program to ship it.

`licences\THIRD-PARTY-NOTICES.md` (installed with the app) has the details and the source
links. PyInstaller and Inno Setup are build tools, not dependencies of the lab.

## Test it before handing it out

1. Install on a clean Windows user account, or a virtual machine.
2. Start the lab and open every page.
3. In Settings → Experiment Python, make the environment.
4. Run the "Iris · Logistic regression" preset on the Classical ML page and the "MNIST · SimpleCNN" preset on the Computer Vision page.
5. Uninstall. The workspace and `%LOCALAPPDATA%\TEDO AI Lab` should still be there.
