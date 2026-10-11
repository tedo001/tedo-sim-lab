# Building the Windows installer

```
 the code ──cx_Freeze──▶ dist\TEDO-AI-Lab\TEDO-AI-Lab.exe ──cx_Freeze bdist_msi──▶ dist\installer\TEDO-AI-Lab-<version>-win64.msi
           (the window)    (a folder that runs)                                    (the one file you hand out)
```

Every tool in the build is MIT, Apache-2.0 or PSF licensed (the owner's licence rule):

| Tool | Licence | What it does |
| --- | --- | --- |
| cx_Freeze | PSF-2.0 | Freezes the app into a folder; writes the `.msi` |
| freeze-core | MIT | The launcher, `TEDO-AI-Lab.exe`. It is the only build-tool code that ships in the app |
| lief | Apache-2.0 | Edits the launcher's icon and resources during the build |
| python-msilib / `msilib` | PSF-2.0 | Writes the Windows Installer database (built into Python 3.11 and 3.12) |
| Windows Installer | part of Windows | Installs, repairs and uninstalls the `.msi` |

PyInstaller (its bootloader is GPL with an exception) and Inno Setup (its own licence) are not
used. `configs/excluded_tools.yaml` lists them, and `tests/test_dependency_licences.py` checks the
build tools in `packaging/requirements-build.txt`.

cx_Freeze builds for the system it runs on, so the app and its `.msi` are built on Windows: on
your PC, or by GitHub Actions (below).

## The easy way: GitHub Actions

*Actions → Windows installer → Run workflow* (branch `tedo`). After about 15 minutes the run's
**Artifacts** section has `TEDO-AI-Lab-setup` (the `.msi`). The workflow:

1. builds the app;
2. checks it opens every page;
3. writes the `.msi`;
4. installs it silently and checks the installed copy;
5. uninstalls it and checks that nothing is left in the install folder.

Pushing a tag `v0.1.0` does the same and attaches the `.msi` to that GitHub release.
Changes under `packaging/` also start a build.

## On your PC

Once: install Python 3.11 (64-bit) from <https://www.python.org/downloads/> and tick **Add
python.exe to PATH**. Check with `python --version`. Nothing else is needed.

Then, in the repository folder:

```powershell
git pull origin tedo
.\packaging\build_installer.bat
```

The script:

1. makes `.venv-build` with the window's libraries (`packaging\requirements-window.txt`) and cx_Freeze (`packaging\requirements-build.txt`);
2. builds `dist\TEDO-AI-Lab\` with `packaging\freeze.py`;
3. runs the built `.exe` with `--smoke-test` and stops if any page fails to open;
4. writes `dist\installer\TEDO-AI-Lab-<version>-win64.msi` from that same folder.

The version comes from `app/__init__.py`. Installing a newer version replaces the older one: the
`.msi` keeps the same upgrade code (`UPGRADE_CODE` in `freeze.py`; never change it).

## Installing

Double-click the `.msi`. It installs for the current user without administrator rights, in
`%LOCALAPPDATA%\Programs\TEDO AI Research Lab`, adds a Start-menu entry and can start the lab
when it finishes. Uninstall it from *Settings → Apps*.

For a silent install (for example on many machines):

```powershell
msiexec /i TEDO-AI-Lab-0.1.0-win64.msi /qn
```

Windows SmartScreen warns about any unsigned installer. Choose *More info → Run anyway*. Code
signing is on the roadmap.

## What is in it

| Part | Contents | Where it goes |
| --- | --- | --- |
| The window | PySide6, numpy, pandas, scikit-learn, MLflow, nltk, pycocotools, keyring (about 580 MB unpacked) | `lib\` in the install folder |
| `lab-code\` | The lab's plain sources (`core`, `labs`, `plugins`), cards in `configs`, and the documents | the install folder |
| The experiment Python | PyTorch (CUDA build with an NVIDIA GPU), XGBoost, OpenCV and the rest of `configs\experiment-requirements.txt` | `%LOCALAPPDATA%\TEDO AI Lab\experiment-python`, made from inside the app (Settings → Experiment Python) |
| Your work | datasets, runs, notebooks, reports, the database | the workspace: `%USERPROFILE%\TEDO-AI-Lab` unless `--workspace` says otherwise |

Training never runs in the window process. Workers start under the experiment Python with
`lab-code` on `PYTHONPATH`, which holds no compiled files. So any Python 3.11+ works there,
whatever version built the window.

Updating or uninstalling never touches the workspace or the experiment Python.

## Licences

Everything bundled is permissively licensed, with one exception: **Qt for Python (PySide6)**,
LGPL-3.0. No permissively licensed Qt binding exists. Qt stays a set of separate, replaceable
libraries in `lib\PySide6`, which is why the app is one folder, not one file.

`licences\THIRD-PARTY-NOTICES.md` (installed with the app) has the details and the source
links. The LGPL-3.0 text travels with the GPL-3.0 text it incorporates, as the LGPL requires.
`frozen_application_license.txt` is the launcher's MIT notice.

## Test it before handing it out

1. Install on a clean Windows user account, or a virtual machine.
2. Start the lab and open every page.
3. In Settings → Experiment Python, make the environment.
4. Run the "Iris · Logistic regression" preset on the Classical ML page and the "MNIST · SimpleCNN" preset on the Computer Vision page.
5. Uninstall. The workspace and `%LOCALAPPDATA%\TEDO AI Lab` should still be there.
