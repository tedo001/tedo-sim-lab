@echo off
rem Build TEDO-AI-Lab.exe and its Windows installer (.msi), from the repository root or anywhere:
rem
rem     packaging\build_installer.bat
rem
rem Needs only 64-bit Python 3.11-3.13 on PATH. Every build tool is permissively licensed
rem (cx_Freeze PSF-2.0, freeze-core MIT, lief Apache-2.0); see packaging\INSTALLER.md.
rem Output:
rem     dist\TEDO-AI-Lab\TEDO-AI-Lab.exe                 the application folder
rem     dist\installer\TEDO-AI-Lab-<version>-win64.msi   the installer to hand out
rem PyTorch is not bundled: the installed app makes its own experiment Python on request.

setlocal
cd /d "%~dp0\.."

where python >nul 2>nul
if errorlevel 1 goto :nopython

if not exist ".venv-build\Scripts\python.exe" python -m venv .venv-build
if errorlevel 1 goto :fail
call ".venv-build\Scripts\activate.bat"

echo === 1/4 Installing the window's libraries and cx_Freeze ===
python -m pip install --upgrade pip
pip install -r packaging\requirements-window.txt -r packaging\requirements-build.txt
if errorlevel 1 goto :fail

echo === 2/4 cx_Freeze: dist\TEDO-AI-Lab\TEDO-AI-Lab.exe ===
if exist "dist\TEDO-AI-Lab" rmdir /s /q "dist\TEDO-AI-Lab"
python packaging\freeze.py build_exe
if errorlevel 1 goto :fail

echo === 3/4 Checking the built app opens every page ===
start "" /wait "dist\TEDO-AI-Lab\TEDO-AI-Lab.exe" --smoke-test
if errorlevel 1 goto :smokefail
echo TEDO-AI-Lab.exe starts and opens every page.

echo === 4/4 The installer: dist\installer\*.msi ===
python packaging\freeze.py bdist_msi --skip-build
if errorlevel 1 goto :fail

echo.
echo Done.
echo   Application: dist\TEDO-AI-Lab\TEDO-AI-Lab.exe
for %%f in (dist\installer\*.msi) do echo   Installer:   %%f
exit /b 0

:nopython
echo Python was not found on PATH. Install 64-bit Python 3.11+ from python.org, tick
echo "Add python.exe to PATH", then open a new terminal and run this again.
exit /b 1

:smokefail
echo.
echo TEDO-AI-Lab.exe was built but did not open every page. To see why, run it from a terminal:
echo     dist\TEDO-AI-Lab\TEDO-AI-Lab.exe --smoke-test
echo and read logs\app.log in the temporary tedo-smoke-* workspace it keeps on failure.
exit /b 1

:fail
echo.
echo The build stopped at the step above - read the error printed there.
exit /b 1
