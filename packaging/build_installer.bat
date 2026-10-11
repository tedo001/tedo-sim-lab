@echo off
rem Build TEDO-AI-Lab.exe and its setup wizard, from the repository root or anywhere:
rem
rem     packaging\build_installer.bat
rem
rem Needs: 64-bit Python 3.11-3.13 on PATH, and Inno Setup 6 (winget install JRSoftware.InnoSetup).
rem Output:
rem     dist\TEDO-AI-Lab\TEDO-AI-Lab.exe                 the application folder
rem     dist\installer\TEDO-AI-Lab-<version>-setup.exe   the setup wizard to hand out
rem PyTorch is not bundled: the installed app makes its own experiment Python on request.

setlocal
cd /d "%~dp0\.."

where python >nul 2>nul
if errorlevel 1 goto :nopython

if not exist ".venv-build\Scripts\python.exe" python -m venv .venv-build
if errorlevel 1 goto :fail
call ".venv-build\Scripts\activate.bat"

echo === 1/4 Installing the window's libraries and PyInstaller ===
python -m pip install --upgrade pip
pip install -r packaging\requirements-window.txt pyinstaller
if errorlevel 1 goto :fail

echo === 2/4 PyInstaller: dist\TEDO-AI-Lab\TEDO-AI-Lab.exe ===
pyinstaller packaging\tedo_lab.spec --noconfirm --clean
if errorlevel 1 goto :fail

echo === 3/4 Checking the built app opens every page ===
start "" /wait "dist\TEDO-AI-Lab\TEDO-AI-Lab.exe" --smoke-test
if errorlevel 1 goto :smokefail
echo TEDO-AI-Lab.exe starts and opens every page.

echo === 4/4 Inno Setup: the setup wizard ===
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" goto :noinno
for /f "delims=" %%v in ('python -c "from app import __version__; print(__version__)"') do set VER=%%v
"%ISCC%" /DAppVersion=%VER% packaging\installer.iss
if errorlevel 1 goto :fail

echo.
echo Done.
echo   Application: dist\TEDO-AI-Lab\TEDO-AI-Lab.exe
echo   Installer:   dist\installer\TEDO-AI-Lab-%VER%-setup.exe
exit /b 0

:nopython
echo Python was not found on PATH. Install 64-bit Python 3.11+ from python.org, tick
echo "Add python.exe to PATH", then open a new terminal and run this again.
exit /b 1

:smokefail
echo.
echo TEDO-AI-Lab.exe was built but did not open every page. To see why, run it from a terminal:
echo     dist\TEDO-AI-Lab\TEDO-AI-Lab.exe --smoke-test
echo and read logs in the temporary tedo-smoke-* workspace it keeps on failure.
exit /b 1

:noinno
echo.
echo TEDO-AI-Lab.exe is built in dist\TEDO-AI-Lab, but Inno Setup 6 was not found, so no setup
echo wizard was made. Install it with:  winget install JRSoftware.InnoSetup  then run this again.
exit /b 1

:fail
echo.
echo The build stopped at the step above - read the error printed there.
exit /b 1
