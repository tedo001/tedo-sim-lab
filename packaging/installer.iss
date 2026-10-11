; Inno Setup script for TEDO AI Research Lab: the Windows setup wizard.
;
; It packages the folder PyInstaller produced; it does not compile Python:
;
;     pyinstaller packaging\tedo_lab.spec --noconfirm --clean     -> dist\TEDO-AI-Lab\
;     iscc packaging\installer.iss                                 -> dist\installer\TEDO-AI-Lab-<ver>-setup.exe
;
; packaging\build_installer.bat runs both; /DAppVersion=x.y.z sets the version (the build
; script passes app/__init__.py's __version__).
;
; Wizard: Welcome -> What is installed and where your data goes -> Install folder -> Start menu
; -> Desktop shortcut -> Installing -> Next steps (experiment Python) -> Finish ("Start the lab").

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define AppName "TEDO AI Research Lab"
#define AppShort "TEDO-AI-Lab"
#define AppPublisher "tedo001"
#define AppExeName "TEDO-AI-Lab.exe"
#define SourceDir "..\dist\TEDO-AI-Lab"

[Setup]
AppId={{5B0F7C2E-8A4D-4E61-9C3B-2F7D1A6E4B90}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/tedo001/tedo-sim-lab
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=..\dist\installer
OutputBaseFilename={#AppShort}-{#AppVersion}-setup
SetupIconFile=tedo-lab.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-machine when elevated, per-user otherwise (no administrator needed).
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}
CloseApplications=yes
RestartApplications=no
InfoBeforeFile=before_install.txt
InfoAfterFile=after_install.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "after_install.txt"; DestDir: "{app}"; DestName: "NEXT-STEPS.txt"; Flags: ignoreversion
Source: "licences\*"; DestDir: "{app}\licences"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"
Name: "{group}\Next steps (experiment Python)"; Filename: "{app}\NEXT-STEPS.txt"
Name: "{group}\Licences"; Filename: "{app}\licences"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

; The workspace (%USERPROFILE%\TEDO-AI-Lab by default) and the experiment Python
; (%LOCALAPPDATA%\TEDO AI Lab) are never touched by an update or by uninstalling: they hold
; the person's work and a multi-GB environment a reinstall should find again.
