; Inno Setup script for the thin Chatterbox TTS installer.
;
; Ships only the self-contained ~8 MB server binary + launcher + manifest;
; the ~1.4 GB model weights are downloaded + verified on first run by
; fetch-models.ps1 (see models.manifest.json / dist/README.md).
;
; Build (needs Inno Setup 6 / ISCC.exe):
;   1. Stage the Vulkan-enabled server binary next to this script as
;      chatterbox-server.exe  (build chatterbox-cpp with
;      -DCHATTERBOX_VULKAN=ON, then cargo build --release with
;      CHATTERBOX_CPP_BUILD_DIR=...\chatterbox-cpp\build_vk).
;   2. Set models.manifest.json's base_url to the public release host.
;   3. ISCC.exe chatterbox.iss   ->  Output\chatterbox-tts-setup.exe

#define AppName "Chatterbox TTS"
#define AppVersion "1.0.0"
#define AppPublisher "Chatterbox AMD Vulkan"

[Setup]
; Stable identity for clean in-place upgrades + a single Add/Remove Programs
; entry. Do NOT change this between versions.
AppId={{3400EB4C-D0EC-4523-B084-7D34371CA608}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Chatterbox TTS
DefaultGroupName=Chatterbox TTS
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=chatterbox-tts-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog commandline

[Files]
Source: "chatterbox-server.exe";   DestDir: "{app}"; Flags: ignoreversion
Source: "fetch-models.ps1";        DestDir: "{app}"; Flags: ignoreversion
Source: "launch.ps1";              DestDir: "{app}"; Flags: ignoreversion
Source: "models.manifest.json";    DestDir: "{app}"; Flags: ignoreversion
Source: "config.yaml";             DestDir: "{app}"; Flags: onlyifdoesntexist

[Dirs]
Name: "{app}\models"

[Icons]
; Shortcuts launch via PowerShell so first-run model download happens.
Name: "{group}\Chatterbox TTS"; Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\launch.ps1"""; \
  WorkingDir: "{app}"
Name: "{autodesktop}\Chatterbox TTS"; Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\launch.ps1"""; \
  WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\Uninstall Chatterbox TTS"; Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -NoProfile -File ""{app}\launch.ps1"""; \
  WorkingDir: "{app}"; Description: "Launch Chatterbox TTS (downloads models on first run)"; \
  Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Remove downloaded weights on uninstall.
Type: filesandordirs; Name: "{app}\models"
