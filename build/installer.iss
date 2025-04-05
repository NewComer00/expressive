#define AppVersion GetEnv("TAG_NAME")

[Setup]
AppName=Expressive-GUI
AppVersion={#AppVersion}
DefaultDirName={pf}\Expressive-GUI
DefaultGroupName=Expressive-GUI
OutputDir=dist
OutputBaseFilename=Expressive-GUI-{#AppVersion}-Installer
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\Expressive-GUI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; Start Menu Shortcuts
Name: "{group}\Expressive-GUI (English)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=en"; WorkingDir: "{app}"
Name: "{group}\Expressive-GUI (中文)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=zh_CN"; WorkingDir: "{app}"

; Desktop Shortcuts
Name: "{userdesktop}\Expressive-GUI (English)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=en"; Tasks: desktopicon; WorkingDir: "{app}"
Name: "{userdesktop}\Expressive-GUI (中文)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=zh_CN"; Tasks: desktopicon; WorkingDir: "{app}"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional tasks:"
