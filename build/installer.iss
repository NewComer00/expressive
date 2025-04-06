#define AppVersion GetEnv("TAG_NAME")

[Setup]
AppName=Expressive-GUI
AppVersion={#AppVersion}
DefaultDirName={pf}\Expressive-GUI
DefaultGroupName=Expressive-GUI
OutputDir=..\dist
OutputBaseFilename=Expressive-GUI-{#AppVersion}-Installer
Compression=lzma
SolidCompression=yes
SetupIconFile=..\assets\icons\app.ico

[Files]
Source: "..\dist\Expressive-GUI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; Start Menu Shortcuts
Name: "{group}\Expressive-GUI (English)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=en"; WorkingDir: "{app}"
Name: "{group}\Expressive-GUI (中文)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=zh_CN"; WorkingDir: "{app}"
Name: "{group}\Expressive-examples"; Filename: "{app}\examples"; WorkingDir: "{app}\examples"

; Desktop Shortcuts
Name: "{userdesktop}\Expressive-GUI (English)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=en"; Tasks: desktopicon; WorkingDir: "{app}"
Name: "{userdesktop}\Expressive-GUI (中文)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=zh_CN"; Tasks: desktopicon; WorkingDir: "{app}"
Name: "{userdesktop}\Expressive-examples"; Filename: "{app}\examples"; Tasks: desktopicon; WorkingDir: "{app}\examples"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional tasks:"
