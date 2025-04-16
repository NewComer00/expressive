#define AppName "Expressive-GUI"
#define AppVersion GetEnv("TAG_NAME")

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\Expressive-GUI
DefaultGroupName=Expressive-GUI
OutputDir=..\dist
OutputBaseFilename=Expressive-GUI-{#AppVersion}-Installer
Compression=lzma
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
SetupIconFile=..\assets\icons\app.ico
PrivilegesRequiredOverridesAllowed=commandline dialog

[Files]
Source: "..\dist\Expressive-GUI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; Start Menu Shortcuts
Name: "{group}\Expressive-GUI (English)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=en"; WorkingDir: "{app}"
Name: "{group}\Expressive-GUI (中文)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=zh_CN"; WorkingDir: "{app}"
Name: "{group}\Expressive-examples"; Filename: "{app}\examples"; WorkingDir: "{app}\examples"; IconFilename: "{app}\assets\icons\examples.ico"

; Desktop Shortcuts
Name: "{autodesktop}\Expressive-GUI (English)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=en"; Tasks: desktopicon; WorkingDir: "{app}"
Name: "{autodesktop}\Expressive-GUI (中文)"; Filename: "{app}\expressive-gui.exe"; Parameters: "--lang=zh_CN"; Tasks: desktopicon; WorkingDir: "{app}"
Name: "{autodesktop}\Expressive-examples"; Filename: "{app}\examples"; Tasks: desktopicon; WorkingDir: "{app}\examples"; IconFilename: "{app}\assets\icons\examples.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional tasks:"
