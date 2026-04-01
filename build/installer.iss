#define AppVersion GetEnv("VERSION")
#define Variant GetEnv("VARIANT")

[Setup]
AppName=Expressive
AppVersion={#AppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={autopf}\Expressive
DefaultGroupName=Expressive
OutputDir=..\dist
OutputBaseFilename=Expressive-{#AppVersion}-Windows-x64-{#Variant}
Compression=lzma
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
SetupIconFile=..\assets\icons\app.ico
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline

[Files]
Source: "..\dist\expressive\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; Start Menu Shortcuts
Name: "{group}\Expressive"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-ExecutionPolicy Bypass -NoExit -Command ""$env:PATH = '{app};' + $env:PATH; Set-Location '{app}'; Write-Host 'Expressive CLI ready. Run ' -NoNewline; Write-Host 'expressive --help' -ForegroundColor Cyan -NoNewline; Write-Host ' for usage.'"" "; WorkingDir: "{app}"; IconFilename: "{app}\assets\icons\cli.ico"
Name: "{group}\Expressive GUI"; Filename: "{app}\expressive-gui.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\icons\gui.ico"
Name: "{group}\Expressive Viewer"; Filename: "{app}\expressive-viewer.exe"; WorkingDir: "{app}"; IconFilename: "{app}\assets\icons\viewer.ico"
Name: "{group}\Expressive Examples"; Filename: "{app}\examples"; WorkingDir: "{app}\examples"; IconFilename: "{app}\assets\icons\examples.ico"

; Desktop Shortcuts
Name: "{autodesktop}\Expressive"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-ExecutionPolicy Bypass -NoExit -Command ""$env:PATH = '{app};' + $env:PATH; Set-Location '{app}'; Write-Host 'Expressive CLI ready. Run ' -NoNewline; Write-Host 'expressive --help' -ForegroundColor Cyan -NoNewline; Write-Host ' for usage.'"" "; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\assets\icons\cli.ico"
Name: "{autodesktop}\Expressive GUI"; Filename: "{app}\expressive-gui.exe"; Tasks: desktopicon; WorkingDir: "{app}"; IconFilename: "{app}\assets\icons\gui.ico"
Name: "{autodesktop}\Expressive Viewer"; Filename: "{app}\expressive-viewer.exe"; Tasks: desktopicon; WorkingDir: "{app}"; IconFilename: "{app}\assets\icons\viewer.ico"
Name: "{autodesktop}\Expressive Examples"; Filename: "{app}\examples"; Tasks: desktopicon; WorkingDir: "{app}\examples"; IconFilename: "{app}\assets\icons\examples.ico"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional tasks:"
