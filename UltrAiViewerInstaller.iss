#define MyAppName "UltrAi Nerve Viewer"
#define MyAppPublisher "UltrAi"
#define MyAppExeName GetEnv("APP_NAME") + ".exe"
#define MyAppVersion GetEnv("ULTAI_VERSION")
#define MyAppDistDir AddBackslash(SourcePath) + "dist\" + GetEnv("APP_NAME")
#define MyAppOutputBaseFilename GetEnv("APP_NAME") + "-installer"

[Setup]
AppId={{59AEE3D2-4AC1-4C7A-A3A7-57D5049119AE}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename={#MyAppOutputBaseFilename}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[InstallDelete]
; Clear the previous version's files so old exes and models don't linger after an upgrade
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\UltrAiNerveViewer-v*.exe"

[Files]
Source: "{#MyAppDistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
