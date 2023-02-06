; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

[Setup]
AppVersion=2.1.1
AppName=ARGUS
AppContact=Kitware, Inc.
AppPublisher=Kitware, Inc.
AppPublisherURL=https://www.kitware.com/
DefaultDirName={autopf}\ARGUS
DefaultGroupName=ARGUS
ChangesEnvironment=yes
OutputBaseFilename=ARGUS_Installer

[Files]
Source: "nssm.exe"; DestDir: "{app}\scripts"
Source: "scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs
; BeforeInstall: stop running service
Source: "dist\argus\*"; DestDir: "{app}\argus"; Flags: recursesubdirs; BeforeInstall: TeardownService('ARGUS')
Source: "bin\*"; DestDir: "{app}\bin"; Flags: recursesubdirs;

[Registry]
; {app}\bin should match the CurUninstallStepChanged entry below
Root: HKLM; \
  Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; \
  ValueType: expandsz; \
  ValueName: "Path"; \
  ValueData: "{olddata};{app}\bin"; \
  Check: IsInSystemPath(ExpandConstant('{app}\bin'))

[Run]
Filename: "{app}\scripts\postinstall.bat"; Flags: runhidden

[UninstallRun]
Filename: "{app}\scripts\preuninstall.bat"; RunOnceId: "ARGUSPreUninstall"; Flags: runhidden

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
const EnvKey = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

procedure TeardownService(const SvcName: string);
var
  ResultCode: Integer;
begin
  Exec(ExpandConstant('{sys}\sc.exe'), 'stop ' + SvcName, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function IsInSystemPath(const Entry: string) : boolean;
var
  Path: string;
begin
  if RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvKey, 'Path', Path) then
    Result := Pos(';' + Uppercase(Entry) + ';', ';' + Uppercase(Path) + ';') = 0
  else
    Result := True;
end;

procedure RemoveFromPath(const Entry: string);
var
  Path: string;
  Index: Integer;
begin
  if RegQueryStringValue(HKEY_LOCAL_MACHINE, EnvKey, 'Path', Path) then
  begin
    Index := Pos(';' + Uppercase(Entry) + ';', ';' + Uppercase(Path) + ';');
    if Index > 0 then
    begin
      { Delete leading semicolon if not first item }
      if Index > 1 then
        Index := Index - 1;
      Delete(Path, Index, Length(Entry) + 1);
      RegWriteStringValue(HKEY_LOCAL_MACHINE, EnvKey, 'Path', Path);
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    { this should be the same as in the Registry install step above }
    RemoveFromPath(ExpandConstant('{app}\bin'));
end;
