<#
.SYNOPSIS
  One-call installer for Chatterbox TTS (Windows x64).

  Typical use (public release, anonymous):
      irm https://github.com/tarekedOz/Chatterbox_AMDVulkan/releases/download/v1/install.ps1 | iex

  Advanced / testing (override host, private host needs a token):
      .\install.ps1 -BaseUrl <release-or-registry-url> [-Token <pat>] `
                    [-InstallDir <dir>] [-NoLaunch]

  Downloads the ~8 MB app package (self-contained server + scripts +
  manifest), then fetch-models.ps1 downloads + verifies the ~1.4 GB
  weights, makes a Start Menu shortcut, and launches.
#>
[CmdletBinding()]
param(
  [string]$BaseUrl = "https://github.com/tarekedOz/Chatterbox_AMDVulkan/releases/download/v1",
  [string]$Token,
  [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Chatterbox TTS'),
  [string]$Package = 'chatterbox-tts-win-x64.zip',
  [switch]$NoLaunch
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$base = $BaseUrl.TrimEnd('/')
$headers = @{}
if ($Token) {
  if ($base -match '/api/v4/') { $headers['PRIVATE-TOKEN'] = $Token }
  else { $headers['Authorization'] = "Bearer $Token" }
}

New-Item -ItemType Directory -Force $InstallDir | Out-Null
Write-Host "Installing Chatterbox TTS -> $InstallDir"

# 1) App package: self-contained server.exe + fetch-models.ps1 + launch.ps1
#    + models.manifest.json + config.yaml.
$zip = Join-Path ([System.IO.Path]::GetTempPath()) $Package
Write-Host "Downloading $Package ..."
Invoke-WebRequest -Uri "$base/$Package" -OutFile $zip -Headers $headers -UseBasicParsing -MaximumRedirection 5
Expand-Archive -Path $zip -DestinationPath $InstallDir -Force
Remove-Item $zip -Force

# 2) Model weights (sha256-verified; skips files already present).
$fetch = Join-Path $InstallDir 'fetch-models.ps1'
$fetchParams = @{ InstallDir = $InstallDir; BaseUrl = $base }
if ($Token) { $fetchParams['Token'] = $Token }
& $fetch @fetchParams
if ($LASTEXITCODE -ne 0) { throw "model download failed" }

# 3) Start Menu shortcut -> launcher.
$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$lnk = Join-Path $startMenu 'Chatterbox TTS.lnk'
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = 'powershell.exe'
$sc.Arguments = "-ExecutionPolicy Bypass -NoProfile -File `"$InstallDir\launch.ps1`""
$sc.WorkingDirectory = $InstallDir
$sc.Description = 'Chatterbox TTS'
$sc.Save()

Write-Host "Installed. Start Menu shortcut: $lnk"
if (-not $NoLaunch) { & (Join-Path $InstallDir 'launch.ps1') }
