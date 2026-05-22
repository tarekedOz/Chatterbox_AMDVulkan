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
  # Advanced: override BOTH the app-package host and the weights host (e.g.
  # a private registry). Left unset for the public flow, where the app
  # package comes from the v1 release and the weights come from whatever
  # base_url the manifest declares (models-v1).
  [string]$BaseUrl,
  [string]$Token,
  [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Chatterbox TTS'),
  [string]$Package = 'chatterbox-tts-win-x64.zip',
  [switch]$NoLaunch
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# App-package host: the v1 release by default; -BaseUrl overrides it.
$pkgBase = if ($BaseUrl) { $BaseUrl.TrimEnd('/') }
           else { 'https://github.com/tarekedOz/Chatterbox_AMDVulkan/releases/download/v1' }
$headers = @{}
if ($Token) {
  if ($pkgBase -match '/api/v4/') { $headers['PRIVATE-TOKEN'] = $Token }
  else { $headers['Authorization'] = "Bearer $Token" }
}

New-Item -ItemType Directory -Force $InstallDir | Out-Null
Write-Host "Installing Chatterbox TTS -> $InstallDir"

# 1) App package: self-contained server.exe + fetch-models.ps1 + launch.ps1
#    + models.manifest.json + config.yaml.
$zip = Join-Path ([System.IO.Path]::GetTempPath()) $Package
Write-Host "Downloading $Package ..."
Invoke-WebRequest -Uri "$pkgBase/$Package" -OutFile $zip -Headers $headers -UseBasicParsing -MaximumRedirection 5
Expand-Archive -Path $zip -DestinationPath $InstallDir -Force
Remove-Item $zip -Force

# 2) Model weights (sha256-verified; skips files already present). Let the
#    manifest's base_url (models-v1) win unless the caller overrode the host.
$fetch = Join-Path $InstallDir 'fetch-models.ps1'
$fetchParams = @{ InstallDir = $InstallDir }
if ($BaseUrl) { $fetchParams['BaseUrl'] = $pkgBase }
if ($Token)   { $fetchParams['Token']   = $Token }
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
