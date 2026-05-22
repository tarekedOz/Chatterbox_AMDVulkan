<#
.SYNOPSIS
  Download + verify the Chatterbox model weights listed in a manifest.
  Run on first launch (the installer ships only the ~8 MB binary).

.PARAMETER ManifestPath
  Path to models.manifest.json (default: alongside this script).

.PARAMETER InstallDir
  Root the manifest 'dest' paths are resolved against (default: script dir).

.PARAMETER BaseUrl
  Override the manifest base_url. Use this to point at a different host
  (e.g. a mirror or a self-hosted package registry).

.PARAMETER Token
  Optional access token for private download hosts. Sent as an
  'Authorization: Bearer' header (PRIVATE-TOKEN for api/v4 registries).
  Not needed for public release assets, which download anonymously.

.NOTES
  Idempotent: files already present with a matching sha256 are skipped, so
  re-running resumes a partial install. Exit code 0 on success, 1 on error.
#>
[CmdletBinding()]
param(
  [string]$ManifestPath,
  [string]$InstallDir,
  [string]$BaseUrl,
  [string]$Token
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # huge speedup for Invoke-WebRequest

# Resolve the script's own directory robustly ($PSScriptRoot can be empty
# depending on how the script is invoked).
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot }
             elseif ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path }
             else { (Get-Location).Path }
if (-not $ManifestPath) { $ManifestPath = Join-Path $scriptDir 'models.manifest.json' }
if (-not $InstallDir)   { $InstallDir   = $scriptDir }

function Get-Sha256($path) {
  # Use .NET directly rather than Get-FileHash so we don't depend on the
  # Microsoft.PowerShell.Utility module autoloading (PSModulePath can be
  # clobbered in some environments). Streams the file -> low memory.
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try {
    $fs = [System.IO.File]::OpenRead($path)
    try { (-join ($sha.ComputeHash($fs) | ForEach-Object { $_.ToString('x2') })) }
    finally { $fs.Dispose() }
  } finally { $sha.Dispose() }
}

try {
  if (-not (Test-Path $ManifestPath)) { throw "manifest not found: $ManifestPath" }
  $manifest = Get-Content -Raw $ManifestPath | ConvertFrom-Json
  $base = if ($BaseUrl) { $BaseUrl } else { $manifest.base_url }
  $base = $base.TrimEnd('/')

  $headers = @{}
  if ($Token) {
    if ($base -match '/api/v4/') {
      $headers['PRIVATE-TOKEN'] = $Token
    } else {
      $headers['Authorization'] = "Bearer $Token"
    }
  }

  $total = $manifest.files.Count
  $i = 0
  foreach ($f in $manifest.files) {
    $i++
    $dest = Join-Path $InstallDir $f.dest
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Force -Path $destDir | Out-Null }

    if ((Test-Path $dest) -and ((Get-Sha256 $dest) -eq $f.sha256.ToLower())) {
      Write-Host ("[{0}/{1}] {2} - present, checksum OK" -f $i, $total, $f.name)
      continue
    }

    $url = "$base/$($f.name)"
    $part = "$dest.part"
    $mb = [math]::Round($f.size / 1MB)
    Write-Host ("[{0}/{1}] {2} - downloading {3} MB from {4}" -f $i, $total, $f.name, $mb, $url)
    Invoke-WebRequest -Uri $url -OutFile $part -Headers $headers -MaximumRedirection 5 -UseBasicParsing

    $got = Get-Sha256 $part
    if ($got -ne $f.sha256.ToLower()) {
      Remove-Item -Force $part -ErrorAction SilentlyContinue
      throw "checksum mismatch for $($f.name): expected $($f.sha256), got $got"
    }
    Move-Item -Force $part $dest
    Write-Host ("        verified -> {0}" -f $dest)
  }

  Write-Host "All models present and verified."
  exit 0
}
catch {
  Write-Error "fetch-models failed: $($_.Exception.Message)"
  exit 1
}
