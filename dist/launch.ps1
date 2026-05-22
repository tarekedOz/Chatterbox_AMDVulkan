<#
  Chatterbox TTS launcher. On first run downloads the model weights
  (fetch-models.ps1), then starts the server and opens the web UI.
  This is the target of the Start Menu / desktop shortcut.
#>
$ErrorActionPreference = 'Stop'
$dir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $dir

# First-run (or repair): ensure all model weights are present + verified.
& (Join-Path $dir 'fetch-models.ps1') -InstallDir $dir
if ($LASTEXITCODE -ne 0) {
  Write-Host ''
  Read-Host 'Model download failed (see above). Press Enter to exit'
  exit 1
}

$exe  = Join-Path $dir 'chatterbox-server.exe'
$addr = '127.0.0.1:8087'
$proc = Start-Process -FilePath $exe -ArgumentList '--config', 'config.yaml' `
                      -WorkingDirectory $dir -PassThru

# Wait for the server to come up, then open the browser.
for ($i = 0; $i -lt 120; $i++) {
  try { Invoke-WebRequest -UseBasicParsing "http://$addr/health" -TimeoutSec 1 | Out-Null; break }
  catch { Start-Sleep -Milliseconds 500 }
}
Start-Process "http://$addr/"

Write-Host "Chatterbox TTS running at http://$addr/  (close this window to stop)"
Wait-Process -Id $proc.Id
