<#
  Uninstall Chatterbox TTS (the PowerShell-installed copy). Stops the
  server, removes the install directory (including the downloaded model
  weights), and deletes the Start Menu / desktop shortcuts.

  This is the target of the "Uninstall Chatterbox TTS" Start Menu shortcut.
  The Inno Setup (.exe) install has its own uninstaller in Add/Remove
  Programs and does not use this script.
#>
$ErrorActionPreference = 'SilentlyContinue'

# This script lives in the install dir; resolve it, then step out so the
# directory isn't held open (current dir) while we delete it.
$installDir = if ($PSScriptRoot) { $PSScriptRoot }
              else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $env:TEMP

Write-Host "Uninstalling Chatterbox TTS from $installDir"

# Stop the server if it's running (releases the exe + any open weights).
Get-Process chatterbox-server -ErrorAction SilentlyContinue | Stop-Process -Force

# Remove shortcuts.
$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
Remove-Item -Force (Join-Path $startMenu 'Chatterbox TTS.lnk')
Remove-Item -Force (Join-Path $startMenu 'Uninstall Chatterbox TTS.lnk')
Remove-Item -Force (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Chatterbox TTS.lnk')

# Remove the install dir: server, scripts, config, and the ~1.4 GB weights.
Remove-Item -Recurse -Force $installDir

Write-Host "Chatterbox TTS has been uninstalled."
