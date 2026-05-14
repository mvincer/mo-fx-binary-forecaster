$ErrorActionPreference = "Stop"

$TaskName = "Daily FX Binary Forecast Report"
$ProjectRoot = "C:\Users\mrmhr\OneDrive\Documents\Python\FX_NonLinear_Forecast_Direction"
$ScriptPath = Join-Path $ProjectRoot "run_daily_binary_fx_report.ps1"

if (-not (Test-Path $ScriptPath)) {
  throw "Missing report script: $ScriptPath"
}

# Earliest sensible daily time after FX daily close (5 PM New York): 6:15 PM local/New York time.
# If your raw data provider refreshes slower, move this to 7:00 PM or early morning.
$Action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`"" `
  -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At 6:15PM

$Settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $Action `
  -Trigger $Trigger `
  -Settings $Settings `
  -Description "Build and email the daily binary FX forecast report." `
  -Force

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Remember to set GMAIL_USER and GMAIL_APP_PASSWORD in your user environment."
