$ErrorActionPreference = "Stop"

# Legacy report runner (predates run_fx_daily_refresh.ps1). Path is derived
# from this script's location so it works regardless of where Mo_Dash lives.
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# Requires these user or machine environment variables:
#   GMAIL_USER
#   GMAIL_APP_PASSWORD
#
# Gmail app password setup:
# Google Account -> Security -> 2-Step Verification -> App passwords

py -3 -m fxnl.daily_binary_forecast_report `
  --source-xlsx "$ProjectRoot\batch_hold_full_export.xlsx" `
  --out-xlsx "$ProjectRoot\daily_binary_fx_forecast.xlsx" `
  --to "moinvestor7@gmail.com" `
  --min-accuracy 0.60 `
  --strong-accuracy 0.65 `
  --lookback-bars 1000 `
  --pca 5 `
  --family-filter "faster" `
  --align-to-next-bar `
  --send-email
