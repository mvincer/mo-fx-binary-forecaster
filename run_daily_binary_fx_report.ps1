$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\mrmhr\OneDrive\Documents\Python\FX_NonLinear_Forecast_Direction"
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
