# FX Non-Linear Forecast for Direction

Directional forecasting on **daily** FX using the **same cached FXCM + FRED + technical features** as the sibling project **Simple Regression Based (FX)** (`../Currencies/`).

**Targets:** (1) **binary** — up vs down on the **h-day** forward simple return; (2) **ternary** — &lt;−0.5%, mid, &gt;+0.5%. **Leads:** 1–10 only.

**Split:** first **50%** of rows in-sample (fit), last **50%** out-of-sample (test). Cross-pair technical columns merged; all `*_close` dropped from **X**.

**Run**

```bash
cd FX_NonLinear_Forecast_Direction
pip install -r requirements.txt
py -m streamlit run app.py
```

Requires populated **`Currencies/data/cache/`**. Optional `.env` in `Currencies/` for API keys.

**Note:** This app’s Python package is **`fxnl/`** (not `src/`), so it does not collide with the Currencies project’s **`src`** package when both are on `sys.path`.
