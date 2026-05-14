"""
Batch PDF report using **ETF Forecaster** ``fx_data_collect`` panels (when ``FXNL_USE_REPO_DATA``)
and **rolling OOS block statistics** (default block size 100).

Equivalent to::

    py -m fxnl.batch_strategy_report --rolling-chunk 100 -o repo_strategy_report.pdf ...

Run from ``FX_NonLinear_Forecast_Direction`` so imports resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    orig0 = sys.argv[0]
    av = list(argv if argv is not None else sys.argv[1:])
    here = Path(__file__).resolve().parents[1]
    pre: list[str] = []
    if "--rolling-chunk" not in av:
        pre.extend(["--rolling-chunk", "100"])
    if "-o" not in av and "--output" not in av:
        pre.extend(["-o", str(here / "repo_strategy_report.pdf")])
    sys.argv = [orig0] + pre + av
    from fxnl.batch_strategy_report import main as batch_main

    batch_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
