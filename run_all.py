import os
import sys
import json
import time
import datetime
import subprocess
import pytz

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PYTHON           = sys.executable
RISK_PATH        = os.path.join(BASE_DIR, "risk_report.json")
COMPLIANCE_PATH  = os.path.join(BASE_DIR, "compliance_report.json")

STEPS = [
    ("Step 1: Macro Analysis",      "macro_agent.py"),
    ("Step 2: Sector Rotation",     "sector_agent.py"),
    ("Step 3: News & Watchlist",    "news_agent.py"),
    ("Step 4: Technical Analysis",  "technical_agent.py"),
    ("Step 5: Risk Management",     "risk_agent.py"),
    ("Step 6: Compliance Check",    "compliance_agent.py"),
]


def run_step(label: str, script: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    result = subprocess.run(
        [PYTHON, os.path.join(BASE_DIR, script)],
        cwd=BASE_DIR,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}. Pipeline stopped.")
        sys.exit(result.returncode)
    print(f"\n{label.split(':')[0]} complete — proceeding to next step")


def print_final_summary(run_ts: str) -> None:
    W = 96

    risk_report       = {}
    compliance_report = {}
    if os.path.exists(RISK_PATH):
        with open(RISK_PATH) as f:
            risk_report = json.load(f)
    if os.path.exists(COMPLIANCE_PATH):
        with open(COMPLIANCE_PATH) as f:
            compliance_report = json.load(f)

    trades   = risk_report.get("trades", [])
    regime   = risk_report.get("regime", "UNKNOWN")
    risk_pct = risk_report.get("risk_pct", 0)
    account  = risk_report.get("account_size", 0)
    proj     = risk_report.get("projections", {})

    cleared      = compliance_report.get("cleared_trades", [])
    reviewed     = compliance_report.get("trades_reviewed", 0)
    market       = compliance_report.get("market_hours", {})
    cleared_set  = {t["ticker"] for t in cleared}

    approved  = [t for t in trades if t.get("status") == "APPROVED"]
    take_it   = [t for t in approved if "TAKE"   in (t.get("claude_verdict") or "").upper()]
    reduce_sz = [t for t in approved if "REDUCE" in (t.get("claude_verdict") or "").upper()]

    total_risk    = sum(float(t.get("estimated_risk") or 0) for t in approved)
    risk_pct_acct = (total_risk / account * 100) if account else 0
    realistic_m1  = proj.get("realistic_month")

    def fmt(v) -> str:
        return str(v or "N/A").replace(",", "")[:10]

    col = "{:<6}  {:<5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:<10}"
    hdr = col.format("Ticker", "Dir", "Entry", "Stop", "Target",
                     "Shares", "Risk $", "Profit $", "Cleared?")
    bar = "─" * (W - 2)

    def trade_row(t: dict) -> str:
        tag = "YES" if t.get("ticker") in cleared_set else "no"
        return col.format(
            t.get("ticker", "N/A"),
            (t.get("direction") or "N/A")[:5],
            fmt(t.get("entry")),
            fmt(t.get("stop")),
            fmt(t.get("target")),
            str(t.get("shares") or "N/A")[:8],
            f"${t.get('estimated_risk') or 'N/A'}"[:8],
            f"${t.get('estimated_profit') or 'N/A'}"[:10],
            tag,
        )

    print(f"\n{'=' * W}")
    print(f"  TRADING PIPELINE COMPLETE — {run_ts}")
    print(f"  Regime: {regime}  |  Account: ${account:,}  |  Risk/trade: {risk_pct * 100:.1f}%")
    print(f"{'=' * W}")

    # ── TAKE IT ───────────────────────────────────────────────────────────────
    print(f"\n  TAKE IT — execute these trades")
    print(f"  {bar}")
    if take_it:
        print(f"  {hdr}")
        print(f"  {bar}")
        for t in take_it:
            print(f"  {trade_row(t)}")
    else:
        print("  None")

    # ── REDUCE SIZE ───────────────────────────────────────────────────────────
    print(f"\n  REDUCE SIZE — trade smaller than the calculated position")
    print(f"  {bar}")
    if reduce_sz:
        print(f"  {hdr}")
        print(f"  {bar}")
        for t in reduce_sz:
            print(f"  {trade_row(t)}")
    else:
        print("  None")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n  {bar}")
    print(f"  Total capital at risk    : ${total_risk:,.2f}  ({risk_pct_acct:.1f}% of account)")
    if realistic_m1 is not None:
        sign = "+" if realistic_m1 >= 0 else ""
        print(f"  Month 1 realistic        : {sign}${realistic_m1:,.2f}"
              f"  →  account ${account + realistic_m1:,.2f}")

    print(f"  {bar}")
    print(f"  Compliance cleared       : {len(cleared)}/{reviewed} trades")
    print(f"  Market status            : {market.get('status','?')}"
          f"  ({market.get('current_time_et','?')})")

    print(f"\n  {bar}")
    print("  NEXT STEPS: These trades are cleared.")
    print("  Paper trading mode — log manually in Alpaca paper trading dashboard.")
    print("  Live mode — the execution agent will place these.")
    print(f"{'=' * W}\n")


def main():
    et     = pytz.timezone("America/New_York")
    run_ts = datetime.datetime.now(et).strftime("%Y-%m-%d %H:%M ET")

    print("=" * 62)
    print("       TRADING AGENT — FULL PIPELINE")
    print(f"       Started: {run_ts}")
    print("=" * 62)

    for i, (label, script) in enumerate(STEPS):
        run_step(label, script)
        if i < len(STEPS) - 1:
            print("Waiting 10 seconds before next step...")
            time.sleep(10)

    print("\n" + "=" * 62)
    print("  ALL STEPS COMPLETE")
    print("=" * 62)
    print_final_summary(run_ts)


if __name__ == "__main__":
    main()
