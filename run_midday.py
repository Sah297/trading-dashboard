import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
import json
import time
import datetime
import subprocess
import pytz
from dotenv import load_dotenv

load_dotenv()

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
PYTHON             = sys.executable
ACCOUNT_STATE_PATH = os.path.join(BASE_DIR, "account_state.json")
RISK_PATH          = os.path.join(BASE_DIR, "risk_report.json")
COMPLIANCE_PATH    = os.path.join(BASE_DIR, "compliance_report.json")

MIN_BUYING_POWER = 500.0

MIDDAY_STEPS = [
    ("Step 1: News & Watchlist",   "news_agent.py"),
    ("Step 2: Technical Analysis", "technical_agent.py"),
    ("Step 3: Risk Management",    "risk_agent.py"),
    ("Step 4: Compliance Check",   "compliance_agent.py"),
]

# risk_agent and compliance_agent both respect ACCOUNT_SIZE_OVERRIDE
NEEDS_OVERRIDE = {"risk_agent.py", "compliance_agent.py"}


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def run_step(label: str, script: str, extra_env: dict | None = None) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [PYTHON, os.path.join(BASE_DIR, script)],
        cwd=BASE_DIR,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}. Pipeline stopped.")
        sys.exit(result.returncode)
    print(f"\n{label.split(':')[0]} complete — proceeding to next step")


def print_midday_summary(run_ts: str, account_state: dict, buying_power: float) -> None:
    W = 96

    risk_report       = load_json(RISK_PATH, {})
    compliance_report = load_json(COMPLIANCE_PATH, {})

    trades   = risk_report.get("trades", [])
    regime   = risk_report.get("regime", "UNKNOWN")
    risk_pct = risk_report.get("risk_pct", 0)
    proj     = risk_report.get("projections", {})

    cleared     = compliance_report.get("cleared_trades", [])
    reviewed    = compliance_report.get("trades_reviewed", 0)
    market      = compliance_report.get("market_hours", {})
    cleared_set = {t["ticker"] for t in cleared}

    approved  = [t for t in trades if t.get("status") == "APPROVED"]
    take_it   = [t for t in approved if "TAKE"   in (t.get("claude_verdict") or "").upper()]
    reduce_sz = [t for t in approved if "REDUCE" in (t.get("claude_verdict") or "").upper()]

    total_risk    = sum(float(t.get("estimated_risk") or 0) for t in approved)
    risk_pct_bp   = (total_risk / buying_power * 100) if buying_power else 0
    realistic_m1  = proj.get("realistic_month")

    open_pos      = account_state.get("open_positions", [])
    daily_pnl     = account_state.get("daily_pnl", 0)
    daily_pnl_pct = account_state.get("daily_pnl_pct", 0)
    d_sign        = "+" if daily_pnl >= 0 else ""

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
    print(f"  MIDDAY PIPELINE COMPLETE — {run_ts}")
    print(f"  Regime: {regime}  |  Buying Power Used: ${buying_power:,.2f}  |  Risk/trade: {risk_pct * 100:.1f}%")
    print(f"{'=' * W}")

    # ── Live account status ────────────────────────────────────────────────────
    print(f"\n  ACCOUNT STATUS  (Alpaca Paper — live)")
    print(f"  {bar}")
    print(f"  Account Value    : ${account_state.get('account_value', 0):,.2f}")
    print(f"  Buying Power     : ${buying_power:,.2f}")
    print(f"  Daily P&L        : {d_sign}${daily_pnl:,.2f}  ({d_sign}{daily_pnl_pct:.2f}%)")
    if open_pos:
        tickers = ", ".join(p["ticker"] for p in open_pos)
        print(f"  Open Positions   : {len(open_pos)}  ({tickers})")
    else:
        print(f"  Open Positions   : 0  (none)")

    # ── TAKE IT ───────────────────────────────────────────────────────────────
    print(f"\n  TAKE IT — execute these trades now")
    print(f"  {bar}")
    if take_it:
        print(f"  {hdr}")
        print(f"  {bar}")
        for t in take_it:
            print(f"  {trade_row(t)}")
    else:
        print("  None")

    # ── REDUCE SIZE ───────────────────────────────────────────────────────────
    print(f"\n  REDUCE SIZE — trade smaller than calculated position")
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
    print(f"  Capital at risk (new trades)     : ${total_risk:,.2f}  ({risk_pct_bp:.1f}% of buying power)")
    if realistic_m1 is not None:
        sign = "+" if realistic_m1 >= 0 else ""
        print(f"  Month 1 realistic (this batch)   : {sign}${realistic_m1:,.2f}")
    print(f"  {bar}")
    print(f"  Compliance cleared               : {len(cleared)}/{reviewed} trades")
    print(f"  Market status                    : {market.get('status','?')}"
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
    print("       TRADING AGENT — MIDDAY PIPELINE")
    print(f"       Started: {run_ts}")
    print("=" * 62)

    # ── Step 0: Sync account from Alpaca ──────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  Step 0: Account Sync")
    print(f"{'=' * 62}")
    result = subprocess.run(
        [PYTHON, os.path.join(BASE_DIR, "account_sync.py")],
        cwd=BASE_DIR,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print("ERROR: account_sync.py failed. Cannot determine buying power. Stopping.")
        sys.exit(1)
    print("Step 0 complete — account state refreshed")

    # ── Buying power gate ─────────────────────────────────────────────────────
    account_state = load_json(ACCOUNT_STATE_PATH, {})
    buying_power  = float(account_state.get("buying_power", 0))

    if buying_power < MIN_BUYING_POWER:
        print(
            f"\nInsufficient buying power for new positions — "
            f"${buying_power:,.2f} available, "
            f"${MIN_BUYING_POWER:,.0f} minimum required."
        )
        print("Skipping midday run.")
        sys.exit(0)

    print(f"\nBuying power: ${buying_power:,.2f} — proceeding with midday pipeline.\n")

    # ── Midday steps ──────────────────────────────────────────────────────────
    override_env = {"ACCOUNT_SIZE_OVERRIDE": str(int(buying_power))}

    for i, (label, script) in enumerate(MIDDAY_STEPS):
        extra = override_env if script in NEEDS_OVERRIDE else None
        run_step(label, script, extra_env=extra)
        if i < len(MIDDAY_STEPS) - 1:
            print("Waiting 10 seconds before next step...")
            time.sleep(10)

    print("\n" + "=" * 62)
    print("  ALL MIDDAY STEPS COMPLETE")
    print("=" * 62)
    print_midday_summary(run_ts, account_state, buying_power)


if __name__ == "__main__":
    main()
