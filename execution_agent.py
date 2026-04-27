"""
execution_agent.py — places bracket orders on Alpaca for approved trades.

Primary data source: trade_approvals.json  {"approvals": [...]}
Fallback chain for missing fields: compliance_report.json → risk_report.json → shortlist.json

Usage:
    python execution_agent.py          # execute all pending approved trades
    python execution_agent.py TICKER   # execute one specific ticker
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
import datetime
import pytz
from pathlib import Path
from typing import Optional

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

APPROVALS_PATH  = BASE_DIR / "trade_approvals.json"
COMPLIANCE_PATH = BASE_DIR / "compliance_report.json"
RISK_PATH       = BASE_DIR / "risk_report.json"
SHORTLIST_PATH  = BASE_DIR / "shortlist.json"
EXECUTED_PATH   = BASE_DIR / "executed_trades.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def now_et() -> str:
    et = pytz.timezone("America/New_York")
    return datetime.datetime.now(et).strftime("%Y-%m-%d %H:%M:%S ET")


def load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_client() -> tradeapi.REST:
    if not API_KEY or not SECRET_KEY:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    return tradeapi.REST(API_KEY, SECRET_KEY, base_url=BASE_URL, api_version="v2")


# ── Fallback trade detail lookups ─────────────────────────────────────────────

def _find_in_compliance(ticker: str) -> Optional[dict]:
    data = load_json(COMPLIANCE_PATH)
    if not data:
        return None
    for trade in data.get("cleared_trades", []):
        if trade.get("ticker", "").upper() == ticker:
            return trade
    return None


def _find_in_risk(ticker: str) -> Optional[dict]:
    data = load_json(RISK_PATH)
    if not data:
        return None
    for trade in data.get("trades", []):
        if trade.get("ticker", "").upper() == ticker:
            return trade
    return None


def _find_in_shortlist(ticker: str) -> Optional[dict]:
    data = load_json(SHORTLIST_PATH)
    if not data:
        return None
    items = data if isinstance(data, list) else []
    for trade in items:
        if trade.get("ticker", "").upper() == ticker:
            # Shortlist uses different key names for prices
            return {
                "ticker":    trade.get("ticker"),
                "direction": trade.get("direction", "LONG"),
                "entry":     trade.get("entry_price"),
                "stop":      trade.get("stop_loss"),
                "target":    trade.get("target_price"),
            }
    return None


def resolve_trade_fields(approval: dict) -> dict:
    """
    Return a complete trade record with entry/stop/target/shares/direction.

    Uses the approval record as the primary source. For any field that is
    still None, walks the fallback chain:
      compliance_report.json → risk_report.json → shortlist.json
    """
    ticker  = approval.get("ticker", "").upper()
    result  = dict(approval)
    needed  = ("entry", "stop", "target", "direction")
    missing = [k for k in needed if result.get(k) is None]

    if not missing:
        return result

    for source_fn in (_find_in_compliance, _find_in_risk, _find_in_shortlist):
        if not missing:
            break
        source = source_fn(ticker)
        if not source:
            continue
        for field in list(missing):
            val = source.get(field)
            if val is not None:
                result[field] = val
                missing.remove(field)

    return result


# ── Order placement ───────────────────────────────────────────────────────────

def place_bracket_order(api: tradeapi.REST, trade: dict) -> dict:
    """
    Submit a limit entry + stop-loss + take-profit bracket order.

    Alpaca bracket orders require whole-share quantities, so fractional
    share counts from the risk agent are rounded to the nearest integer.
    The OCO stop/target pair activates automatically once the entry fills.
    """
    ticker    = trade["ticker"].upper()
    entry     = round(float(trade["entry"]),  2)
    stop      = round(float(trade["stop"]),   2)
    target    = round(float(trade["target"]), 2)
    direction = trade.get("direction", "LONG").upper()
    side      = "buy" if direction == "LONG" else "sell"

    raw_shares = float(trade.get("shares") or 0)
    shares     = max(1, int(round(raw_shares)))

    print(f"\n[{ticker}] {direction} bracket order")
    print(f"  Entry (limit) : ${entry:.2f}  x {shares} shares"
          + (f"  (rounded from {raw_shares})" if raw_shares != shares else ""))
    print(f"  Stop loss     : ${stop:.2f}")
    print(f"  Take profit   : ${target:.2f}")

    order = api.submit_order(
        symbol=ticker,
        qty=shares,
        side=side,
        type="limit",
        time_in_force="gtc",
        limit_price=entry,
        order_class="bracket",
        stop_loss={"stop_price": stop},
        take_profit={"limit_price": target},
    )
    return order._raw


# ── Core execution logic ──────────────────────────────────────────────────────

def execute_ticker(
    api: tradeapi.REST,
    approval: dict,
    approvals_list: list,
    executed: dict,
) -> bool:
    """
    Execute a single approved trade from an approval record.

    Mutates approvals_list entries and executed in-place; the caller
    persists both to disk after each call.
    Returns True on success, False on any error.
    """
    ticker = approval.get("ticker", "").upper()
    trade  = resolve_trade_fields(approval)

    missing = [k for k in ("entry", "stop", "target") if not trade.get(k)]
    if missing:
        print(f"[{ticker}] Cannot execute — missing fields: {missing}")
        return False

    try:
        order_raw = place_bracket_order(api, trade)
    except Exception as e:
        print(f"[{ticker}] ERROR placing order: {e}")
        return False

    order_id   = order_raw.get("id", "unknown")
    raw_shares = float(trade.get("shares") or 0)
    shares     = max(1, int(round(raw_shares)))

    executed[ticker] = {
        "ticker":      ticker,
        "order_id":    order_id,
        "direction":   trade.get("direction"),
        "shares":      shares,
        "entry":       trade.get("entry"),
        "stop":        trade.get("stop"),
        "target":      trade.get("target"),
        "status":      order_raw.get("status"),
        "executed_at": now_et(),
    }

    for item in approvals_list:
        if item.get("ticker", "").upper() == ticker:
            item["executed"]    = True
            item["executed_at"] = now_et()
            item["order_id"]    = order_id
            break

    print(f"[{ticker}] Order placed. ID: {order_id}  Status: {order_raw.get('status')}")
    return True


# ── Approvals loader ──────────────────────────────────────────────────────────

def load_approvals() -> tuple:
    """
    Load trade_approvals.json. Returns (raw_data, approvals_list).

    Handles both the new list structure {"approvals": [...]} and the legacy
    dict structure {ticker: {approved, notes, decided_at}} so that any
    existing approval records keep working after the format migration.
    """
    raw = load_json(APPROVALS_PATH, default={})

    if isinstance(raw, dict) and "approvals" in raw:
        return raw, raw["approvals"]

    # Legacy dict structure — wrap it so we can iterate uniformly
    approvals_list = []
    for ticker, info in raw.items():
        if isinstance(info, dict):
            approvals_list.append({
                "ticker":   ticker.upper(),
                "approved": info.get("approved", False),
                "executed": info.get("executed", False),
                **info,
            })
    wrapped = {"approvals": approvals_list}
    return wrapped, approvals_list


# ── Main entry point ──────────────────────────────────────────────────────────

def run(filter_ticker: Optional[str] = None) -> None:
    """Process all pending approved trades, or a single ticker if specified."""
    print("=" * 56)
    print("      EXECUTION AGENT — ALPACA PAPER TRADING")
    print("=" * 56)

    try:
        api  = get_client()
        acct = api.get_account()
        print(f"\nAlpaca connected — status: {acct.status}  "
              f"buying power: ${float(acct.buying_power):,.2f}")
    except Exception as e:
        print(f"\nERROR connecting to Alpaca: {e}")
        sys.exit(1)

    approvals_raw, approvals_list = load_approvals()
    executed = load_json(EXECUTED_PATH, default={})

    pending = [
        entry for entry in approvals_list
        if entry.get("approved") and not entry.get("executed")
        and (filter_ticker is None
             or entry.get("ticker", "").upper() == filter_ticker.upper())
    ]

    if not pending:
        label = f" for {filter_ticker}" if filter_ticker else ""
        print(f"\nNo pending approved trades{label} to execute.")
        return

    tickers_str = ", ".join(e["ticker"] for e in pending)
    print(f"\nExecuting {len(pending)} trade(s): {tickers_str}\n")

    success_count = 0
    for approval in pending:
        ok = execute_ticker(api, approval, approvals_list, executed)
        # Persist after every attempt so partial progress is never lost
        save_json(EXECUTED_PATH,  executed)
        save_json(APPROVALS_PATH, approvals_raw)
        if ok:
            success_count += 1
        else:
            print(f"[{approval['ticker']}] Skipped.")

    print(f"\n{'=' * 56}")
    print(f"  Done — {success_count}/{len(pending)} orders placed.")
    print(f"  Confirmations saved to executed_trades.json")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    filter_ticker = sys.argv[1].upper() if len(sys.argv) > 1 else None
    run(filter_ticker)
