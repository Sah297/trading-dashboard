import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import sys
import datetime
import pytz
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

BASE_DIR    = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(BASE_DIR, "account_state.json")
PAPER_URL   = "https://paper-api.alpaca.markets"


def get_client() -> tradeapi.REST:
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    return tradeapi.REST(api_key, secret_key, base_url=PAPER_URL, api_version="v2")


def _read_manual_budget() -> float:
    """Return the trading budget from settings.json, or 0 if not set."""
    path = os.path.join(BASE_DIR, "settings.json")
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        return float(json.load(f).get("trading_budget") or 0)


def fetch_state(api: tradeapi.REST) -> dict:
    account   = api.get_account()
    positions = api.list_positions()

    equity       = float(account.equity)
    buying_power = float(account.buying_power)
    cash         = float(account.cash)
    last_equity  = float(account.last_equity)

    daily_pnl     = equity - last_equity
    daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0.0

    open_positions    = []
    total_unreal_pnl  = 0.0
    committed_capital = 0.0

    for pos in positions:
        unreal_pl   = float(pos.unrealized_pl)
        unreal_plpc = float(pos.unrealized_plpc) * 100
        market_val  = float(pos.market_value)
        total_unreal_pnl  += unreal_pl
        committed_capital += market_val
        open_positions.append({
            "ticker":          pos.symbol,
            "qty":             float(pos.qty),
            "side":            pos.side,
            "avg_entry_price": round(float(pos.avg_entry_price), 2),
            "current_price":   round(float(pos.current_price), 2),
            "market_value":    round(market_val, 2),
            "unrealized_pl":   round(unreal_pl, 2),
            "unrealized_plpc": round(unreal_plpc, 2),
        })

    # Manual budget (from settings.json); fallback to all available cash
    manual_budget = _read_manual_budget() or cash
    remaining_budget    = max(0.0, manual_budget - committed_capital)
    available_to_invest = remaining_budget

    et = pytz.timezone("America/New_York")
    last_updated = datetime.datetime.now(et).strftime("%Y-%m-%d %H:%M ET")

    return {
        "account_value":       round(equity, 2),
        "cash_available":      round(cash, 2),
        "buying_power":        round(buying_power, 2),
        "manual_budget":       round(manual_budget, 2),
        "committed_capital":   round(committed_capital, 2),
        "remaining_budget":    round(remaining_budget, 2),
        "available_to_invest": round(available_to_invest, 2),
        "open_positions":      open_positions,
        "open_positions_count": len(open_positions),
        "open_positions_value": round(committed_capital, 2),
        "daily_pnl":           round(daily_pnl, 2),
        "daily_pnl_pct":       round(daily_pnl_pct, 2),
        "total_pnl":           round(total_unreal_pnl, 2),
        "last_updated":        last_updated,
    }


def print_summary(state: dict) -> None:
    positions   = state["open_positions"]
    tickers     = [p["ticker"] for p in positions]
    daily_pnl   = state["daily_pnl"]
    d_sign      = "+" if daily_pnl >= 0 else ""
    daily_pct   = state["daily_pnl_pct"]
    n_pos       = len(positions)
    pos_str     = f"({n_pos} position{'s' if n_pos != 1 else ''})" if n_pos else ""

    W = 46
    print("\n" + "=" * W)
    print("  CAPITAL MANAGEMENT SUMMARY")
    print("=" * W)
    print(f"  Real Alpaca Cash    : ${state['cash_available']:>12,.2f}")
    print(f"  Manual Budget Set   : ${state['manual_budget']:>12,.2f}")
    print(f"  In Open Trades      : ${state['committed_capital']:>12,.2f}  {pos_str}")
    print(f"  {'─' * (W - 2)}")
    print(f"  Available to Invest : ${state['available_to_invest']:>12,.2f}  ← agent uses this")
    print("=" * W)
    print(f"  Daily P&L: {d_sign}${abs(daily_pnl):,.2f}  ({d_sign}{daily_pct:.2f}%)")
    if tickers:
        print(f"  Open positions: {', '.join(tickers)}")
    print(f"  Last Updated: {state['last_updated']}")

    if positions:
        print()
        col = "  {:<6}  {:>5}  {:>8}  {:>8}  {:>10}  {:>7}"
        print(col.format("Ticker", "Qty", "Entry", "Now", "P&L $", "P&L %"))
        print(f"  {'─' * (W - 2)}")
        for p in positions:
            sign = "+" if p["unrealized_pl"] >= 0 else ""
            print(col.format(
                p["ticker"],
                f"{p['qty']:.0f}",
                f"${p['avg_entry_price']:.2f}",
                f"${p['current_price']:.2f}",
                f"{sign}${p['unrealized_pl']:.2f}",
                f"{sign}{p['unrealized_plpc']:.1f}%",
            ))


def main():
    print("=" * 56)
    print("      ACCOUNT SYNC — ALPACA PAPER TRADING")
    print("=" * 56)

    try:
        api = get_client()
        print("\nConnecting to Alpaca paper trading API...")
        state = fetch_state(api)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(state, f, indent=2)

    print_summary(state)
    print(f"Account state saved to account_state.json")


if __name__ == "__main__":
    main()
