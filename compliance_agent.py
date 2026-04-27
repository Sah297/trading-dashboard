import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import re
import json
import sys
import time
import datetime
import contextlib
import pytz
from dotenv import load_dotenv
import anthropic
import yfinance as yf

load_dotenv()


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ============================================================
# EDIT YOUR COMPLIANCE SETTINGS HERE
# ============================================================
_ACCOUNT_SIZE_DEFAULT = 100000
_account_state_path   = os.path.join(os.path.dirname(__file__), "account_state.json")

if os.getenv("ACCOUNT_SIZE_OVERRIDE"):
    ACCOUNT_SIZE = int(float(os.getenv("ACCOUNT_SIZE_OVERRIDE")))
    print(f"Account size overridden by midday pipeline: ${ACCOUNT_SIZE:,}")
elif os.path.exists(_account_state_path):
    with open(_account_state_path) as _f:
        ACCOUNT_SIZE = int(float(json.load(_f).get("account_value") or _ACCOUNT_SIZE_DEFAULT))
    print(f"Account size loaded from Alpaca: ${ACCOUNT_SIZE:,}")
else:
    ACCOUNT_SIZE = _ACCOUNT_SIZE_DEFAULT
    print(f"Account size loaded from config (no account_state.json found): ${ACCOUNT_SIZE:,}")

ACCOUNT_UNDER_25K = True       # enables PDT rule
PDT_TRADES_USED_THIS_WEEK = 0  # update this each week
PDT_LIMIT = 3                  # max day trades per 5 days
TRADING_HOURS_ONLY = True      # only trade 9:30am-4pm ET
EARNINGS_BLACKOUT_DAYS = 2     # days before earnings to avoid
MAX_SECTOR_CONCENTRATION = 0.40  # max 40% in one sector
MAX_SINGLE_STOCK = 0.20        # max 20% in one stock
ALLOWED_MARKETS = ["US"]       # US only for now
# ============================================================

BASE_DIR    = os.path.dirname(__file__)
RISK_PATH   = os.path.join(BASE_DIR, "risk_report.json")
REGIME_PATH = os.path.join(BASE_DIR, "market_regime.json")
REPORT_PATH = os.path.join(BASE_DIR, "compliance_report.json")

SYSTEM_PROMPT = (
    "You are a trading compliance officer familiar with SEC regulations, FINRA rules, "
    "and broker requirements for retail traders. Review these trades for a US retail trader "
    "with under $25,000 in their account. Check for:\n"
    "1. Pattern Day Trader rule violations or risks\n"
    "2. Any concentration risks that could violate sound portfolio management\n"
    "3. Earnings announcement risks for each ticker\n"
    "4. Any other compliance or legal considerations a beginner trader should know about\n"
    "Give a PASS, CAUTION, or BLOCK verdict for each trade with a one-line plain English explanation."
)

# US market holidays (NYSE) — update each year
US_MARKET_HOLIDAYS = {
    # 2025
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 20),
    datetime.date(2025, 2, 17),
    datetime.date(2025, 4, 18),
    datetime.date(2025, 5, 26),
    datetime.date(2025, 6, 19),
    datetime.date(2025, 7, 4),
    datetime.date(2025, 9, 1),
    datetime.date(2025, 11, 27),
    datetime.date(2025, 12, 25),
    # 2026
    datetime.date(2026, 1, 1),
    datetime.date(2026, 1, 19),
    datetime.date(2026, 2, 16),
    datetime.date(2026, 4, 3),
    datetime.date(2026, 5, 25),
    datetime.date(2026, 6, 19),
    datetime.date(2026, 7, 3),
    datetime.date(2026, 9, 7),
    datetime.date(2026, 11, 26),
    datetime.date(2026, 12, 25),
}


def call_claude_with_retry(client, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except Exception as e:
            if "529" in str(e) or "overloaded" in str(e).lower() or "rate_limit" in str(e).lower():
                wait_time = (2 ** attempt) * 10
                print(f"  API overloaded, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded - API still overloaded")


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def is_possible_day_trade(time_horizon: str) -> bool:
    if not time_horizon:
        return False
    h = time_horizon.lower()
    return bool(re.search(
        r"\b1[-\s]?2\s*day|\bintraday\b|\bsame[-\s]?day\b|\b1\s*(?:trading\s*)?day\b",
        h
    ))


# ── Market hours ──────────────────────────────────────────────────────────────

def check_market_hours() -> dict:
    et      = pytz.timezone("America/New_York")
    now_et  = datetime.datetime.now(et)
    today   = now_et.date()

    is_weekend = today.weekday() >= 5
    is_holiday = today in US_MARKET_HOLIDAYS

    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    is_market_hours = market_open <= now_et <= market_close

    if is_weekend or is_holiday:
        status = "CLOSED"
    elif is_market_hours:
        status = "OPEN"
    else:
        status = "CLOSED"

    warnings = []
    if is_weekend:
        warnings.append(f"WARNING: Today is {today.strftime('%A')} — markets are closed.")
    elif is_holiday:
        warnings.append("WARNING: Today is a US market holiday — markets are closed.")
    elif not is_market_hours:
        warnings.append(
            "REMINDER: Markets are currently closed. "
            "Place orders only between 9:30am–4:00pm ET on weekdays."
        )
    else:
        warnings.append(
            "REMINDER: Markets are open. Place trades between 9:30am–4:00pm ET only."
        )

    return {
        "status":           status,
        "current_time_et":  now_et.strftime("%Y-%m-%d %H:%M ET"),
        "is_weekend":       is_weekend,
        "is_holiday":       is_holiday,
        "is_market_hours":  is_market_hours,
        "warnings":         warnings,
    }


# ── PDT check ─────────────────────────────────────────────────────────────────

def check_pdt(trade: dict) -> dict:
    time_horizon        = trade.get("time_horizon", "") or ""
    possible_day_trade  = is_possible_day_trade(time_horizon)

    flags  = []
    status = "CLEAR"

    if possible_day_trade:
        if ACCOUNT_UNDER_25K:
            if PDT_TRADES_USED_THIS_WEEK >= PDT_LIMIT:
                flags.append(
                    f"PDT_BLOCKED: {PDT_TRADES_USED_THIS_WEEK}/{PDT_LIMIT} day trades used "
                    f"this week. Adding another violates the PDT rule."
                )
                status = "PDT_BLOCKED"
            else:
                remaining = PDT_LIMIT - PDT_TRADES_USED_THIS_WEEK
                flags.append(
                    f"PDT WARNING: Horizon '{time_horizon}' may be a day trade. "
                    f"{remaining} day trade(s) remaining this week."
                )
                status = "PDT_WARNING"
        else:
            flags.append(
                f"NOTE: Possible day trade (horizon: '{time_horizon}'). "
                "Account > $25K so PDT rule does not apply."
            )
    else:
        flags.append("PDT CLEAR: Trade horizon does not suggest intraday activity.")

    return {
        "flags":                flags,
        "status":               status,
        "is_possible_day_trade": possible_day_trade,
    }


# ── Earnings blackout ─────────────────────────────────────────────────────────

def get_earnings_info(ticker: str) -> dict:
    today         = datetime.date.today()
    earnings_date = None

    try:
        t = yf.Ticker(ticker)

        # Method 1: .calendar dict
        with suppress_stderr():
            cal = t.calendar

        if isinstance(cal, dict):
            raw = cal.get("Earnings Date")
            if raw is not None:
                items = list(raw) if hasattr(raw, "__iter__") else [raw]
                future = []
                for item in items:
                    try:
                        d = item.date() if hasattr(item, "date") else None
                        if d and d >= today:
                            future.append(d)
                    except Exception:
                        pass
                if future:
                    earnings_date = min(future)

        # Method 2: .earnings_dates DataFrame
        if earnings_date is None:
            try:
                with suppress_stderr():
                    ed_df = t.earnings_dates
                if ed_df is not None and not ed_df.empty:
                    future = [
                        idx.date() for idx in ed_df.index
                        if hasattr(idx, "date") and idx.date() >= today
                    ]
                    if future:
                        earnings_date = min(future)
            except Exception:
                pass

    except Exception as e:
        return {
            "ticker":        ticker,
            "earnings_date": None,
            "days_until":    None,
            "status":        "UNKNOWN",
            "note":          "Verify earnings manually at earnings.com before trading",
            "error":         str(e),
        }

    if earnings_date is None:
        return {
            "ticker":        ticker,
            "earnings_date": None,
            "days_until":    None,
            "status":        "UNKNOWN",
            "note":          "Verify earnings manually at earnings.com before trading",
        }

    days_until = (earnings_date - today).days

    if days_until < 0:
        status = "PASSED"
    elif days_until == 0:
        status = "BLOCKED"   # earnings today
    elif days_until == 1:
        status = "BLOCKED"   # earnings tomorrow
    elif days_until <= EARNINGS_BLACKOUT_DAYS:
        status = "EARNINGS_RISK"
    else:
        status = "CLEAR"

    return {
        "ticker":       ticker,
        "earnings_date": earnings_date.isoformat(),
        "days_until":   days_until,
        "status":       status,
    }


# ── Sector info ───────────────────────────────────────────────────────────────

def get_sector(ticker: str) -> str:
    try:
        with suppress_stderr():
            info = yf.Ticker(ticker).info
        return info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


# ── Sector concentration ──────────────────────────────────────────────────────

def check_sector_concentration(trades: list[dict], sector_map: dict) -> dict:
    sector_dollars: dict[str, float] = {}
    for t in trades:
        sector  = sector_map.get(t["ticker"], "Unknown")
        dollars = float(t.get("position_dollars") or 0)
        sector_dollars[sector] = sector_dollars.get(sector, 0.0) + dollars

    sector_pct = {
        s: round(d / ACCOUNT_SIZE * 100, 1)
        for s, d in sector_dollars.items()
    }

    warnings = []
    for sector, pct in sector_pct.items():
        if pct / 100 > MAX_SECTOR_CONCENTRATION:
            warnings.append(
                f"SECTOR WARNING: {sector} at {pct:.1f}% exceeds "
                f"{MAX_SECTOR_CONCENTRATION * 100:.0f}% max concentration."
            )
        else:
            warnings.append(f"Sector {sector}: {pct:.1f}% — within limits.")

    return {"sector_breakdown": sector_pct, "warnings": warnings}


# ── Position size check ───────────────────────────────────────────────────────

def check_position_size(trade: dict) -> dict:
    raw_pct     = trade.get("position_pct") or 0
    pct_decimal = raw_pct / 100 if raw_pct > 1 else raw_pct

    flags  = []
    status = "CLEAR"

    if pct_decimal > MAX_SINGLE_STOCK:
        flags.append(
            f"SIZE WARNING: {trade['ticker']} is {raw_pct:.1f}% of portfolio, "
            f"exceeding {MAX_SINGLE_STOCK * 100:.0f}% single-stock limit."
        )
        status = "WARNING"
    else:
        flags.append(
            f"SIZE CLEAR: {trade['ticker']} at {raw_pct:.1f}% is within "
            f"{MAX_SINGLE_STOCK * 100:.0f}% single-stock limit."
        )

    # Cross-check with risk_agent position_dollars
    pos_dollars = trade.get("position_dollars") or 0
    derived_pct = pos_dollars / ACCOUNT_SIZE * 100 if ACCOUNT_SIZE else 0
    if abs(derived_pct - raw_pct) > 1.0:
        flags.append(
            f"NOTE: risk_agent shows ${pos_dollars} ({derived_pct:.1f}%) "
            f"vs stored position_pct {raw_pct:.1f}% — minor rounding difference."
        )

    return {"flags": flags, "status": status}


# ── Claude prompt ─────────────────────────────────────────────────────────────

def build_claude_prompt(trades: list[dict], market_hours: dict, regime: str) -> str:
    lines = [
        f"ACCOUNT SIZE: ${ACCOUNT_SIZE:,}  |  ACCOUNT UNDER $25K: {ACCOUNT_UNDER_25K}",
        f"MARKET REGIME: {regime}",
        f"PDT TRADES USED THIS WEEK: {PDT_TRADES_USED_THIS_WEEK}/{PDT_LIMIT}",
        f"MARKET STATUS: {market_hours['status']}  ({market_hours['current_time_et']})",
        "",
        "TRADES FOR COMPLIANCE REVIEW:",
        "",
    ]

    for t in trades:
        comp = t.get("compliance", {})
        pdt  = comp.get("pdt", {})
        earn = comp.get("earnings", {})
        size = comp.get("position_size", {})

        lines += [
            f"── {t['ticker']} ({t.get('direction', 'LONG')}) ──",
            f"   Entry: ${t.get('entry','N/A')}  Stop: ${t.get('stop','N/A')}  "
            f"Target: ${t.get('target','N/A')}",
            f"   Shares: {t.get('shares','N/A')}  "
            f"Position: ${t.get('position_dollars','N/A')} ({t.get('position_pct','N/A')}% of account)",
            f"   Time Horizon: {t.get('time_horizon','N/A')}",
            f"   Sector: {t.get('sector','Unknown')}",
            f"   PDT Status: {pdt.get('status','N/A')}  —  {'; '.join(pdt.get('flags',[]))}",
            f"   Earnings: {earn.get('earnings_date','N/A')} "
            f"({earn.get('days_until','N/A')} days away) — {earn.get('status','N/A')}"
            + (f" — {earn['note']}" if earn.get('note') else ""),
            f"   Position Size: {size.get('status','N/A')} — {'; '.join(size.get('flags',[]))}",
            "",
        ]

    lines.append(
        "For EACH trade above give exactly:\n"
        "### [TICKER]\n"
        "**Verdict:** PASS / CAUTION / BLOCK\n"
        "**Reason:** one-line plain English explanation\n"
    )
    return "\n".join(lines)


def parse_claude_verdicts(response: str, tickers: list[str]) -> dict:
    verdicts = {}
    for ticker in tickers:
        pat   = rf"###\s*{re.escape(ticker)}\b(.*?)(?=###|\Z)"
        match = re.search(pat, response, re.DOTALL)
        block = match.group(1).strip() if match else ""

        def extract(label: str, blk: str = block) -> str:
            m = re.search(rf"\*\*{re.escape(label)}.*?\*\*[:\s]*(.*)", blk)
            return m.group(1).strip() if m else "N/A"

        verdicts[ticker] = {
            "verdict": extract("Verdict"),
            "reason":  extract("Reason"),
        }
    return verdicts


# ── Print report ──────────────────────────────────────────────────────────────

def print_report(
    trades:       list[dict],
    market_hours: dict,
    sector_check: dict,
    verdicts:     dict,
    cleared:      list[dict],
) -> None:
    W = 80
    cleared_tickers = {t["ticker"] for t in cleared}

    print("\n" + "=" * W)
    print("                    COMPLIANCE REPORT")
    print("=" * W)
    print(f"  Account Size    : ${ACCOUNT_SIZE:,}  |  Under $25K: {ACCOUNT_UNDER_25K}")
    print(f"  PDT Trades Used : {PDT_TRADES_USED_THIS_WEEK}/{PDT_LIMIT} this week")
    print(f"  Market Status   : {market_hours['status']}  ({market_hours['current_time_et']})")
    for w in market_hours["warnings"]:
        print(f"  {w}")
    print()

    # ── Per-trade details ──────────────────────────────────────────────────────
    for t in trades:
        ticker = t["ticker"]
        comp   = t.get("compliance", {})
        pdt    = comp.get("pdt", {})
        earn   = comp.get("earnings", {})
        size   = comp.get("position_size", {})
        v      = verdicts.get(ticker, {})

        cleared_tag = "CLEARED" if ticker in cleared_tickers else "BLOCKED"
        print(f"{'─' * W}")
        print(f"  {ticker:<6}  {t.get('direction','LONG'):<5}  |  "
              f"Claude: {v.get('verdict','N/A'):<8}  |  Final: {cleared_tag}")
        print(f"  Sector: {t.get('sector','Unknown'):<28}  "
              f"Position: ${t.get('position_dollars','N/A')} ({t.get('position_pct','N/A')}%)")

        # PDT
        print(f"  PDT:      {pdt.get('status','N/A')}")
        for flag in pdt.get("flags", []):
            print(f"    → {flag}")

        # Earnings
        earn_date   = earn.get("earnings_date") or "N/A"
        earn_days   = earn.get("days_until")
        earn_status = earn.get("status", "UNKNOWN")
        earn_note   = earn.get("note", "")
        days_str    = f"{earn_days} days away" if earn_days is not None else "unknown timing"
        print(f"  Earnings: {earn_date}  ({days_str})  —  {earn_status}")
        if earn_note:
            print(f"    → {earn_note}")

        # Position size
        for flag in size.get("flags", []):
            print(f"  {flag}")

        # Claude verdict
        if v.get("verdict", "N/A") != "N/A":
            print(f"  Claude Verdict : {v['verdict']}")
            print(f"  Reason         : {v.get('reason','N/A')}")
        print()

    # ── PDT summary ────────────────────────────────────────────────────────────
    pdt_warned = [
        t["ticker"] for t in trades
        if t["compliance"]["pdt"]["status"] in ("PDT_WARNING", "PDT_BLOCKED")
    ]
    if pdt_warned:
        print(f"{'─' * W}")
        print(f"  PDT WARNINGS: {', '.join(pdt_warned)}")
        print(f"  Trades used this week: {PDT_TRADES_USED_THIS_WEEK}/{PDT_LIMIT}")
        print(f"  Update PDT_TRADES_USED_THIS_WEEK at the top of this file each week.")
        print()

    # ── Earnings summary ────────────────────────────────────────────────────────
    earn_issues = [
        t for t in trades
        if t["compliance"]["earnings"]["status"] not in ("CLEAR", "UNKNOWN", "PASSED")
    ]
    if earn_issues:
        print(f"{'─' * W}")
        print("  EARNINGS DATES FOUND:")
        for t in earn_issues:
            earn = t["compliance"]["earnings"]
            print(f"    {t['ticker']}: {earn.get('earnings_date','N/A')}  "
                  f"({earn.get('days_until','N/A')} days)  —  {earn.get('status','N/A')}")
        print()

    # ── Sector concentration ────────────────────────────────────────────────────
    print(f"{'─' * W}")
    print("  SECTOR CONCENTRATION:")
    for msg in sector_check.get("warnings", []):
        print(f"    {msg}")
    print()

    # ── Final cleared trades ────────────────────────────────────────────────────
    print(f"{'=' * W}")
    print("  FINAL CLEARED TRADES  (passed risk agent + compliance agent)")
    print(f"{'─' * W}")

    if not cleared:
        print("  No trades cleared compliance. Do not trade today.")
    else:
        col = "{:<6}  {:<5}  {:>8}  {:>8}  {:>8}  {:>8}  {:<10}  {:<10}"
        print(col.format("Ticker", "Dir", "Entry", "Stop", "Target", "Shares",
                         "Compliance", "Sector"))
        print("─" * W)
        for t in cleared:
            v = verdicts.get(t["ticker"], {})
            print(col.format(
                t.get("ticker", "N/A"),
                t.get("direction", "N/A")[:5],
                str(t.get("entry", "N/A")),
                str(t.get("stop", "N/A")),
                str(t.get("target", "N/A")),
                str(t.get("shares", "N/A"))[:8],
                v.get("verdict", "N/A")[:10],
                t.get("sector", "Unknown")[:10],
            ))

    print()
    print(f"{'─' * W}")
    print("  NEXT STEPS:")
    print("  These trades are cleared. In paper trading mode — log these manually")
    print("  in Alpaca paper trading dashboard.")
    print("  In live mode — the execution agent will place these.")
    print(f"{'=' * W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("       TRADING AGENT — COMPLIANCE CHECK")
    print("=" * 62)

    risk_report = load_json(RISK_PATH, {})
    regime_data = load_json(REGIME_PATH, {})

    if not risk_report:
        print("No risk_report.json found. Run risk_agent.py first.")
        sys.exit(1)

    all_trades = risk_report.get("trades", [])
    regime     = risk_report.get("regime", regime_data.get("regime", "CAUTIOUS"))
    approved   = [t for t in all_trades if t.get("status") == "APPROVED"]

    if not approved:
        print("No APPROVED trades in risk_report.json. Nothing to check.")
        sys.exit(0)

    print(f"\nRegime : {regime}")
    print(f"Approved trades to review: {[t['ticker'] for t in approved]}\n")

    # Market hours (once for all trades)
    print("Checking market hours...")
    market_hours = check_market_hours()
    for w in market_hours["warnings"]:
        print(f"  {w}")

    # Per-ticker checks
    enriched:   list[dict] = []
    sector_map: dict[str, str] = {}

    for trade in approved:
        ticker = trade["ticker"]
        print(f"\nChecking {ticker}...")

        print(f"  Fetching sector info...")
        sector = get_sector(ticker)
        sector_map[ticker] = sector
        print(f"  Sector: {sector}")

        print(f"  Fetching earnings date...")
        earn_check = get_earnings_info(ticker)
        ed = earn_check.get("earnings_date", "N/A") or "N/A"
        print(f"  Earnings: {ed} ({earn_check.get('days_until','?')} days) — {earn_check['status']}")

        pdt_check  = check_pdt(trade)
        size_check = check_position_size(trade)

        enriched_trade = dict(trade)
        enriched_trade["sector"] = sector
        enriched_trade["compliance"] = {
            "pdt":           pdt_check,
            "earnings":      earn_check,
            "position_size": size_check,
        }
        enriched.append(enriched_trade)

    # Sector concentration (across all approved trades)
    print("\nChecking sector concentration...")
    sector_check = check_sector_concentration(approved, sector_map)
    for w in sector_check["warnings"]:
        print(f"  {w}")

    # Claude compliance review
    print("\nSending trades to Claude for compliance review...")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt  = build_claude_prompt(enriched, market_hours, regime)
    msg     = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    verdicts = parse_claude_verdicts(
        msg.content[0].text,
        [t["ticker"] for t in enriched],
    )

    # Determine cleared trades
    # BLOCKED if: Claude says BLOCK, or earnings within 1 day, or PDT limit exceeded
    cleared: list[dict] = []
    for t in enriched:
        ticker      = t["ticker"]
        v           = verdicts.get(ticker, {})
        verdict     = (v.get("verdict") or "N/A").upper()
        earn_status = t["compliance"]["earnings"]["status"]
        pdt_status  = t["compliance"]["pdt"]["status"]

        is_blocked = (
            verdict == "BLOCK"
            or earn_status == "BLOCKED"
            or pdt_status == "PDT_BLOCKED"
        )
        if not is_blocked:
            cleared.append(t)

    # Print full report
    print_report(enriched, market_hours, sector_check, verdicts, cleared)

    # Save compliance_report.json
    cleared_tickers = {t["ticker"] for t in cleared}

    report = {
        "generated_at":        datetime.datetime.now().isoformat(),
        "regime":              regime,
        "account_size":        ACCOUNT_SIZE,
        "account_under_25k":   ACCOUNT_UNDER_25K,
        "pdt_trades_used":     PDT_TRADES_USED_THIS_WEEK,
        "pdt_limit":           PDT_LIMIT,
        "market_hours":        market_hours,
        "sector_concentration": sector_check,
        "trades_reviewed":     len(enriched),
        "trades_cleared":      len(cleared),
        "trades": [
            {
                "ticker":                t["ticker"],
                "direction":             t["direction"],
                "entry":                 t["entry"],
                "stop":                  t["stop"],
                "target":                t["target"],
                "shares":                t.get("shares"),
                "position_dollars":      t.get("position_dollars"),
                "position_pct":          t.get("position_pct"),
                "sector":                t.get("sector", "Unknown"),
                "compliance_pdt":        t["compliance"]["pdt"]["status"],
                "compliance_earnings":   t["compliance"]["earnings"]["status"],
                "earnings_date":         t["compliance"]["earnings"].get("earnings_date"),
                "earnings_days_until":   t["compliance"]["earnings"].get("days_until"),
                "compliance_size":       t["compliance"]["position_size"]["status"],
                "claude_verdict":        verdicts.get(t["ticker"], {}).get("verdict", "N/A"),
                "claude_reason":         verdicts.get(t["ticker"], {}).get("reason", "N/A"),
                "cleared":               t["ticker"] in cleared_tickers,
            }
            for t in enriched
        ],
        "cleared_trades": [
            {
                "ticker":           t["ticker"],
                "direction":        t["direction"],
                "entry":            t["entry"],
                "stop":             t["stop"],
                "target":           t["target"],
                "shares":           t.get("shares"),
                "position_dollars": t.get("position_dollars"),
                "sector":           t.get("sector", "Unknown"),
                "claude_verdict":   verdicts.get(t["ticker"], {}).get("verdict", "N/A"),
            }
            for t in cleared
        ],
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Compliance report saved to compliance_report.json")
    print(f"Result: {len(cleared)}/{len(enriched)} trades cleared compliance.\n")


if __name__ == "__main__":
    main()
