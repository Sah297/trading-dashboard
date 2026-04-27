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

# ============================================================
# EDIT YOUR ACCOUNT SETTINGS HERE
# ============================================================
_ACCOUNT_SIZE_DEFAULT = 100000
_account_state_path   = os.path.join(os.path.dirname(__file__), "account_state.json")
_settings_path        = os.path.join(os.path.dirname(__file__), "settings.json")

MAX_RISK_PER_TRADE  = 0.02    # 2% max risk per trade
MAX_DAILY_LOSS      = 0.03    # 3% max daily loss limit
MAX_OPEN_POSITIONS  = 8       # max simultaneous open trades
MAX_POSITION_SIZE   = 0.20    # max 20% of account in one stock
PDT_TRADES_USED     = 0       # day trades used this week (0-3)
PDT_LIMIT           = 3       # max day trades per 5 days (PDT rule)
ACCOUNT_UNDER_25K   = True    # set False if account > $25,000
MONTHLY_TARGET_PCT  = 0.10    # 10% = conservative | 20% = moderate | 30% = aggressive
# ============================================================

# ── Capital management: read account_state + settings ──────────
if os.getenv("ACCOUNT_SIZE_OVERRIDE"):
    # Midday pipeline: bypass budget logic entirely
    ACCOUNT_SIZE          = int(float(os.getenv("ACCOUNT_SIZE_OVERRIDE")))
    MANUAL_TRADING_BUDGET = ACCOUNT_SIZE
    _cash                 = ACCOUNT_SIZE
    _committed            = 0
    _n_pos                = 0
    print(f"Capital (midday override): ${ACCOUNT_SIZE:,}")
else:
    _cash       = float(_ACCOUNT_SIZE_DEFAULT)
    _committed  = 0.0
    _n_pos      = 0
    _manual_bgt = 0.0
    _remaining  = 0.0

    if os.path.exists(_account_state_path):
        with open(_account_state_path) as _f:
            _acct = json.load(_f)
        _cash      = float(_acct.get("cash_available") or _acct.get("account_value") or _ACCOUNT_SIZE_DEFAULT)
        _committed = float(_acct.get("committed_capital") or 0)
        _n_pos     = int(_acct.get("open_positions_count") or 0)
        _manual_bgt = float(_acct.get("manual_budget") or 0)
        _remaining  = float(_acct.get("available_to_invest") or _acct.get("remaining_budget") or 0)

    # settings.json can override risk/position config
    if os.path.exists(_settings_path):
        with open(_settings_path) as _sf:
            _app = json.load(_sf)
        _sett_budget = float(_app.get("trading_budget") or 0)
        if _sett_budget:
            _manual_bgt = _sett_budget
            _remaining  = max(0.0, _manual_bgt - _committed)
        if _app.get("max_risk_per_trade"):
            MAX_RISK_PER_TRADE = float(_app["max_risk_per_trade"])
        if _app.get("max_open_positions"):
            MAX_OPEN_POSITIONS = int(_app["max_open_positions"])

    if not _manual_bgt:
        _manual_bgt = _cash
        _remaining  = max(0.0, _cash - _committed)

    MANUAL_TRADING_BUDGET = int(_manual_bgt)
    ACCOUNT_SIZE          = int(_remaining)

    _W = 46
    _pos_str = f"({_n_pos} position{'s' if _n_pos != 1 else ''})" if _n_pos else ""
    print("=" * _W)
    print("  CAPITAL MANAGEMENT SUMMARY")
    print("=" * _W)
    print(f"  Real Alpaca Cash    : ${_cash:>12,.0f}")
    print(f"  Manual Budget Set   : ${_manual_bgt:>12,.0f}")
    print(f"  In Open Trades      : ${_committed:>12,.0f}  {_pos_str}")
    print(f"  {'─' * (_W - 2)}")
    print(f"  Available to Invest : ${_remaining:>12,.0f}  ← agent uses this")
    print("=" * _W)
# ─────────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(__file__)
SHORTLIST_PATH  = os.path.join(BASE_DIR, "shortlist.json")
REGIME_PATH     = os.path.join(BASE_DIR, "market_regime.json")
HOT_PATH        = os.path.join(BASE_DIR, "hot_sectors.json")
TECHNICAL_PATH  = os.path.join(BASE_DIR, "technical_analysis.json")
REPORT_PATH     = os.path.join(BASE_DIR, "risk_report.json")
APPROVALS_PATH  = os.path.join(BASE_DIR, "trade_approvals.json")

SYSTEM_PROMPT = (
    "You are a risk manager trained on the principles from "
    "Trading in the Zone by Mark Douglas, The New Trading for a Living by Dr Alexander Elder, "
    "and Reminiscences of a Stock Operator. Review these trade setups and their risk calculations. "
    "For each trade provide:\n"
    "1. A one-line risk verdict (TAKE IT / REDUCE SIZE / SKIP IT)\n"
    "2. The single biggest risk factor for this specific trade\n"
    "3. What would make you more confident in this trade\n"
    "Always think in terms of protecting capital first, profits second. "
    "In a CAUTIOUS regime be especially strict."
)

REGIME_RISK = {
    "BULLISH":   MAX_RISK_PER_TRADE,
    "CAUTIOUS":  0.01,
    "DEFENSIVE": 0.005,
    "PAUSE":     0.0,
}

CYCLES_PER_MONTH = 4   # roughly weekly trading cycles


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


def parse_price(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def get_rsi_map(technical: list[dict]) -> dict[str, float]:
    result = {}
    for rec in technical:
        ticker  = rec.get("ticker", "")
        rsi_raw = rec.get("rsi_assessment", "")
        m = re.search(r"[\d.]+", str(rsi_raw))
        if m:
            result[ticker] = float(m.group(0))
    return result


def get_cold_tickers(hot_data: dict) -> set[str]:
    cold = set()
    for s in hot_data.get("cold_sectors", []):
        cold.add(s.get("etf", "").upper())
    return cold


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    sector_map = {}
    for ticker in tickers:
        try:
            with suppress_stderr():
                info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector") or "Unknown"
        except Exception:
            sector_map[ticker] = "Unknown"
    return sector_map


def apply_sector_concentration(trades_with_risk: list[dict], sector_map: dict) -> None:
    for trade in trades_with_risk:
        trade["sector"] = sector_map.get(trade["ticker"], "Unknown")

    sector_groups: dict[str, list[dict]] = {}
    for trade in trades_with_risk:
        if trade["status"] == "APPROVED":
            sector_groups.setdefault(trade["sector"], []).append(trade)

    for sector, group in sector_groups.items():
        if len(group) > 2:
            group.sort(key=lambda t: t["position"].get("rr_ratio", 0), reverse=True)
            for trade in group[2:]:
                trade["status"] = "WARNING"
                trade["flags"].append(
                    f"WARNING: Sector concentration limit — max 2 trades per sector "
                    f"({sector}). Kept top 2 by R/R ratio."
                )


def is_day_trade(time_horizon: str) -> bool:
    if not time_horizon:
        return False
    return bool(re.search(r"\b1\s*trading\s*day\b|\bsame\s*day\b|\bintraday\b",
                          time_horizon.lower()))


def calc_position(entry: float, stop: float, target: float,
                  direction: str, risk_dollars: float) -> dict:
    if direction.upper() == "SHORT":
        risk_per_share   = stop - entry
        reward_per_share = entry - target
    else:
        risk_per_share   = entry - stop
        reward_per_share = target - entry

    if risk_per_share <= 0:
        return {}

    shares      = risk_dollars / risk_per_share
    pos_dollars = shares * entry
    pos_pct     = pos_dollars / ACCOUNT_SIZE
    rr_ratio    = reward_per_share / risk_per_share

    capped = False
    if pos_pct > MAX_POSITION_SIZE:
        shares      = (ACCOUNT_SIZE * MAX_POSITION_SIZE) / entry
        pos_dollars = shares * entry
        pos_pct     = MAX_POSITION_SIZE
        capped      = True

    return {
        "shares":      round(shares, 2),
        "pos_dollars": round(pos_dollars, 2),
        "pos_pct":     round(pos_pct * 100, 2),
        "rr_ratio":    round(rr_ratio, 2),
        "est_profit":  round(shares * reward_per_share, 2),
        "est_loss":    round(shares * risk_per_share, 2),
        "capped":      capped,
    }


def evaluate_trade(trade: dict, pos: dict, regime: str,
                   rsi_map: dict, cold_tickers: set,
                   trade_index: int) -> tuple[str, list[str]]:
    ticker    = trade.get("ticker", "")
    direction = trade.get("direction", "LONG").upper()
    flags: list[str] = []
    status = "APPROVED"

    if regime in ("DEFENSIVE", "PAUSE"):
        flags.append(f"REJECTED: regime is {regime} — no new trades")
        return "REJECTED", flags

    if not pos:
        flags.append("REJECTED: could not calculate position (invalid prices)")
        return "REJECTED", flags

    if pos["rr_ratio"] < 1.5:
        flags.append(f"REJECTED: R/R {pos['rr_ratio']} is below minimum 1.5")
        status = "REJECTED"

    if regime == "CAUTIOUS" and ticker in cold_tickers:
        flags.append(f"WARNING: {ticker} is in a cold sector under CAUTIOUS regime")
        if status == "APPROVED":
            status = "WARNING"

    rsi = rsi_map.get(ticker)
    if rsi is not None:
        if direction == "LONG" and rsi > 80:
            flags.append(f"WARNING: RSI {rsi} overbought (>80) on a LONG")
            if status == "APPROVED":
                status = "WARNING"
        elif direction == "SHORT" and rsi < 20:
            flags.append(f"WARNING: RSI {rsi} oversold (<20) on a SHORT")
            if status == "APPROVED":
                status = "WARNING"

    if ACCOUNT_UNDER_25K and PDT_TRADES_USED >= PDT_LIMIT:
        if is_day_trade(trade.get("time_horizon", "")):
            flags.append(f"WARNING: PDT limit reached ({PDT_TRADES_USED}/{PDT_LIMIT})")
            if status == "APPROVED":
                status = "WARNING"

    if trade_index + 1 > MAX_OPEN_POSITIONS:
        flags.append(f"WARNING: would exceed MAX_OPEN_POSITIONS ({MAX_OPEN_POSITIONS})")
        if status == "APPROVED":
            status = "WARNING"

    if not flags:
        flags.append("No issues found")

    return status, flags


# ── Growth projections ────────────────────────────────────────────────────────

def calc_growth_projections(approved: list[dict]) -> dict:
    total_reward = sum(t["position"].get("est_profit", 0) for t in approved)
    total_risk   = sum(t["position"].get("est_loss",   0) for t in approved)

    def cycle_profit(win_rate: float) -> float:
        return win_rate * total_reward - (1 - win_rate) * total_risk

    optimistic_cycle   = total_reward                    # 100% win rate
    realistic_cycle    = cycle_profit(0.55)
    pessimistic_cycle  = cycle_profit(0.40)

    optimistic_month   = optimistic_cycle  * CYCLES_PER_MONTH
    realistic_month    = realistic_cycle   * CYCLES_PER_MONTH
    pessimistic_month  = pessimistic_cycle * CYCLES_PER_MONTH

    def compound(monthly_gain: float, months: int) -> float:
        rate = monthly_gain / ACCOUNT_SIZE
        return round(ACCOUNT_SIZE * ((1 + rate) ** months), 2)

    def required_rr(target_monthly: float) -> float:
        n = len(approved)
        if n == 0 or total_risk == 0:
            return 0.0
        avg_risk = total_risk / n
        # 4 cycles * n trades per cycle * (0.55*RR - 0.45) * avg_risk = target
        needed = (target_monthly / (CYCLES_PER_MONTH * n * avg_risk) + 0.45) / 0.55
        return round(needed, 2)

    scenarios = {}
    for label, pct in [("conservative", MONTHLY_TARGET_PCT),
                        ("moderate",     min(MONTHLY_TARGET_PCT * 2, 0.40)),
                        ("aggressive",   min(MONTHLY_TARGET_PCT * 3, 0.60))]:
        target_monthly = ACCOUNT_SIZE * pct
        scenarios[label] = {
            "monthly_target":    round(target_monthly, 2),
            "target_pct":        pct,
            "required_rr":       required_rr(target_monthly),
            "cycles_needed":     round(target_monthly / realistic_cycle, 1) if realistic_cycle > 0 else "N/A",
            "m1":  compound(target_monthly, 1),
            "m3":  compound(target_monthly, 3),
            "m6":  compound(target_monthly, 6),
            "m12": compound(target_monthly, 12),
        }

    avg_rr = (sum(t["position"].get("rr_ratio", 0) for t in approved) / len(approved)
              if approved else 0)

    return {
        "n_approved":         len(approved),
        "total_reward":       round(total_reward, 2),
        "total_risk":         round(total_risk, 2),
        "avg_rr":             round(avg_rr, 2),
        "optimistic_month":   round(optimistic_month, 2),
        "realistic_month":    round(realistic_month, 2),
        "pessimistic_month":  round(pessimistic_month, 2),
        "scenarios":          scenarios,
    }


def build_projection_prompt(proj: dict, regime: str) -> str:
    s = proj["scenarios"]
    lines = [
        f"ACCOUNT SIZE: ${ACCOUNT_SIZE:,}  |  REGIME: {regime}",
        f"APPROVED TRADES: {proj['n_approved']}  |  AVG R/R: {proj['avg_rr']}",
        f"Total reward if all win: ${proj['total_reward']:,.2f}",
        f"Total risk if all lose:  ${proj['total_risk']:,.2f}",
        "",
        "MONTHLY OUTCOME ESTIMATES (4 cycles/month):",
        f"  Optimistic  (100% win rate): +${proj['optimistic_month']:,.2f}",
        f"  Realistic   (55% win rate) : +${proj['realistic_month']:,.2f}",
        f"  Pessimistic (40% win rate) : ${proj['pessimistic_month']:,.2f}",
        "",
        "GROWTH SCENARIOS:",
    ]
    for name, sc in s.items():
        lines += [
            f"  {name.upper()} (+{sc['target_pct']*100:.0f}%/month = +${sc['monthly_target']:,.0f}):",
            f"    Required R/R: {sc['required_rr']}  Cycles needed/month: {sc['cycles_needed']}",
            f"    Month 1:  ${ACCOUNT_SIZE:,} → ${sc['m1']:,.0f}  (+${sc['m1']-ACCOUNT_SIZE:,.0f})",
            f"    Month 3:  ${ACCOUNT_SIZE:,} → ${sc['m3']:,.0f}  (+${sc['m3']-ACCOUNT_SIZE:,.0f})",
            f"    Month 6:  ${ACCOUNT_SIZE:,} → ${sc['m6']:,.0f}  (+${sc['m6']-ACCOUNT_SIZE:,.0f})",
            f"    Month 12: ${ACCOUNT_SIZE:,} → ${sc['m12']:,.0f}  (+${sc['m12']-ACCOUNT_SIZE:,.0f})",
            "",
        ]
    lines.append(
        "You are a trading coach and risk manager. Review these growth projections for a "
        f"beginner trader with a ${ACCOUNT_SIZE:,} account in a {regime} market regime. Be honest about:\n"
        "1. Whether these targets are realistic given the current regime\n"
        "2. The biggest threat to achieving each scenario\n"
        "3. What the trader must do consistently to hit the moderate target\n"
        "4. A realistic expectation for month 1 specifically\n"
        "Be encouraging but brutally honest — protect this trader from overconfidence."
    )
    return "\n".join(lines)


def print_growth_projections(proj: dict, coach_text: str) -> None:
    W = 80
    s = proj["scenarios"]
    print("\n" + "=" * W)
    print("                    GROWTH PROJECTIONS")
    print("=" * W)
    print(f"  Approved trades : {proj['n_approved']}  |  Avg R/R : {proj['avg_rr']}")
    print(f"  Total upside    : ${proj['total_reward']:>8,.2f}  (all trades win)")
    print(f"  Total downside  : ${proj['total_risk']:>8,.2f}  (all trades lose)")
    print()
    print("  Single-month outcome (4 cycles):")
    print(f"    Optimistic  (100% wins)  : +${proj['optimistic_month']:,.2f}")
    print(f"    Realistic   ( 55% wins)  : +${proj['realistic_month']:,.2f}")
    print(f"    Pessimistic ( 40% wins)  :  ${proj['pessimistic_month']:,.2f}")
    print()

    labels = {"conservative": "CONSERVATIVE (+10%/mo)", "moderate": "MODERATE (+20%/mo)",
              "aggressive":   "AGGRESSIVE (+30%/mo)"}
    for key, title in labels.items():
        sc = s[key]
        print(f"  {title}  —  target ${sc['monthly_target']:,.0f}/month")
        print(f"    Required R/R: {sc['required_rr']}   Cycles needed: {sc['cycles_needed']}/month")
        print(f"    {'Month':>6}  {'Account Value':>14}  {'Gain':>10}")
        print(f"    {'─'*6}  {'─'*14}  {'─'*10}")
        for m, key2 in ((1, "m1"), (3, "m3"), (6, "m6"), (12, "m12")):
            val  = sc[key2]
            gain = val - ACCOUNT_SIZE
            print(f"    {m:>5}   ${val:>13,.0f}  +${gain:>9,.0f}")
        print()

    print("─" * W)
    print("  COACHING REVIEW (Claude):")
    print("─" * W)
    for line in coach_text.strip().splitlines():
        print(f"  {line}")
    print("=" * W)


def build_claude_prompt(trades_with_risk: list[dict], regime: str) -> str:
    lines = [
        f"ACCOUNT SIZE: ${ACCOUNT_SIZE:,}",
        f"MARKET REGIME: {regime}",
        f"DAILY LOSS LIMIT: ${ACCOUNT_SIZE * MAX_DAILY_LOSS:,.0f}",
        "",
        "TRADE SETUPS WITH RISK CALCULATIONS:",
        "",
    ]
    for t in trades_with_risk:
        p = t["position"]
        lines.append(f"── {t['ticker']} ({t['direction']}) ──")
        lines.append(f"   Entry: ${t['entry']}  Stop: ${t['stop']}  Target: ${t['target']}")
        lines.append(f"   Shares: {p.get('shares','N/A')}  Position: ${p.get('pos_dollars','N/A')} ({p.get('pos_pct','N/A')}% of account)")
        lines.append(f"   Risk: ${p.get('est_loss','N/A')}  Reward: ${p.get('est_profit','N/A')}  R/R: {p.get('rr_ratio','N/A')}")
        lines.append(f"   Setup: {t.get('setup_type','N/A')}  Horizon: {t.get('time_horizon','N/A')}")
        lines.append(f"   Status: {t['status']}  Flags: {'; '.join(t['flags'])}")
        lines.append(f"   Why Selected: {t.get('why_selected','')[:120]}")
        lines.append("")
    lines.append(
        "For EACH trade above give exactly:\n"
        "### [TICKER]\n"
        "**Verdict:** TAKE IT / REDUCE SIZE / SKIP IT\n"
        "**Biggest Risk:** ...\n"
        "**What Would Increase Confidence:** ...\n"
    )
    return "\n".join(lines)


def parse_claude_verdicts(response: str, tickers: list[str]) -> dict[str, dict]:
    verdicts = {}
    for ticker in tickers:
        pat   = rf"###\s*{re.escape(ticker)}\b(.*?)(?=###|\Z)"
        match = re.search(pat, response, re.DOTALL)
        block = match.group(1).strip() if match else ""

        def extract(label: str, blk: str = block) -> str:
            m = re.search(rf"\*\*{re.escape(label)}.*?\*\*[:\s]*(.*)", blk)
            return m.group(1).strip() if m else "N/A"

        verdicts[ticker] = {
            "verdict":            extract("Verdict"),
            "biggest_risk":       extract("Biggest Risk"),
            "confidence_booster": extract("What Would Increase Confidence"),
        }
    return verdicts


def print_report(trades_with_risk: list[dict], regime: str,
                 risk_per_trade: float, verdicts: dict) -> list[dict]:
    W = 80
    print("\n" + "=" * W)
    print("              RISK MANAGEMENT REPORT")
    print("=" * W)
    print(f"  Account Size       : ${ACCOUNT_SIZE:,}")
    print(f"  Daily Loss Limit   : ${ACCOUNT_SIZE * MAX_DAILY_LOSS:,.0f}")
    print(f"  Max Open Positions : {MAX_OPEN_POSITIONS}")
    print(f"  Max Position Size  : {MAX_POSITION_SIZE * 100:.0f}% (${ACCOUNT_SIZE * MAX_POSITION_SIZE:,.0f})")
    print(f"  Regime             : {regime}")
    print(f"  Risk Per Trade     : {risk_per_trade * 100:.1f}% (${ACCOUNT_SIZE * risk_per_trade:,.0f})")
    if regime != "BULLISH":
        print(f"  ⚠️  Regime adjustment: risk reduced to {risk_per_trade * 100:.1f}%")
    print()

    approved_trades = []

    for t in trades_with_risk:
        ticker  = t["ticker"]
        p       = t["position"]
        status  = t["status"]
        v       = verdicts.get(ticker, {})
        verdict = (v.get("verdict") or "N/A").upper()

        skipped_by_claude = status == "WARNING" and any(
            "Downgraded by Claude" in f for f in t["flags"]
        )

        if status == "APPROVED" and "TAKE" in verdict:
            label = "APPROVED — TAKE IT"
        elif status == "APPROVED" and "REDUCE" in verdict:
            label = "APPROVED — REDUCE SIZE"
        elif skipped_by_claude:
            label = "SKIPPED BY CLAUDE"
        elif status == "WARNING":
            label = "WARNING"
        else:
            label = "REJECTED"

        print(f"{'─' * W}")
        print(f"  {ticker:<6}  {t['direction']:<5}  [{label}]")
        if p:
            print(f"  Shares: {p.get('shares','N/A'):<8} "
                  f"Position: ${p.get('pos_dollars','N/A'):<10} "
                  f"({p.get('pos_pct','N/A')}%)"
                  + (" [CAPPED]" if p.get("capped") else ""))
            print(f"  Risk:   ${p.get('est_loss','N/A'):<10} "
                  f"Reward: ${p.get('est_profit','N/A'):<10} "
                  f"R/R: {p.get('rr_ratio','N/A')}")
        for flag in t["flags"]:
            print(f"  → {flag}")
        if v.get("verdict") and v["verdict"] != "N/A":
            print(f"  Claude Verdict     : {v['verdict']}")
            print(f"  Biggest Risk       : {v['biggest_risk']}")
            print(f"  Confidence Booster : {v['confidence_booster']}")
        print()

        if status == "APPROVED":
            approved_trades.append(t)

    # ── Summary ────────────────────────────────────────────────────────────────
    take_it    = [t for t in approved_trades
                  if "TAKE" in (verdicts.get(t["ticker"], {}).get("verdict") or "").upper()]
    reduce_sz  = [t for t in approved_trades
                  if "REDUCE" in (verdicts.get(t["ticker"], {}).get("verdict") or "").upper()]
    claude_skp = [t for t in trades_with_risk
                  if any("Downgraded by Claude" in f for f in t["flags"])]
    sys_warn   = [t for t in trades_with_risk
                  if t["status"] == "WARNING"
                  and not any("Downgraded by Claude" in f for f in t["flags"])]
    hard_rej   = [t for t in trades_with_risk if t["status"] == "REJECTED"]

    print("=" * W)
    print(f"  APPROVED (Claude: TAKE IT)          : {len(take_it)}")
    print(f"  APPROVED WITH CAUTION (REDUCE SIZE) : {len(reduce_sz)}")
    print(f"  SKIPPED BY CLAUDE (SKIP IT)         : {len(claude_skp)}")
    print(f"  WARNING (system flags)              : {len(sys_warn)}")
    print(f"  REJECTED (hard rule violations)     : {len(hard_rej)}")
    print()

    if claude_skp:
        print("  WHY CLAUDE SKIPPED THESE TRADES:")
        for t in claude_skp:
            v = verdicts.get(t["ticker"], {})
            print(f"    {t['ticker']:<6}: {v.get('biggest_risk', 'N/A')}")
        print()

    col = "{:<6}  {:<5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}"
    hdr = col.format("Ticker", "Dir", "Entry", "Stop", "Target", "Shares", "Risk $")

    if take_it:
        print("  TAKE IT — ready to trade now:")
        print(f"  {hdr}")
        print("  " + "─" * 62)
        for t in take_it:
            p = t["position"]
            print("  " + col.format(
                t["ticker"], t["direction"][:5],
                str(t["entry"]), str(t["stop"]), str(t["target"]),
                str(p.get("shares", "N/A"))[:8],
                f"${p.get('est_loss', 'N/A')}",
            ))
        print()

    if reduce_sz:
        print("  REDUCE SIZE — trade smaller than the calculated position:")
        print(f"  {hdr}")
        print("  " + "─" * 62)
        for t in reduce_sz:
            p = t["position"]
            print("  " + col.format(
                t["ticker"], t["direction"][:5],
                str(t["entry"]), str(t["stop"]), str(t["target"]),
                str(p.get("shares", "N/A"))[:8],
                f"${p.get('est_loss', 'N/A')}",
            ))
        print()

    if not approved_trades:
        print("  No APPROVED trades — do not trade today.")

    print("=" * W)
    return approved_trades


def main():
    print("=" * 62)
    print("        TRADING AGENT — RISK MANAGEMENT")
    print("=" * 62)

    shortlist   = load_json(SHORTLIST_PATH, [])
    regime_data = load_json(REGIME_PATH,    {})
    hot_data    = load_json(HOT_PATH,       {})
    technical   = load_json(TECHNICAL_PATH, [])

    if not shortlist:
        print("No shortlist.json found. Run technical_agent.py first.")
        sys.exit(1)

    # ── Insufficient capital guard ────────────────────────────────────────────
    if ACCOUNT_SIZE < 500 and not os.getenv("ACCOUNT_SIZE_OVERRIDE"):
        print(f"\n{'─' * 62}")
        print("  Insufficient capital for new trades.")
        print(f"  Current budget : ${MANUAL_TRADING_BUDGET:,.0f}")
        print(f"  In open trades : ${_committed:,.0f}")
        print(f"  Available      : ${ACCOUNT_SIZE:,.0f}")
        print("  Options:")
        print("  1. Wait for a trade to hit its target/stop")
        print("  2. Increase your budget in app Settings")
        print("  3. Close an existing position manually in Alpaca")
        print(f"{'─' * 62}\n")
        sys.exit(0)

    regime       = regime_data.get("regime", "CAUTIOUS").upper()
    risk_pct     = REGIME_RISK.get(regime, MAX_RISK_PER_TRADE)
    risk_dollars = ACCOUNT_SIZE * risk_pct
    rsi_map      = get_rsi_map(technical)
    cold_tickers = get_cold_tickers(hot_data)

    print(f"\nRegime: {regime}  |  Risk/trade: {risk_pct * 100:.1f}%  (${risk_dollars:,.0f})")
    if regime == "PAUSE":
        print("Regime is PAUSE — all trades rejected automatically.")

    print(f"\nAnalyzing {len(shortlist)} trades...\n")

    trades_with_risk: list[dict] = []
    for i, trade in enumerate(shortlist):
        ticker    = trade.get("ticker", "UNKNOWN")
        direction = trade.get("direction", "LONG")
        entry     = parse_price(trade.get("entry_price"))
        stop      = parse_price(trade.get("stop_loss"))
        target    = parse_price(trade.get("target_price"))

        if entry is None or stop is None or target is None:
            pos    = {}
            status = "REJECTED"
            flags  = ["REJECTED: missing entry/stop/target prices"]
        else:
            pos           = calc_position(entry, stop, target, direction, risk_dollars)
            status, flags = evaluate_trade(trade, pos, regime, rsi_map, cold_tickers, i)

        trades_with_risk.append({
            "ticker":       ticker,
            "direction":    direction,
            "entry":        entry,
            "stop":         stop,
            "target":       target,
            "position":     pos,
            "status":       status,
            "flags":        flags,
            "setup_type":   trade.get("setup_type", ""),
            "time_horizon": trade.get("time_horizon", ""),
            "why_selected": trade.get("why_selected", ""),
        })

    # ── Sector concentration cap ──────────────────────────────────────────────
    print("Fetching sector data for concentration check...")
    sector_map = get_sector_map([t["ticker"] for t in trades_with_risk])
    apply_sector_concentration(trades_with_risk, sector_map)
    print(f"Sector map: { {t: s for t, s in sector_map.items()} }\n")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Trade review ──────────────────────────────────────────────────────────
    print("Sending trade analysis to Claude for review...")
    verdict_msg = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_claude_prompt(trades_with_risk, regime)}],
    )
    verdicts = parse_claude_verdicts(verdict_msg.content[0].text,
                                     [t["ticker"] for t in trades_with_risk])

    # Downgrade APPROVED trades where Claude said SKIP IT
    for trade in trades_with_risk:
        if trade["status"] == "APPROVED":
            v = verdicts.get(trade["ticker"], {})
            if "SKIP" in (v.get("verdict") or "").upper():
                trade["status"] = "WARNING"
                trade["flags"].append(
                    "Downgraded by Claude risk review — Claude said SKIP IT"
                )

    approved_trades = print_report(trades_with_risk, regime, risk_pct, verdicts)

    # ── Capital allocation (greedy, sorted by R/R descending) ─────────────────
    if approved_trades:
        approved_trades.sort(
            key=lambda t: t["position"].get("rr_ratio", 0), reverse=True
        )
        remaining = ACCOUNT_SIZE
        allocated = []
        W_alloc   = 62

        print(f"\n{'─' * W_alloc}")
        print(f"  Capital allocation  (available: ${ACCOUNT_SIZE:,.0f})")
        print(f"{'─' * W_alloc}")

        for trade in approved_trades:
            p           = trade["position"]
            pos_dollars = p.get("pos_dollars") or 0

            if pos_dollars <= 0:
                continue

            if remaining >= 500 and pos_dollars <= remaining:
                # Full allocation
                remaining -= pos_dollars
                allocated.append(trade)
                print(f"  {trade['ticker']:<6} {trade['direction']:<5}  "
                      f"${pos_dollars:>8,.0f}  (remaining: ${remaining:,.0f})")

            elif remaining >= 500:
                # Scale down position to fit remaining capital
                scale = remaining / pos_dollars
                for key in ("shares", "est_loss", "est_profit"):
                    if p.get(key) is not None:
                        p[key] = round(p[key] * scale, 2)
                p["pos_dollars"] = round(remaining, 2)
                p["pos_pct"]     = round(p.get("pos_pct", 0) * scale, 2)
                trade["flags"].append(
                    f"Position scaled to fit available capital: ${remaining:,.0f}"
                )
                print(f"  {trade['ticker']:<6} {trade['direction']:<5}  "
                      f"${remaining:>8,.0f}  (scaled ← ${pos_dollars:,.0f}, remaining: $0)")
                allocated.append(trade)
                remaining = 0

            else:
                reason = (
                    f"INSUFFICIENT CAPITAL — need ${pos_dollars:,.0f}, "
                    f"only ${remaining:,.0f} left"
                )
                for tw in trades_with_risk:
                    if tw["ticker"] == trade["ticker"] and tw["status"] == "APPROVED":
                        tw["status"] = "WARNING"
                        tw["flags"].append(reason)
                        break
                print(f"  {trade['ticker']:<6} {trade['direction']:<5}  "
                      f"INSUFFICIENT CAPITAL — skipped")

        print(f"{'─' * W_alloc}")
        print(f"  Allocated: {len(allocated)} trades  |  "
              f"Skipped: {len(approved_trades) - len(allocated)}  |  "
              f"Capital used: ${ACCOUNT_SIZE - remaining:,.0f}  |  "
              f"Remaining: ${remaining:,.0f}")
        print(f"{'─' * W_alloc}\n")
        approved_trades = allocated

    # ── Growth projections ────────────────────────────────────────────────────
    if approved_trades:
        proj = calc_growth_projections(approved_trades)

        print("\nSending growth projections to Claude for coaching review...")
        coach_msg = call_claude_with_retry(
            client,
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system="You are a trading coach and risk manager.",
            messages=[{"role": "user",
                       "content": build_projection_prompt(proj, regime)}],
        )
        print_growth_projections(proj, coach_msg.content[0].text)
    else:
        proj = {}
        print("\nNo approved trades — skipping growth projections.")

    # ── Save report ───────────────────────────────────────────────────────────
    report_records = []
    for t in trades_with_risk:
        v = verdicts.get(t["ticker"], {})
        p = t["position"]
        report_records.append({
            "ticker":             t["ticker"],
            "direction":          t["direction"],
            "entry":              t["entry"],
            "stop":               t["stop"],
            "target":             t["target"],
            "shares":             p.get("shares"),
            "position_dollars":   p.get("pos_dollars"),
            "position_pct":       p.get("pos_pct"),
            "estimated_risk":     p.get("est_loss"),
            "estimated_profit":   p.get("est_profit"),
            "rr_ratio":           p.get("rr_ratio"),
            "status":             t["status"],
            "flags":              t["flags"],
            "sector":             t.get("sector", "Unknown"),
            "claude_verdict":     v.get("verdict", "N/A"),
            "biggest_risk":       v.get("biggest_risk", "N/A"),
            "confidence_booster": v.get("confidence_booster", "N/A"),
        })

    with open(REPORT_PATH, "w") as f:
        json.dump({
            "regime":       regime,
            "risk_pct":     risk_pct,
            "account_size": ACCOUNT_SIZE,
            "trades":       report_records,
            "projections":  proj,
        }, f, indent=2)
    print(f"\nRisk report saved to risk_report.json")

    # ── Save / update trade_approvals.json ────────────────────────────────────
    # Preserve any existing user decisions (approved / executed flags).
    et_tz     = pytz.timezone("America/New_York")
    timestamp = datetime.datetime.now(et_tz).strftime("%Y-%m-%d %H:%M ET")

    existing_map: dict = {}
    if os.path.exists(APPROVALS_PATH):
        existing_raw = load_json(APPROVALS_PATH, {})
        if isinstance(existing_raw, dict) and "approvals" in existing_raw:
            for entry in existing_raw["approvals"]:
                t = entry.get("ticker", "").upper()
                existing_map[t] = entry

    new_approvals: list = []
    for t in trades_with_risk:
        ticker_up = t["ticker"].upper()
        p         = t["position"]
        existing  = existing_map.get(ticker_up, {})
        new_approvals.append({
            "ticker":           ticker_up,
            "approved":         existing.get("approved", False),
            "executed":         existing.get("executed", False),
            "direction":        t["direction"],
            "entry":            t["entry"],
            "stop":             t["stop"],
            "target":           t["target"],
            "shares":           p.get("shares"),
            "position_dollars": p.get("pos_dollars"),
            "risk_dollars":     p.get("est_loss"),
            "status":           t["status"],
            "timestamp":        timestamp,
        })

    with open(APPROVALS_PATH, "w") as f:
        json.dump({"approvals": new_approvals}, f, indent=2)
    print(f"Trade approvals saved to trade_approvals.json")


if __name__ == "__main__":
    main()
