import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import re
import json
import sys
import math
import time
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic
import yfinance as yf
import pandas as pd


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


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

load_dotenv()

WATCHLIST_PATH  = os.path.join(os.path.dirname(__file__), "watchlist.json")
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "technical_analysis.json")
SHORTLIST_PATH  = os.path.join(os.path.dirname(__file__), "shortlist.json")

BATCH_SIZE = 10

SYSTEM_PROMPT = (
    "You are an expert technical analyst trained on the Candlestick Trading Bible and "
    "John Murphy's Technical Analysis of Financial Markets. For each stock you will:\n"
    "1. Identify the most significant candlestick pattern in the last 3 candles\n"
    "2. Determine if price is above or below 50 and 200 day moving averages\n"
    "3. Assess RSI - overbought above 70, oversold below 30, or neutral\n"
    "4. State trend direction - uptrend, downtrend or sideways\n"
    "5. Give a specific entry price, stop loss price and target price\n"
    "6. Give a confidence score out of 10\n"
    "Only recommend a trade if at least 3 factors align. Always define risk first."
)


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def fetch_technical_data(ticker: str) -> dict | None:
    try:
        end   = datetime.today()
        start = end - timedelta(days=365)  # extra history for reliable MAs
        with suppress_stderr():
            stock = yf.Ticker(ticker)
            hist  = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty or len(hist) < 60:
            return None

        closes  = hist["Close"]
        volumes = hist["Volume"]

        current_price = round(float(closes.iloc[-1]), 4)
        ma50  = round(float(closes.rolling(50).mean().iloc[-1]), 4)
        ma200 = round(float(closes.rolling(200).mean().iloc[-1]), 4)
        rsi   = compute_rsi(closes)

        avg_vol_20 = float(volumes.iloc[-20:].mean())
        avg_vol_5  = float(volumes.iloc[-5:].mean())
        volume_trend = (
            "increasing" if avg_vol_5 > avg_vol_20 * 1.1
            else "decreasing" if avg_vol_5 < avg_vol_20 * 0.9
            else "stable"
        )

        # Last 3 candles for pattern recognition
        last3 = hist.tail(3)[["Open", "High", "Low", "Close", "Volume"]].copy()
        last3 = last3.round(4)
        candles = last3.reset_index().rename(columns={"Date": "date"})
        candles["date"] = candles["date"].dt.strftime("%Y-%m-%d")
        candles_list = candles.to_dict(orient="records")

        # 3-month subset for context
        hist_3m = hist.tail(63)[["Open", "High", "Low", "Close", "Volume"]].copy()
        hist_3m = hist_3m.round(4)

        adr = round(float((hist["High"] - hist["Low"]).tail(20).mean()), 4)

        return {
            "ticker":          ticker,
            "current_price":   current_price,
            "ma50":            ma50,
            "ma200":           ma200,
            "rsi_14":          rsi,
            "volume_trend":    volume_trend,
            "avg_daily_range": adr,
            "last_3_candles":  candles_list,
            "data_points":     len(hist_3m),
            "volume_nonzero":  float(volumes.sum()) > 0,
        }
    except Exception:
        return None


def data_is_valid(data: dict) -> bool:
    if data.get("data_points", 0) < 30:
        return False

    for field in ("current_price", "ma50", "ma200", "rsi_14", "avg_daily_range"):
        val = data.get(field)
        if val is None:
            return False
        try:
            f = float(val)
        except (TypeError, ValueError):
            return False
        if math.isnan(f) or math.isinf(f):
            return False
        if field == "current_price" and f == 0.0:
            return False

    candles = data.get("last_3_candles")
    if not candles or len(candles) == 0:
        return False

    if not data.get("volume_nonzero", False):
        return False

    return True


def build_data_block(all_data: list[dict]) -> str:
    lines = []
    for d in all_data:
        lines.append(f"--- {d['ticker']} ---")
        lines.append(f"Current Price : ${d['current_price']}")
        lines.append(f"50-day MA     : ${d['ma50']}")
        lines.append(f"200-day MA    : ${d['ma200']}")
        lines.append(f"RSI (14)      : {d['rsi_14']}")
        lines.append(f"Avg Daily Range: ${d['avg_daily_range']}")
        lines.append(f"Volume Trend  : {d['volume_trend']}")
        lines.append("Last 3 Candles (O/H/L/C/V):")
        for c in d["last_3_candles"]:
            lines.append(
                f"  {c['date']}  O:{c['Open']}  H:{c['High']}  "
                f"L:{c['Low']}  C:{c['Close']}  V:{int(c['Volume'])}"
            )
        lines.append("")
    return "\n".join(lines)


def analyze_with_claude(data_block: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    user_message = (
        "Here is the technical data for each stock on the watchlist:\n\n"
        f"{data_block}\n"
        "For EACH stock provide a structured analysis using this exact format:\n\n"
        "### [TICKER]\n"
        "**Candlestick Pattern (last 3 candles):** ...\n"
        "**MA Position:** Price vs 50-day and 200-day MA\n"
        "**RSI Assessment:** (Overbought / Oversold / Neutral) — value\n"
        "**Trend Direction:** (Uptrend / Downtrend / Sideways)\n"
        "**Volume:** (Confirming / Diverging)\n"
        "**Entry Price:** $...\n"
        "**Stop Loss:** $...\n"
        "**Target Price:** $...\n"
        "**Risk/Reward Ratio:** ...\n"
        "**Confidence Score:** .../10\n"
        "**Trade Recommendation:** (LONG / SHORT / NO TRADE) — one sentence rationale\n"
    )

    message = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def parse_analysis_to_records(analysis: str, all_data: list[dict]) -> list[dict]:
    records = []
    for d in all_data:
        ticker  = d["ticker"]
        pat     = rf"###\s*{re.escape(ticker)}\b(.*?)(?=###|\Z)"
        match   = re.search(pat, analysis, re.DOTALL)
        block   = match.group(1).strip() if match else "Analysis not found."

        def extract(label: str, blk: str = block) -> str:
            m = re.search(rf"\*\*{re.escape(label)}.*?\*\*[:\s]*(.*)", blk)
            return m.group(1).strip() if m else ""

        records.append({
            "ticker":              ticker,
            "current_price":       d["current_price"],
            "candlestick_pattern": extract("Candlestick Pattern"),
            "ma_position":         extract("MA Position"),
            "rsi_assessment":      extract("RSI Assessment"),
            "trend_direction":     extract("Trend Direction"),
            "volume":              extract("Volume"),
            "entry_price":         extract("Entry Price"),
            "stop_loss":           extract("Stop Loss"),
            "target_price":        extract("Target Price"),
            "risk_reward":         extract("Risk/Reward Ratio"),
            "confidence_score":    extract("Confidence Score"),
            "recommendation":      extract("Trade Recommendation"),
            "full_analysis":       block,
        })
    return records


def build_shortlist_prompt(records: list[dict], data_by_ticker: dict[str, dict]) -> str:
    lines = ["Here are the full technical analysis results for all stocks:\n"]
    for entry in records:
        raw = data_by_ticker.get(entry["ticker"], {})
        lines.append(f"TICKER: {entry['ticker']}  |  Price: ${entry['current_price']}  |  ADR: ${raw.get('avg_daily_range', 'N/A')}")
        lines.append(f"  Pattern      : {entry['candlestick_pattern']}")
        lines.append(f"  MA Position  : {entry['ma_position']}")
        lines.append(f"  RSI          : {entry['rsi_assessment']}")
        lines.append(f"  Trend        : {entry['trend_direction']}")
        lines.append(f"  Volume       : {entry['volume']}")
        lines.append(f"  Entry        : {entry['entry_price']}")
        lines.append(f"  Stop Loss    : {entry['stop_loss']}")
        lines.append(f"  Target       : {entry['target_price']}")
        lines.append(f"  R/R Ratio    : {entry['risk_reward']}")
        lines.append(f"  Confidence   : {entry['confidence_score']}")
        lines.append(f"  Recommendation: {entry['recommendation']}")
        lines.append("")
    lines.append(
        "Review ALL of the above and select the TOP 10 stocks with the strongest chart setups — "
        "best combination of trend alignment, candlestick pattern, RSI, and risk/reward ratio.\n\n"
        "Today's date is April 21, 2025. For Time Horizon, use the ADR and entry-to-target "
        "distance to estimate how many trading days the move might take, then convert to a "
        "calendar date range from April 21 2025 (skip weekends).\n\n"
        "For each of the top 10 use this exact format:\n\n"
        "### [RANK]. [TICKER]\n"
        "**Why Selected:** (explain what combination of factors makes this the best setup)\n"
        "**Edge Over Others:** (what specifically makes it stronger than the stocks not selected)\n"
        "**Setup Type:** (Momentum / Mean Reversion / Breakout / Trend Continuation)\n"
        "**Time Horizon:** X to Y trading days (approx. [Month DD] – [Month DD], 2025)\n"
        "**Final Recommendation:** (LONG / SHORT) — entry $X, stop $Y, target $Z\n"
    )
    return "\n".join(lines)


def select_top10_with_claude(records: list[dict], data_by_ticker: dict[str, dict]) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = build_shortlist_prompt(records, data_by_ticker)
    message = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=3500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _parse_price(value: str) -> float | None:
    """Strip $ and commas then parse to float. Returns None on failure."""
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _calc_rr(entry: str, stop: str, target: str, direction: str) -> str:
    """Calculate risk/reward ratio from price strings. Returns formatted string or ''."""
    e = _parse_price(entry)
    s = _parse_price(stop)
    t = _parse_price(target)
    if None in (e, s, t):
        return ""
    if direction.upper() == "SHORT":
        risk   = s - e
        reward = e - t
    else:
        risk   = e - s
        reward = t - e
    if risk <= 0:
        return ""
    return f"{round(reward / risk, 2)}:1"


def _first_nonempty(*values) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return "N/A"


def parse_shortlist(shortlist_text: str, records: list[dict], data_by_ticker: dict[str, dict]) -> list[dict]:
    records_by_ticker = {r["ticker"]: r for r in records}
    shortlist = []
    seen: set[str] = set()

    for match in re.finditer(r"###\s*\d+\.\s*([A-Z]{1,5})\b(.*?)(?=###\s*\d+\.|\Z)", shortlist_text, re.DOTALL):
        ticker = match.group(1).strip()
        if ticker in seen:
            continue
        seen.add(ticker)

        block = match.group(2).strip()

        def extract(label: str, blk: str = block) -> str:
            m = re.search(rf"\*\*{re.escape(label)}.*?\*\*[:\s]*(.*)", blk)
            return m.group(1).strip() if m else ""

        base = records_by_ticker.get(ticker, {})
        raw  = _lookup_raw(ticker, data_by_ticker)

        # Try to pull trade levels from the Final Recommendation text first,
        # then fall back to what was parsed from the full technical analysis.
        final_rec = extract("Final Recommendation")
        inline_entry  = re.search(r"entry[:\s]+\$?([\d.,]+)", final_rec, re.IGNORECASE)
        inline_stop   = re.search(r"stop[:\s]+\$?([\d.,]+)", final_rec, re.IGNORECASE)
        inline_target = re.search(r"target[:\s]+\$?([\d.,]+)", final_rec, re.IGNORECASE)

        direction_match = re.search(r"\b(LONG|SHORT)\b", final_rec, re.IGNORECASE)
        direction = direction_match.group(1).upper() if direction_match else _first_nonempty(base.get("recommendation", ""))

        # Always pull current_price from yfinance data — never rely on Claude for it
        yf_price = raw.get("current_price")
        current_price = str(yf_price) if yf_price is not None else _first_nonempty(base.get("current_price"))

        entry_val  = _first_nonempty(
                         inline_entry.group(1) if inline_entry else None,
                         base.get("entry_price"),
                     )
        stop_val   = _first_nonempty(
                         inline_stop.group(1) if inline_stop else None,
                         base.get("stop_loss"),
                     )
        target_val = _first_nonempty(
                         inline_target.group(1) if inline_target else None,
                         base.get("target_price"),
                     )

        rr_claude = _first_nonempty(base.get("risk_reward"))
        rr_calc   = _calc_rr(entry_val, stop_val, target_val, direction)
        # Use Claude's R/R only when it looks like a real ratio; otherwise compute it
        rr_final  = rr_claude if re.match(r"[\d.]+\s*:", rr_claude) else (rr_calc or rr_claude)

        # Confidence: use Claude's value if real; shortlist selection implies at least 7/10
        raw_conf = _first_nonempty(base.get("confidence_score"))
        confidence = raw_conf if (raw_conf != "N/A" and "/" in raw_conf) else "7/10"

        shortlist.append({
            "ticker":               ticker,
            "current_price":        current_price,
            "direction":            direction,
            "why_selected":         extract("Why Selected"),
            "edge_over_others":     extract("Edge Over Others"),
            "setup_type":           extract("Setup Type"),
            "time_horizon":         extract("Time Horizon"),
            "final_recommendation": final_rec,
            "entry_price":          entry_val,
            "stop_loss":            stop_val,
            "target_price":         target_val,
            "risk_reward":          rr_final,
            "confidence_score":     confidence,
        })

    return shortlist


def _lookup_raw(ticker: str, data_by_ticker: dict) -> dict:
    """Return raw yfinance data for ticker, trying prefix match for crypto (BTC → BTC-USD)."""
    if ticker in data_by_ticker:
        return data_by_ticker[ticker]
    for key in data_by_ticker:
        if key.startswith(ticker + "-") or key.startswith(ticker + "."):
            return data_by_ticker[key]
    return {}


def _th_short(time_horizon: str) -> str:
    """Return just the 'X to Y trading days' part for the compact table row."""
    if not time_horizon or time_horizon == "N/A":
        return "N/A"
    paren = time_horizon.find("(")
    short = time_horizon[:paren].strip() if paren != -1 else time_horizon
    return short[:28]


def print_shortlist(shortlist: list[dict]) -> None:
    W = 118
    print("\n" + "=" * W)
    print("                           TOP 10 SHORTLIST — STRONGEST SETUPS")
    print("=" * W)

    # Summary table
    col = "{:>4}  {:<6}  {:<5}  {:>10}  {:>10}  {:>10}  {:>6}  {:<28}  {:<10}"
    print(col.format("Rank", "Ticker", "Dir", "Entry", "Stop", "Target", "R/R", "Time Horizon", "Confidence"))
    print("─" * W)
    for i, e in enumerate(shortlist, 1):
        rr   = (e.get("risk_reward") or "N/A")[:6]
        th   = _th_short(e.get("time_horizon", "N/A"))
        conf = (e.get("confidence_score") or "N/A")[:10]
        def clean(v: str) -> str:
            return v.replace(",", "").rstrip(". ")[:10]

        print(col.format(
            i,
            e["ticker"],
            e.get("direction", "N/A")[:5],
            clean(e["entry_price"]),
            clean(e["stop_loss"]),
            clean(e["target_price"]),
            rr,
            th,
            conf,
        ))

    # Detailed breakdown
    print("\n" + "=" * W)
    print("                          DETAILED BREAKDOWN")
    print("=" * W)
    for i, entry in enumerate(shortlist, 1):
        print(f"\n{'─' * W}")
        print(f"  #{i}  {entry['ticker']}  |  Price: ${entry['current_price']}  |  {entry.get('setup_type', 'N/A')}  |  {entry.get('direction', 'N/A')}")
        print(f"{'─' * W}")
        print(f"  Why Selected     : {entry['why_selected']}")
        print(f"  Edge Over Others : {entry['edge_over_others']}")
        print(f"  Setup Type       : {entry.get('setup_type', 'N/A')}")
        print(f"  Time Horizon     : {entry.get('time_horizon', 'N/A')}")
        print(f"  Recommendation   : {entry['final_recommendation']}")
        print(f"  Entry            : {entry['entry_price']}")
        print(f"  Stop Loss        : {entry['stop_loss']}")
        print(f"  Target           : {entry['target_price']}")
        print(f"  Risk/Reward      : {entry.get('risk_reward', 'N/A')}")
        print(f"  Confidence       : {entry['confidence_score']}")
    print("\n" + "=" * W)


def print_report(records: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("        TECHNICAL ANALYSIS REPORT")
    print("=" * 62)
    for r in records:
        print(f"\n{'─' * 62}")
        print(f"  {r['ticker']}  |  Current Price: ${r['current_price']}")
        print(f"{'─' * 62}")
        print(f"  Candlestick Pattern : {r['candlestick_pattern']}")
        print(f"  MA Position         : {r['ma_position']}")
        print(f"  RSI                 : {r['rsi_assessment']}")
        print(f"  Trend               : {r['trend_direction']}")
        print(f"  Volume              : {r['volume']}")
        print(f"  Entry               : {r['entry_price']}")
        print(f"  Stop Loss           : {r['stop_loss']}")
        print(f"  Target              : {r['target_price']}")
        print(f"  Risk/Reward         : {r['risk_reward']}")
        print(f"  Confidence          : {r['confidence_score']}")
        print(f"  Recommendation      : {r['recommendation']}")
    print("\n" + "=" * 62)


def main():
    print("=" * 62)
    print("       TRADING AGENT — TECHNICAL ANALYSIS")
    print("=" * 62)

    if not os.path.exists(WATCHLIST_PATH):
        print("Please run news_agent.py first.")
        sys.exit(1)

    with open(WATCHLIST_PATH) as f:
        tickers = json.load(f)

    if not tickers:
        print("watchlist.json is empty. Please run news_agent.py first.")
        sys.exit(1)

    print(f"\nWatchlist: {len(tickers)} tickers")
    print("\nFetching and validating technical data...")

    all_data = []
    for ticker in tickers:
        data = fetch_technical_data(ticker)
        if data is None or not data_is_valid(data):
            print(f"  {ticker}: removed (failed validation)")
            continue
        all_data.append(data)

    if not all_data:
        print("No valid tickers after validation. Exiting.")
        sys.exit(1)

    batches = [all_data[i : i + BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"\nValidated {len(all_data)}/{len(tickers)} tickers — sending to Claude in {total_batches} batch(es) (3 parallel workers)")

    def analyze_batch(batch: list[dict], batch_num: int, total: int) -> dict[str, str]:
        data_block = build_data_block(batch)
        analysis   = analyze_with_claude(data_block)
        print(f"  Batch {batch_num}/{total} complete ({len(batch)} tickers analyzed)")
        return {batch_num: analysis}

    batch_results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for i, batch in enumerate(batches):
            futures[executor.submit(analyze_batch, batch, i + 1, total_batches)] = i
            time.sleep(2)  # stagger submissions to avoid simultaneous burst
        for future in as_completed(futures):
            batch_results.update(future.result())

    # Reassemble in original order so parse_analysis_to_records matches all_data order
    combined_analysis = "\n".join(batch_results[n] for n in sorted(batch_results))
    records = parse_analysis_to_records(combined_analysis, all_data)
    data_by_ticker = {d["ticker"]: d for d in all_data}

    # Drop any records that came back with no candlestick pattern
    records = [r for r in records if r.get("candlestick_pattern")]

    if not records:
        print("No records parsed from Claude response. Exiting.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to technical_analysis.json")

    print_report(records)

    print("\nAsking Claude to select the top 10 strongest setups...")
    shortlist_text = select_top10_with_claude(records, data_by_ticker)
    shortlist = parse_shortlist(shortlist_text, records, data_by_ticker)

    with open(SHORTLIST_PATH, "w") as f:
        json.dump(shortlist, f, indent=2)
    print(f"Shortlist saved to shortlist.json")

    print_shortlist(shortlist)


if __name__ == "__main__":
    main()
