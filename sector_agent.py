import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import sys
import math
import re
import time
import contextlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic
import yfinance as yf
import pandas as pd

load_dotenv()

BASE_DIR        = os.path.dirname(__file__)
REGIME_PATH     = os.path.join(BASE_DIR, "market_regime.json")
HOT_PATH        = os.path.join(BASE_DIR, "hot_sectors.json")

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLC":  "Communications",
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


SYSTEM_PROMPT = (
    "You are a sector rotation expert. Given the current macro regime and "
    "ETF momentum data, identify the top 3 hot sectors and bottom 3 sectors "
    "to avoid. For each hot sector list 5 specific stock tickers within that "
    "sector that are most likely to benefit. Consider both the technical momentum "
    "of the ETF and the macro tailwinds from the regime analysis. "
    "Explain your reasoning clearly."
)


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    val   = float(rsi.iloc[-1])
    return round(val, 2) if not math.isnan(val) else 50.0


def fetch_etf_data(etf: str) -> dict | None:
    try:
        end   = datetime.today()
        start = end - timedelta(days=120)
        with suppress_stderr():
            hist = yf.Ticker(etf).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
        if hist.empty or len(hist) < 30:
            return None

        closes  = hist["Close"]
        volumes = hist["Volume"]
        current = float(closes.iloc[-1])

        ret_1m  = round((current / float(closes.iloc[-21]) - 1) * 100, 2) if len(closes) >= 21 else None
        ret_3m  = round((current / float(closes.iloc[0])  - 1) * 100, 2)
        rsi     = compute_rsi(closes)
        ma50    = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
        ma_pos  = "above" if (ma50 and current > ma50) else "below"

        avg_vol_20 = float(volumes.iloc[-20:].mean())
        avg_vol_5  = float(volumes.iloc[-5:].mean())
        vol_trend  = (
            "increasing" if avg_vol_5 > avg_vol_20 * 1.1
            else "decreasing" if avg_vol_5 < avg_vol_20 * 0.9
            else "stable"
        )

        return {
            "etf":          etf,
            "name":         SECTOR_ETFS[etf],
            "current":      round(current, 2),
            "return_1m":    ret_1m,
            "return_3m":    ret_3m,
            "rsi":          rsi,
            "ma50_position": ma_pos,
            "volume_trend": vol_trend,
        }
    except Exception:
        return None


def build_etf_block(etf_data: list[dict], regime: dict) -> str:
    lines = [
        f"MACRO REGIME: {regime.get('regime', 'UNKNOWN')}",
        f"Sectors to Favor : {', '.join(regime.get('sectors_to_favor', []))}",
        f"Sectors to Avoid : {', '.join(regime.get('sectors_to_avoid', []))}",
        f"Key Macro Risk   : {regime.get('key_ignored_risk', '')}",
        "",
        "SECTOR ETF DATA:",
    ]
    for d in etf_data:
        lines.append(
            f"  {d['etf']} ({d['name']}): "
            f"Price ${d['current']}  "
            f"1M {d['return_1m']:+.1f}%  "
            f"3M {d['return_3m']:+.1f}%  "
            f"RSI {d['rsi']}  "
            f"MA50 {d['ma50_position']}  "
            f"Vol {d['volume_trend']}"
        )
    return "\n".join(lines)


def analyze_with_claude(etf_block: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    user_message = (
        f"{etf_block}\n\n"
        "Based on the macro regime and ETF momentum above, identify:\n"
        "- TOP 3 HOT SECTORS with 5 specific stock tickers each\n"
        "- BOTTOM 3 SECTORS to avoid\n\n"
        "At the very end output a JSON block wrapped in ```json ... ``` with:\n"
        "{\n"
        '  "hot_sectors": [\n'
        '    {"name": "...", "etf": "XLK", "reason": "...", "tickers": ["T1","T2","T3","T4","T5"]},\n'
        '    ...\n'
        '  ],\n'
        '  "cold_sectors": [\n'
        '    {"name": "...", "etf": "XLU", "reason": "..."},\n'
        '    ...\n'
        '  ]\n'
        "}"
    )
    message = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=2500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def parse_sectors(response: str) -> dict:
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            all_tickers: list[str] = []
            for s in data.get("hot_sectors", []):
                all_tickers.extend(s.get("tickers", []))
            data["sector_tickers"] = list(dict.fromkeys(all_tickers))
            return data
        except json.JSONDecodeError:
            pass
    return {"hot_sectors": [], "cold_sectors": [], "sector_tickers": []}


def print_summary(data: dict) -> None:
    W = 62
    print("\n" + "=" * W)
    print("          SECTOR ROTATION ANALYSIS")
    print("=" * W)
    print("\n  HOT SECTORS:")
    for i, s in enumerate(data.get("hot_sectors", []), 1):
        tickers = ", ".join(s.get("tickers", []))
        print(f"    {i}. {s['etf']} — {s['name']}")
        print(f"       Reason  : {s['reason']}")
        print(f"       Tickers : {tickers}")
    print("\n  COLD SECTORS:")
    for i, s in enumerate(data.get("cold_sectors", []), 1):
        print(f"    {i}. {s['etf']} — {s['name']}")
        print(f"       Reason  : {s['reason']}")
    flat = data.get("sector_tickers", [])
    print(f"\n  Sector tickers added to watchlist: {flat}")
    print("=" * W)


def main():
    print("=" * 62)
    print("        TRADING AGENT — SECTOR ROTATION")
    print("=" * 62)

    regime: dict = {}
    if os.path.exists(REGIME_PATH):
        with open(REGIME_PATH) as f:
            regime = json.load(f)
        print(f"\nLoaded macro regime: {regime.get('regime', 'UNKNOWN')}")
    else:
        print("\nNo market_regime.json found — running without macro context")

    print("\nFetching sector ETF data...")
    etf_data = []
    for etf in SECTOR_ETFS:
        d = fetch_etf_data(etf)
        if d:
            print(f"  {etf}: {d['return_1m']:+.1f}% 1M  RSI {d['rsi']}  MA50 {d['ma50_position']}")
            etf_data.append(d)
        else:
            print(f"  {etf}: skipped (no data)")

    if not etf_data:
        print("No ETF data available. Exiting.")
        sys.exit(1)

    etf_block = build_etf_block(etf_data, regime)

    print("\nSending sector data to Claude for analysis...")
    response = analyze_with_claude(etf_block)

    data = parse_sectors(response)

    with open(HOT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Sector analysis saved to hot_sectors.json")

    print_summary(data)


if __name__ == "__main__":
    main()
