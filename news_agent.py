import os
import re
import json
import sys
import time
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import anthropic


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

BASE_DIR       = os.path.dirname(__file__)
WATCHLIST_PATH = os.path.join(BASE_DIR, "watchlist.json")
REGIME_PATH    = os.path.join(BASE_DIR, "market_regime.json")
HOT_PATH       = os.path.join(BASE_DIR, "hot_sectors.json")

load_dotenv()

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US&count=50",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://www.investing.com/rss/news.rss",
    "https://www.marketwatch.com/rss/topstories",
    "https://seekingalpha.com/feed.xml",
    "https://www.fool.com/feeds/index.aspx",
    "https://finance.yahoo.com/news/rssindex",
    "https://www.benzinga.com/feed",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

NS = {
    "media": "http://search.yahoo.com/mrss/",
    "dc":    "http://purl.org/dc/elements/1.1/",
}

BASE_SYSTEM_PROMPT = (
    "You are an expert stock analyst trained on Benjamin Graham's margin of safety, "
    "Peter Lynch's growth investing principles, and market sentiment analysis. "
    "You will be given a large batch of financial news headlines. Your job is to: "
    "1) Identify every company or stock ticker mentioned or implied in the news. "
    "2) Analyze the sentiment and potential impact for each. "
    "3) Rank and return the top 10 stocks to watch today with ticker symbol, company name, "
    "reason, investing lens, and recommended action. "
    "Focus on finding genuine opportunities the market may be mispricing. "
    "You must also scan broadly and return as many tickers as possible — minimum 30, maximum 50."
)


def load_regime() -> dict:
    if os.path.exists(REGIME_PATH):
        with open(REGIME_PATH) as f:
            return json.load(f)
    return {}


def load_sector_tickers() -> list[str]:
    if os.path.exists(HOT_PATH):
        with open(HOT_PATH) as f:
            data = json.load(f)
        return data.get("sector_tickers", [])
    return []


def _text(element, tag: str, ns_prefix: str | None = None) -> str:
    if ns_prefix:
        child = element.find(f"{{{NS[ns_prefix]}}}{tag}")
    else:
        child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def fetch_feed(url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title   = _text(item, "title")
            summary = _text(item, "description")
            if title:
                items.append({"title": title, "summary": summary})
        return items
    except Exception:
        return []


def fetch_all_news() -> list[dict]:
    all_items: list[dict] = []
    for url in RSS_FEEDS:
        all_items.extend(fetch_feed(url))
    seen: set[str] = set()
    unique = []
    for item in all_items:
        key = item["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def build_news_block(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        line = f"{i}. {item['title']}"
        if item["summary"]:
            summary = item["summary"][:200].replace("\n", " ")
            line += f"\n   Summary: {summary}"
        lines.append(line)
    return "\n".join(lines)


def build_system_prompt(regime: dict) -> str:
    prompt = BASE_SYSTEM_PROMPT
    r = regime.get("regime", "")
    if not r:
        return prompt

    prompt += f"\n\nCURRENT MARKET REGIME: {r}."

    if r in ("CAUTIOUS", "DEFENSIVE"):
        safe = regime.get("safe_havens", [])
        avoid = regime.get("sectors_to_avoid", [])
        favor = regime.get("sectors_to_favor", [])
        if safe:
            prompt += f" Safe havens right now: {', '.join(safe)}."
        if avoid:
            prompt += f" Sectors to avoid: {', '.join(avoid)}."
        if favor:
            prompt += f" Sectors with tailwinds: {', '.join(favor)}."
        prompt += (
            " Adjust recommendations accordingly — downgrade any pick in an "
            "at-risk sector and flag defensive or safe-haven names more prominently."
        )
    elif r == "BULLISH":
        favor = regime.get("sectors_to_favor", [])
        if favor:
            prompt += f" Sectors with strongest macro tailwinds: {', '.join(favor)}."
        prompt += " Lean towards growth and momentum names."

    risk = regime.get("key_ignored_risk", "")
    if risk:
        prompt += f" Key ignored macro risk to be aware of: {risk}."

    return prompt


def analyze_with_claude(news_block: str, regime: dict) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system_prompt = build_system_prompt(regime)

    regime_note = ""
    r = regime.get("regime", "")
    if r in ("CAUTIOUS", "DEFENSIVE"):
        safe = regime.get("safe_havens", [])
        regime_note = (
            f"\n\n⚠️  REGIME WARNING: Current market regime is {r}. "
            f"Safe havens: {', '.join(safe) if safe else 'N/A'}. "
            "Prioritise capital preservation. Flag any high-risk picks clearly."
        )

    user_message = (
        f"Here are today's financial news headlines and summaries:\n\n"
        f"{news_block}{regime_note}\n\n"
        "Based on all of the above, identify the TOP 10 STOCKS TO WATCH TODAY. "
        "For each stock format your response exactly like this:\n\n"
        "### [RANK]. [TICKER] — [Company Name]\n"
        "**Reason:** ...\n"
        "**Investing Lens:** (Margin of Safety / Growth / Sentiment / Mixed)\n"
        "**Recommended Action:** (Buy Watch / Research Further / Cautious Watch / Avoid)\n\n"
        "After the top 10 list, add a brief 2-3 sentence Market Outlook summary.\n\n"
        "IMPORTANT: After the Market Outlook, identify every company mentioned or implied in "
        "the news and return as many tickers as possible, minimum 30, maximum 50. "
        "End your response with a line in exactly this format (ticker symbols only, no extra text):\n"
        "TICKERS: [JSON array, e.g. [\"AAPL\", \"MSFT\", ...]]"
    )

    message = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def extract_and_save_tickers(analysis: str, sector_tickers: list[str]) -> list[str]:
    tickers: list[str] = []

    json_match = re.search(r"TICKERS:\s*(\[.*?\])", analysis, re.DOTALL)
    if json_match:
        try:
            tickers = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    if not tickers:
        csv_match = re.search(r"TICKERS:\s*([A-Z][A-Z0-9,\s\-\.]{5,})", analysis, re.MULTILINE)
        if csv_match:
            tickers = [t.strip() for t in csv_match.group(1).split(",") if re.match(r"^[A-Z]{1,5}$", t.strip())]

    if not tickers:
        tickers = re.findall(r"###\s*\d+\.\s*([A-Z]{1,5})\s*[—-]", analysis)

    tickers = list(dict.fromkeys(tickers))

    # Append sector tickers from hot_sectors.json (deduplicated)
    for t in sector_tickers:
        if t not in tickers:
            tickers.append(t)

    with open(WATCHLIST_PATH, "w") as f:
        json.dump(tickers, f, indent=2)
    return tickers


def main():
    print("=" * 62)
    print("           TRADING AGENT — NEWS ANALYSIS")
    print("=" * 62)

    # Load macro regime — gate on PAUSE before doing anything else
    regime = load_regime()
    r = regime.get("regime", "")
    if r:
        print(f"\nMarket regime loaded: {r}")

    if r == "PAUSE":
        pause_cond = regime.get("pause_condition", "No condition specified.")
        print("\n" + "=" * 62)
        print("  ⛔  TRADING PAUSED — MARKET REGIME: PAUSE")
        print("=" * 62)
        print(f"  Condition to resume: {pause_cond}")
        print("  Run macro_agent.py again when conditions change.")
        print("=" * 62)
        sys.exit(0)

    if r in ("CAUTIOUS", "DEFENSIVE"):
        safe = regime.get("safe_havens", [])
        print("\n" + "⚠️  " * 10)
        print(f"  WARNING: Regime is {r}. Safe havens: {', '.join(safe) if safe else 'N/A'}")
        print("⚠️  " * 10)

    sector_tickers = load_sector_tickers()
    if sector_tickers:
        print(f"\nSector tickers loaded from hot_sectors.json: {sector_tickers}")

    print("\nFetching RSS feeds...")
    items = fetch_all_news()
    print(f"Total unique headlines collected: {len(items)}")

    if len(items) < 5:
        print("Too few headlines fetched. Check your network or feeds.")
        return

    news_block = build_news_block(items)

    print("\nSending headlines to Claude for analysis...")
    analysis = analyze_with_claude(news_block, regime)

    print("\n" + "=" * 62)
    if r in ("CAUTIOUS", "DEFENSIVE"):
        print(f"  ⚠️  TOP 10 STOCKS TO WATCH — REGIME: {r}")
    else:
        print("         TOP 10 STOCKS TO WATCH TODAY")
    print("=" * 62)
    print(analysis)
    print("=" * 62)

    tickers = extract_and_save_tickers(analysis, sector_tickers)
    print(f"\nWatchlist saved to watchlist.json ({len(tickers)} tickers): {tickers}")


if __name__ == "__main__":
    main()
