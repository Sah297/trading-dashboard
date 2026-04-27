import os
import re
import json
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

load_dotenv()

BASE_DIR         = os.path.dirname(__file__)
REGIME_PATH      = os.path.join(BASE_DIR, "market_regime.json")

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

SYSTEM_PROMPT = (
    "You are a macro analyst and geopolitical risk expert trained on Ray Dalio's "
    "Principles and market cycle theory. Analyze these headlines and:\n"
    "1. Identify any active macro events (wars, Fed decisions, trade wars, "
    "sanctions, elections, oil shocks, pandemics, banking crises)\n"
    "2. For each event assess: which sectors it HELPS and which it HURTS\n"
    "3. Determine overall market regime: BULLISH, CAUTIOUS, DEFENSIVE or PAUSE\n"
    "4. If PAUSE: explain exactly what needs to resolve before trading resumes\n"
    "5. If CAUTIOUS or DEFENSIVE: list which asset classes are safe havens now\n"
    "6. Identify the single most important macro risk the market is ignoring\n"
    "Return a structured analysis with clear REGIME decision at the top."
)


def fetch_feed(url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title   = (item.findtext("title") or "").strip()
            summary = (item.findtext("description") or "").strip()
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
            line += f"\n   {item['summary'][:200].replace(chr(10), ' ')}"
        lines.append(line)
    return "\n".join(lines)


def analyze_with_claude(news_block: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    user_message = (
        f"Here are today's financial and geopolitical news headlines:\n\n{news_block}\n\n"
        "Analyze the macro environment and return ONLY a valid JSON object with these exact fields. "
        "No markdown, no backticks, no preamble, no explanation outside the JSON — just raw JSON:\n"
        "{\n"
        '  "regime": "BULLISH or CAUTIOUS or DEFENSIVE or PAUSE — pick exactly one",\n'
        '  "active_events": ["list of active macro events detected"],\n'
        '  "sectors_to_avoid": ["list of sectors to avoid"],\n'
        '  "sectors_to_favor": ["list of sectors with tailwinds"],\n'
        '  "safe_havens": ["list of safe haven assets if regime is CAUTIOUS/DEFENSIVE"],\n'
        '  "key_ignored_risk": "single sentence describing the most important ignored risk",\n'
        '  "reasoning": "2-3 sentence summary of why this regime was chosen",\n'
        '  "pause_condition": "what must resolve before trading resumes, empty string if not PAUSE"\n'
        "}\n"
        "The regime field must be exactly one of: BULLISH, CAUTIOUS, DEFENSIVE, PAUSE."
    )
    message = call_claude_with_retry(
        client,
        model="claude-sonnet-4-5",
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


VALID_REGIMES = ("BULLISH", "CAUTIOUS", "DEFENSIVE", "PAUSE")


def _fallback_regime(text: str) -> str:
    upper = text.upper()
    # Scan in priority order: PAUSE first (most restrictive), then DEFENSIVE, CAUTIOUS, BULLISH
    for r in ("PAUSE", "DEFENSIVE", "CAUTIOUS", "BULLISH"):
        if r in upper:
            return r
    return "CAUTIOUS"


def parse_regime(response: str) -> dict:
    # Strip any accidental backtick fences before attempting parse
    cleaned = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    # Primary: parse entire response as JSON
    try:
        data = json.loads(cleaned)
        regime = str(data.get("regime", "")).strip().upper()
        if regime not in VALID_REGIMES:
            regime = _fallback_regime(cleaned)
            print(f"  Warning: Claude returned unrecognised regime — inferred '{regime}' from text")
        data["regime"] = regime
        return data
    except json.JSONDecodeError:
        pass

    # Secondary: try to extract the first {...} block from the response
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            regime = str(data.get("regime", "")).strip().upper()
            if regime not in VALID_REGIMES:
                regime = _fallback_regime(cleaned)
            data["regime"] = regime
            return data
        except json.JSONDecodeError:
            pass

    # Fallback: extract regime from raw text only
    regime = _fallback_regime(response)
    print(f"  Warning: JSON parsing failed entirely — inferred regime '{regime}' from raw text")
    return {
        "regime":           regime,
        "active_events":    [],
        "sectors_to_avoid": [],
        "sectors_to_favor": [],
        "safe_havens":      [],
        "key_ignored_risk": "",
        "reasoning":        response[:500],
        "pause_condition":  "",
    }


def print_summary(data: dict) -> None:
    W = 62
    print("\n" + "=" * W)
    print("          MACRO REGIME ANALYSIS")
    print("=" * W)
    print(f"  REGIME         : {data['regime']}")
    print(f"  Active Events  : {', '.join(data['active_events']) or 'None detected'}")
    print(f"  Sectors Favor  : {', '.join(data['sectors_to_favor']) or 'N/A'}")
    print(f"  Sectors Avoid  : {', '.join(data['sectors_to_avoid']) or 'N/A'}")
    print(f"  Safe Havens    : {', '.join(data['safe_havens']) or 'N/A'}")
    print(f"  Ignored Risk   : {data['key_ignored_risk']}")
    if data.get("pause_condition"):
        print(f"  Pause Until    : {data['pause_condition']}")
    print(f"\n  Reasoning: {data['reasoning']}")
    print("=" * W)


def main():
    print("=" * 62)
    print("        TRADING AGENT — MACRO ANALYSIS")
    print("=" * 62)

    print("\nFetching RSS feeds...")
    items = fetch_all_news()
    print(f"Total unique headlines collected: {len(items)}")

    if len(items) < 5:
        print("Too few headlines fetched. Check your network or feeds.")
        return

    news_block = build_news_block(items)

    print("\nSending headlines to Claude for macro analysis...")
    response = analyze_with_claude(news_block)

    data = parse_regime(response)
    print(f"Regime identified: {data['regime']}")

    with open(REGIME_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Market regime saved to market_regime.json (regime={data['regime']})")

    print_summary(data)


if __name__ == "__main__":
    main()
