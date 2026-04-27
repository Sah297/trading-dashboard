import json
import os
import subprocess
import sys
import datetime
import pytz
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

DATA_FILES = {
    "account":    BASE_DIR / "account_state.json",
    "regime":     BASE_DIR / "market_regime.json",
    "sectors":    BASE_DIR / "hot_sectors.json",
    "watchlist":  BASE_DIR / "watchlist.json",
    "shortlist":  BASE_DIR / "shortlist.json",
    "risk":       BASE_DIR / "risk_report.json",
    "compliance": BASE_DIR / "compliance_report.json",
    "approvals":  BASE_DIR / "trade_approvals.json",
    "executed":   BASE_DIR / "executed_trades.json",
}

PYTHON        = sys.executable
SETTINGS_PATH = BASE_DIR / "settings.json"

SETTINGS_DEFAULTS: dict = {
    "trading_budget":     20000,
    "max_risk_per_trade": 0.01,
    "max_open_positions": 8,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(key: str, default: Any = None) -> Any:
    path = DATA_FILES[key]
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def save(key: str, data: Any) -> None:
    with open(DATA_FILES[key], "w") as f:
        json.dump(data, f, indent=2)


def now_et() -> str:
    et = pytz.timezone("America/New_York")
    return datetime.datetime.now(et).strftime("%Y-%m-%d %H:%M:%S ET")


def require(key: str) -> Any:
    data = load(key)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"{DATA_FILES[key].name} not found — run the pipeline first.",
        )
    return data


def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            stored = json.load(f)
        result = dict(SETTINGS_DEFAULTS)
        result.update(stored)
        return result
    settings = dict(SETTINGS_DEFAULTS)
    settings["last_updated"] = "Never"
    return settings


def save_settings(data: dict) -> dict:
    et = pytz.timezone("America/New_York")
    data["last_updated"] = datetime.datetime.now(et).strftime("%Y-%m-%d %H:%M ET")
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return data

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Trading Agent API",
    version="1.0",
    description="REST API for the trading agent pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class ApprovalBody(BaseModel):
    approved: bool
    notes: str = ""


class SettingsBody(BaseModel):
    trading_budget:     Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    max_open_positions: Optional[int]   = None


class BudgetBody(BaseModel):
    amount: float

# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Trading Agent API",
        "version": "1.0",
        "endpoints": [
            "GET  /health",
            "GET  /account",
            "GET  /regime",
            "GET  /sectors",
            "GET  /trades/shortlist",
            "GET  /trades/approved",
            "GET  /trades/cleared",
            "GET  /report/risk",
            "GET  /report/compliance",
            "GET  /settings",
            "POST /settings",
            "POST /budget",
            "GET  /debug/reports",
            "POST /trades/approve/{ticker}",
            "GET  /trades/executed",
            "POST /pipeline/run",
            "POST /pipeline/midday",
        ],
    }

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": now_et()}

# ── Account ───────────────────────────────────────────────────────────────────

@app.get("/account")
def account():
    state = require("account")
    # Back-fill new fields for account_state.json files written before this update
    committed = state.get("committed_capital") or sum(
        p.get("market_value", 0) for p in state.get("open_positions", [])
    )
    if "committed_capital"   not in state: state["committed_capital"]   = round(committed, 2)
    if "open_positions_count" not in state: state["open_positions_count"] = len(state.get("open_positions", []))
    if "open_positions_value" not in state: state["open_positions_value"] = round(committed, 2)
    # Always pull manual_budget from settings so it's fresh
    sett = load_settings()
    manual_budget = float(sett.get("trading_budget") or state.get("cash_available") or 0)
    state["manual_budget"]       = round(manual_budget, 2)
    state["remaining_budget"]    = round(max(0.0, manual_budget - committed), 2)
    state["available_to_invest"] = state["remaining_budget"]
    return state

# ── Regime ────────────────────────────────────────────────────────────────────

@app.get("/regime")
def regime():
    return require("regime")

# ── Sectors ───────────────────────────────────────────────────────────────────

@app.get("/sectors")
def sectors():
    return require("sectors")

# ── Trades ───────────────────────────────────────────────────────────────────

@app.get("/trades/shortlist")
def trades_shortlist():
    return require("shortlist")


def _risk_lookup(risk_raw: dict) -> dict[str, dict]:
    """Ticker → trade dict from risk_report["trades"]."""
    return {
        t["ticker"]: t
        for t in risk_raw.get("trades", [])
        if t.get("ticker")
    }

def _shortlist_lookup() -> dict[str, dict]:
    """Ticker → shortlist entry dict (includes setup_type, why_selected, etc.)."""
    path = DATA_FILES["shortlist"]
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    items = data if isinstance(data, list) else []
    return {t["ticker"]: t for t in items if t.get("ticker")}

def _merge(compliance_trade: dict, risk: dict, sl: dict) -> dict:
    """Return one merged trade record from all three sources."""
    return {
        # ── Compliance fields (base) ──────────────────────────────
        "ticker":              compliance_trade.get("ticker"),
        "direction":           compliance_trade.get("direction"),
        "entry":               compliance_trade.get("entry"),
        "stop":                compliance_trade.get("stop"),
        "target":              compliance_trade.get("target"),
        "shares":              compliance_trade.get("shares"),
        "position_dollars":    compliance_trade.get("position_dollars"),
        "position_pct":        compliance_trade.get("position_pct"),
        "sector":              compliance_trade.get("sector"),
        "compliance_verdict":  compliance_trade.get("claude_verdict"),
        "compliance_reason":   compliance_trade.get("claude_reason"),
        "compliance_pdt":      compliance_trade.get("compliance_pdt"),
        "compliance_earnings": compliance_trade.get("compliance_earnings"),
        "earnings_date":       compliance_trade.get("earnings_date"),
        # ── Risk-agent fields ─────────────────────────────────────
        "estimated_risk":     risk.get("estimated_risk"),
        "estimated_profit":   risk.get("estimated_profit"),
        "rr_ratio":           risk.get("rr_ratio"),
        "flags":              risk.get("flags"),
        "biggest_risk":       risk.get("biggest_risk"),
        "confidence_booster": risk.get("confidence_booster"),
        "risk_verdict":       risk.get("claude_verdict"),
        "status":             risk.get("status"),
        # ── Shortlist / technical-agent fields ────────────────────
        "setup_type":      sl.get("setup_type"),
        "why_selected":    sl.get("why_selected"),
        "edge_over_others": sl.get("edge_over_others"),
        "time_horizon":    sl.get("time_horizon"),
        "confidence":      sl.get("confidence_score"),
    }


@app.get("/trades/approved")
def trades_approved():
    risk_raw = require("risk")
    rl  = _risk_lookup(risk_raw)
    sl  = _shortlist_lookup()

    approved = [
        _merge(t, rl.get(t["ticker"], {}), sl.get(t["ticker"], {}))
        for t in risk_raw.get("trades", [])
        if t.get("status") == "APPROVED"
    ]
    return {
        "regime": risk_raw.get("regime"),
        "count":  len(approved),
        "trades": approved,
    }


@app.get("/trades/cleared")
def trades_cleared():
    approvals_raw = load("approvals", default={})
    risk_raw      = load("risk", default={})

    if not approvals_raw:
        raise HTTPException(
            status_code=404,
            detail="trade_approvals.json not found — run the pipeline first.",
        )

    # Build risk description lookup: ticker → risk record
    risk_by_ticker = {
        t["ticker"]: t
        for t in risk_raw.get("trades", [])
        if t.get("ticker")
    }

    approvals_list = (
        approvals_raw.get("approvals", [])
        if isinstance(approvals_raw, dict) and "approvals" in approvals_raw
        else []
    )

    trades = []
    for entry in approvals_list:
        if entry.get("status") != "APPROVED":
            continue
        ticker = entry.get("ticker", "")
        risk   = risk_by_ticker.get(ticker, {})
        trades.append({
            "ticker":             ticker,
            "approved":           entry.get("approved", False),
            "executed":           entry.get("executed", False),
            "direction":          entry.get("direction"),
            "entry":              entry.get("entry"),
            "stop":               entry.get("stop"),
            "target":             entry.get("target"),
            "shares":             entry.get("shares"),
            "position_dollars":   entry.get("position_dollars"),
            "risk_dollars":       entry.get("risk_dollars"),
            "status":             entry.get("status"),
            "timestamp":          entry.get("timestamp"),
            "rr_ratio":           risk.get("rr_ratio"),
            "estimated_risk":     risk.get("estimated_risk"),
            "estimated_profit":   risk.get("estimated_profit"),
            "flags":              risk.get("flags"),
            "biggest_risk":       risk.get("biggest_risk"),
            "confidence_booster": risk.get("confidence_booster"),
            "claude_verdict":     risk.get("claude_verdict"),
        })

    return {
        "count":  len(trades),
        "regime": risk_raw.get("regime"),
        "trades": trades,
    }

# ── Full reports ──────────────────────────────────────────────────────────────

@app.get("/report/risk")
def report_risk():
    return require("risk")


@app.get("/report/compliance")
def report_compliance():
    return require("compliance")

# ── Trade approvals ───────────────────────────────────────────────────────────

@app.post("/trades/approve/{ticker}")
def approve_trade(ticker: str, body: ApprovalBody):
    ticker = ticker.upper()
    try:
        approvals_path = DATA_FILES["approvals"]
        if not approvals_path.exists():
            return {"status": "error", "message": "trade_approvals.json not found — run the pipeline first."}

        with open(approvals_path) as f:
            approvals_data = json.load(f)

        if not isinstance(approvals_data, dict) or "approvals" not in approvals_data:
            return {"status": "error", "message": f"trade_approvals.json has unexpected structure: {list(approvals_data.keys()) if isinstance(approvals_data, dict) else type(approvals_data).__name__}"}

        found = False
        for entry in approvals_data["approvals"]:
            if entry.get("ticker", "").upper() == ticker:
                entry["approved"]   = body.approved
                entry["notes"]      = body.notes
                entry["decided_at"] = now_et()
                found = True
                break

        if not found:
            return {"status": "error", "message": f"{ticker} not found in trade_approvals.json. Run the pipeline first."}

        with open(approvals_path, "w") as f:
            json.dump(approvals_data, f, indent=2)

        status   = "approved" if body.approved else "rejected"
        response = {"status": status, "ticker": ticker, "notes": body.notes}

        if body.approved:
            script = str(BASE_DIR / "execution_agent.py")
            subprocess.Popen(
                [PYTHON, script, ticker],
                cwd=str(BASE_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            response["execution"] = "triggered"
            response["message"]   = f"Execution agent started for {ticker}. Check executed_trades.json for confirmation."

        return response

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/trades/executed")
def trades_executed():
    executed = load("executed", default={})
    trades = list(executed.values())
    return {
        "count":  len(trades),
        "trades": trades,
    }

# ── Budget shortcut ──────────────────────────────────────────────────────────

@app.post("/budget")
def set_budget(body: BudgetBody):
    if body.amount < 0:
        raise HTTPException(status_code=400, detail="Budget must be a positive number.")

    # Persist to settings.json
    sett = load_settings()
    sett["trading_budget"] = body.amount
    save_settings(sett)

    # Read current committed capital and immediately update account_state.json
    acct_path = DATA_FILES["account"]
    committed = 0.0
    if acct_path.exists():
        acct = json.loads(acct_path.read_text())
        committed = float(acct.get("committed_capital") or 0)
        acct["manual_budget"]       = body.amount
        acct["remaining_budget"]    = max(0.0, body.amount - committed)
        acct["available_to_invest"] = acct["remaining_budget"]
        with open(acct_path, "w") as f:
            json.dump(acct, f, indent=2)

    remaining = max(0.0, body.amount - committed)
    return {
        "manual_budget":     body.amount,
        "committed_capital": committed,
        "remaining_budget":  remaining,
        "message":           f"Budget updated. Available to invest: ${remaining:,.0f}",
    }

# ── Debug ────────────────────────────────────────────────────────────────────

@app.get("/debug/reports")
def debug_reports():
    risk_path       = DATA_FILES["risk"]
    compliance_path = DATA_FILES["compliance"]

    risk_raw       = json.loads(risk_path.read_text())       if risk_path.exists()       else None
    compliance_raw = json.loads(compliance_path.read_text()) if compliance_path.exists() else None

    # Summarise risk_report structure without sending the full file
    risk_summary = None
    if risk_raw is not None:
        risk_summary = {
            "top_level_keys":    list(risk_raw.keys()),
            "trades_count":      len(risk_raw.get("trades", [])),
            "sample_trade_keys": list(risk_raw["trades"][0].keys()) if risk_raw.get("trades") else [],
            "regime":            risk_raw.get("regime"),
            "account_size":      risk_raw.get("account_size"),
            "full":              risk_raw,          # include full data for diagnosis
        }

    compliance_summary = None
    if compliance_raw is not None:
        compliance_summary = {
            "top_level_keys":      list(compliance_raw.keys()),
            "cleared_count":       len(compliance_raw.get("cleared_trades", [])),
            "trades_count":        len(compliance_raw.get("trades", [])),
            "sample_cleared_keys": list(compliance_raw["cleared_trades"][0].keys())
                                   if compliance_raw.get("cleared_trades") else [],
            "full":                compliance_raw,  # include full data for diagnosis
        }

    return {
        "risk_report_exists":       risk_path.exists(),
        "compliance_report_exists": compliance_path.exists(),
        "risk_report":              risk_summary,
        "compliance_report":        compliance_summary,
    }

# ── Settings ─────────────────────────────────────────────────────────────────

@app.get("/settings")
def get_settings():
    return load_settings()


@app.post("/settings")
def post_settings(body: SettingsBody):
    settings = load_settings()
    if body.trading_budget     is not None:
        settings["trading_budget"]     = body.trading_budget
    if body.max_risk_per_trade is not None:
        settings["max_risk_per_trade"] = body.max_risk_per_trade
    if body.max_open_positions is not None:
        settings["max_open_positions"] = body.max_open_positions
    # risk_agent.py reads settings.json at startup — no file rewrite needed
    return save_settings(settings)

# ── Pipeline triggers ─────────────────────────────────────────────────────────

@app.post("/pipeline/run")
def pipeline_run():
    script = str(BASE_DIR / "run_all.py")
    subprocess.Popen(
        [PYTHON, script],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"status": "started", "message": "Pipeline running in background."}


@app.post("/pipeline/midday")
def pipeline_midday():
    script = str(BASE_DIR / "run_midday.py")
    subprocess.Popen(
        [PYTHON, script],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"status": "started", "message": "Midday pipeline running in background."}

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
