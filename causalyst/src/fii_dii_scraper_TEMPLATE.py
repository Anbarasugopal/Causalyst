"""Real FII/DII scraper using the live NSE endpoint.

This is a daily collector pattern that appends to a local SQLite store and is
idempotent on reruns. The paper's institutional-flow contribution is live-only
forward history, not a full backfill, and it must be reported as such.
"""

import sqlite3
from datetime import date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "causalyst" / "data" / "fii_dii_history.sqlite"
NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def fetch_today() -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=20)
    response = session.get(NSE_FII_DII_URL, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return [
        {
            "date": item.get("date"),
            "category": item.get("category"),
            "buy_value": float(item.get("buyValue", 0) or 0.0),
            "sell_value": float(item.get("sellValue", 0) or 0.0),
            "net_value": float(item.get("netValue", 0) or 0.0),
        }
        for item in payload
    ]


def append_to_history(records: list[dict]):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fii_dii_daily (date TEXT, category TEXT, buy_value REAL, sell_value REAL, net_value REAL, UNIQUE(date, category))"
    )
    conn.executemany(
        "INSERT OR IGNORE INTO fii_dii_daily(date, category, buy_value, sell_value, net_value) VALUES (?, ?, ?, ?, ?)",
        [(row["date"], row["category"], row["buy_value"], row["sell_value"], row["net_value"]) for row in records],
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    rows = fetch_today()
    append_to_history(rows)
    print(f"Stored {len(rows)} FII/DII rows for {date.today().isoformat()} to {DB_PATH}")
