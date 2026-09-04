"""
upstox_fetch.py

Pulls daily historical OHLCV candles from Upstox for a configurable stock
universe, and writes them out in the same wide-DataFrame format used by
synthetic_data.py and real_data_experiment.py -- so causal_discovery.py and
graph_builder.py work on it completely unchanged.

Requires a valid access token -- run upstox_auth.py first (once per day).

Usage:
    python3 upstox_fetch.py
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from upstox_auth import load_saved_token

load_dotenv()

# Map your stock universe to Upstox instrument_keys.
# Look these up from Upstox's instrument master file (linked in their docs) --
# format is "EXCHANGE_SEGMENT|ISIN". These are placeholders you MUST verify;
# ISINs shown here are illustrative and may not be current -- confirm each
# one against the actual instrument master before relying on this.
INSTRUMENT_KEYS = {
    "RELIANCE":  "NSE_EQ|INE002A01018",
    "TCS":       "NSE_EQ|INE467B01029",
    "HDFCBANK":  "NSE_EQ|INE040A01034",
    "INFY":      "NSE_EQ|INE009A01021",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "SBIN":      "NSE_EQ|INE062A01020",
    "HCLTECH":   "NSE_EQ|INE860A01027",
    "WIPRO":     "NSE_EQ|INE075A01022",
    "AXISBANK":  "NSE_EQ|INE238A01034",
}

BASE_URL = "https://api.upstox.com/v2/historical-candle"


def fetch_daily_candles(instrument_key: str, from_date: str, to_date: str, token: str) -> pd.DataFrame:
    """Fetch daily OHLCV candles for one instrument between two dates (inclusive)."""
    url = f"{BASE_URL}/{instrument_key}/day/{to_date}/{from_date}"
    resp = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=20,
    )
    if resp.status_code == 401:
        raise RuntimeError("Access token expired/invalid -- re-run upstox_auth.py to get a fresh one.")
    resp.raise_for_status()
    data = resp.json()

    candles = data.get("data", {}).get("candles", [])
    # Each candle: [timestamp, open, high, low, close, volume, open_interest]
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df[["date", "close"]].sort_values("date")


def build_wide_dataframe(universe: dict, from_date: str, to_date: str, token: str) -> pd.DataFrame:
    """Fetch every stock and merge into one wide DataFrame: one row per date,
    one column per stock's closing price -- matching real_data_experiment.py's
    load_returns() input shape before the log-return step."""
    frames = {}
    for ticker, ikey in universe.items():
        print(f"Fetching {ticker} ({ikey})...")
        try:
            df = fetch_daily_candles(ikey, from_date, to_date, token)
            frames[ticker] = df.set_index("date")["close"]
        except Exception as e:
            print(f"  FAILED for {ticker}: {e}")
        time.sleep(0.3)  # be polite to the rate limit

    wide = pd.DataFrame(frames)
    wide.index.name = "date"
    return wide


if __name__ == "__main__":
    token = load_saved_token()
    if not token:
        raise SystemExit("No saved token found. Run 'python3 upstox_auth.py' first.")

    # Example: last 30 days. For a full backfill, page through in <1-year
    # chunks -- Upstox's historical-candle endpoint limits how far back a
    # single request can span; check current limits in their docs before
    # requesting multi-year history in one call.
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=30)).isoformat()

    wide_df = build_wide_dataframe(INSTRUMENT_KEYS, from_date, to_date, token)

    out_path = "real_market_data_upstox.csv"
    wide_df.to_csv(out_path)
    print(f"\nSaved {wide_df.shape[0]} rows x {wide_df.shape[1]} stocks -> {out_path}")
    print("This file has the same shape as synthetic_data.py's output --")
    print("feed it into causal_discovery.py / real_data_experiment.py unchanged.")
