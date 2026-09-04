"""
upstox_auth.py

Handles Upstox's OAuth2 login flow. Upstox access tokens expire daily, so
this needs to be run once per trading day before upstox_fetch.py will work.

Setup (do this once):
    1. Create an app at https://upstox.com/developer -> get API_KEY, API_SECRET
    2. Set the app's Redirect URI to http://localhost:8080/callback
    3. Put both in a .env file (see .env.example) -- never hardcode them

Daily flow:
    1. This script prints a login URL
    2. Open it in a browser, log in to Upstox, approve access
    3. You'll be redirected to your redirect_uri with ?code=XXXXX in the URL
    4. Paste that code back into this script when prompted
    5. It exchanges the code for an access_token and saves it to token.json

This is the standard OAuth "authorization code" flow -- there is no way to
skip the manual browser step within Upstox's normal ToS; fully automating
it requires their separate "Access Token Request API" with manual mobile
approval, which is a further step beyond this script.
"""

import json
import os
import webbrowser
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("UPSTOX_API_KEY")
API_SECRET = os.environ.get("UPSTOX_API_SECRET")
REDIRECT_URI = os.environ.get("UPSTOX_REDIRECT_URI", "http://localhost:8080/callback")

TOKEN_FILE = Path(__file__).parent / "token.json"

LOGIN_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def get_login_url() -> str:
    return f"{LOGIN_URL}?client_id={API_KEY}&redirect_uri={REDIRECT_URI}&response_type=code"


def exchange_code_for_token(auth_code: str) -> dict:
    resp = requests.post(
        TOKEN_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": auth_code,
            "client_id": API_KEY,
            "client_secret": API_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    if not resp.ok:
        print(f"\n--- Upstox token exchange failed (HTTP {resp.status_code}) ---")
        print(f"Response body: {resp.text}")
        print("---")
        resp.raise_for_status()
    token_data = resp.json()
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
    return token_data


def load_saved_token() -> Optional[str]:
    """Returns today's saved access token if one exists, else None.
    (Doesn't check expiry precisely -- Upstox tokens expire ~3:30am IST daily,
    so if a call fails with 401, just re-run this script.)"""
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        return data.get("access_token")
    return None


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        raise SystemExit("Set UPSTOX_API_KEY and UPSTOX_API_SECRET in your .env file first.")

    url = get_login_url()
    print(f"1. Opening login URL in your browser:\n   {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print("2. Log in to Upstox, approve access.")
    print("3. You'll land on your redirect URI with ?code=XXXXX in the address bar.")
    auth_code = input("4. Paste the 'code' value here: ").strip()

    token_data = exchange_code_for_token(auth_code)
    print(f"\nSaved access token to {TOKEN_FILE}. Valid for today's trading session.")
