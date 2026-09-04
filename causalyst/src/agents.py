"""
agents.py

Stage 4: multi-agent debate and synthesis.

This version keeps the original structure but adds a real local evidence layer:
- live RSS sentiment via the local FinBERT model (ProsusAI/finbert)
- live FII/DII evidence from the NSE scraper and local daily history store
- graceful fallback behavior when Anthropic is not configured

The GNN is not retrained from this evidence; it is only used to enrich the
explainability layer at inference time.
"""

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import feedparser
import requests
from transformers import pipeline

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FII_DII_DB = PROJECT_ROOT / "causalyst" / "data" / "fii_dii_history.sqlite"


@dataclass
class AgentResult:
   role: str
   stance: str
   confidence: float
   reasoning: str
   evidence_used: List[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _load_finbert_pipeline():
   return pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", device=-1)


def _ticker_regex(ticker: str):
   t = re.escape(ticker.upper())
   return re.compile(rf"\b{t}\b")


def _moneycontrol_rss() -> List[Dict[str, str]]:
   urls = [
       "https://www.moneycontrol.com/rss/MCtopnews.xml",
       "https://www.moneycontrol.com/rss/MCmarkets.xml",
       "https://www.moneycontrol.com/rss/MCindustry.xml",
   ]
   entries = []
   for url in urls:
       try:
           parsed = feedparser.parse(url)
           for item in parsed.entries[:20]:
               title = (getattr(item, "title", "") or "").strip()
               summary = (getattr(item, "summary", "") or "").strip()
               if title or summary:
                   entries.append({"title": title, "summary": summary, "link": getattr(item, "link", "")})
       except Exception:
           continue
   return entries


def live_sentiment_evidence_for_ticker(ticker: str, max_articles: int = 8) -> Dict[str, object]:
   ticker_norm = ticker.upper()
   entries = _moneycontrol_rss()
   relevant = []
   for entry in entries:
       text = f"{entry['title']} {entry['summary']}".upper()
       if ticker_norm in text or any(token in text for token in ["NIFTY", "SENSEX", "MARKET", "STOCK", "BANKING", "IT", "AUTO", "ENERGY"]):
           relevant.append(entry)
   if not relevant:
       return {
           "label": "neutral",
           "sentiment_score": 0.0,
           "confidence": 0.2,
           "evidence": ["No directly relevant live market RSS coverage was found for this ticker in the last RSS batch."],
           "source": "moneycontrol_rss",
       }

   scored = []
   model = _load_finbert_pipeline()
   for entry in relevant[:max_articles]:
       text = f"{entry['title']} {entry['summary']}"
       try:
           result = model(text, truncation=True, max_length=512)[0]
           label = result["label"]
           score = float(result["score"])
           scored.append({"title": entry["title"], "label": label, "score": score})
       except Exception:
           continue

   if not scored:
       return {
           "label": "neutral",
           "sentiment_score": 0.0,
           "confidence": 0.2,
           "evidence": ["RSS articles were fetched but the local FinBERT model could not score them in this environment."],
           "source": "moneycontrol_rss",
       }

   weights = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
   sentiment_score = sum(weights.get(item["label"].lower(), 0.0) * item["score"] for item in scored) / max(len(scored), 1)
   avg_conf = sum(item["score"] for item in scored) / len(scored)
   label = "bullish" if sentiment_score > 0.15 else "bearish" if sentiment_score < -0.15 else "neutral"
   evidence = [f"{item['title']} -> {item['label']} ({item['score']:.3f})" for item in scored[:5]]
   return {
       "label": label,
       "sentiment_score": round(float(sentiment_score), 4),
       "confidence": round(float(min(0.95, max(0.2, avg_conf))), 4),
       "evidence": evidence,
       "source": "moneycontrol_rss + finbert",
   }


def _nse_fii_dii_json() -> List[Dict[str, str]]:
   headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
   session = requests.Session()
   session.get("https://www.nseindia.com", headers=headers, timeout=20)
   response = session.get("https://www.nseindia.com/api/fiidiiTradeReact", headers=headers, timeout=20)
   response.raise_for_status()
   return response.json()


def get_recent_fii_dii_snapshot() -> Dict[str, object]:
   FII_DII_DB.parent.mkdir(parents=True, exist_ok=True)
   if not FII_DII_DB.exists():
       data = _nse_fii_dii_json()
       rows = []
       for item in data:
           rows.append({
               "date": item.get("date"),
               "category": item.get("category"),
               "buy_value": item.get("buyValue"),
               "sell_value": item.get("sellValue"),
               "net_value": item.get("netValue"),
           })
       import sqlite3
       con = sqlite3.connect(FII_DII_DB)
       con.execute("CREATE TABLE IF NOT EXISTS fii_dii_daily (date TEXT, category TEXT, buy_value REAL, sell_value REAL, net_value REAL, UNIQUE(date, category))")
       con.executemany("INSERT OR IGNORE INTO fii_dii_daily(date, category, buy_value, sell_value, net_value) VALUES (?, ?, ?, ?, ?)", [(r["date"], r["category"], float(r["buy_value"]), float(r["sell_value"]), float(r["net_value"])) for r in rows])
       con.commit()
       con.close()

   import sqlite3
   con = sqlite3.connect(FII_DII_DB)
   rows = con.execute("SELECT date, category, buy_value, sell_value, net_value FROM fii_dii_daily ORDER BY date DESC, category ASC").fetchall()
   con.close()
   records = [{"date": d, "category": c, "buy_value": b, "sell_value": s, "net_value": n} for d, c, b, s, n in rows]
   if not records:
       return {"label": "neutral", "confidence": 0.2, "evidence": ["No FII/DII record was found in the local daily store."], "source": "nse_fii_dii_store"}

   latest = records[0]
   latest_date = latest["date"]
   total_net = sum(float(r["net_value"]) for r in records if r["date"] == latest_date)
   evidence = [f"{r['category']} net {r['net_value']} Cr on {r['date']}" for r in records if r["date"] == latest_date]
   label = "bullish" if total_net > 0 else "bearish" if total_net < 0 else "neutral"
   return {
       "label": label,
       "confidence": 0.65 if abs(total_net) > 100 else 0.45,
       "evidence": evidence,
       "source": "nse_fii_dii_store",
       "latest_date": latest_date,
       "total_net_cr": round(float(total_net), 2),
   }


def build_live_evidence_for_ticker(ticker: str, sector: str = None) -> Dict[str, object]:
   sentiment = live_sentiment_evidence_for_ticker(ticker)
   fii_dii = get_recent_fii_dii_snapshot()
   sector_value = sector or "Unknown"
   return {
       "sentiment": sentiment,
       "institutional_flow": fii_dii,
       "sector": sector_value,
       "source_summary": {
           "sentiment_source": sentiment.get("source"),
           "institutional_flow_source": fii_dii.get("source"),
           "sector_label": sector_value,
       },
   }


def _call_claude(system_prompt: str, user_prompt: str) -> str:
   api_key = os.environ.get("ANTHROPIC_API_KEY")
   if not api_key:
       raise RuntimeError("Set ANTHROPIC_API_KEY in your environment before running agents.py")
   resp = requests.post(
       API_URL,
       headers={
           "x-api-key": api_key,
           "anthropic-version": "2023-06-01",
           "content-type": "application/json",
       },
       json={
           "model": MODEL,
           "max_tokens": 500,
           "system": system_prompt,
           "messages": [{"role": "user", "content": user_prompt}],
       },
       timeout=60,
   )
   resp.raise_for_status()
   data = resp.json()
   return "".join(block["text"] for block in data["content"] if block["type"] == "text")


AGENT_PROMPTS = {
   "fundamentals": (
       "You are the Fundamentals analyst on an equity research panel. You only consider company-specific evidence: earnings, guidance, corporate actions, credit ratings. Ignore flow, sentiment, and macro data even if given. Respond ONLY with JSON: {stance, confidence (0-1), reasoning (<80 words)}."
   ),
   "sentiment": (
       "You are the Sentiment analyst. You only consider news and social sentiment evidence. State plainly if the causal evidence for sentiment is weak. Respond ONLY with JSON: {stance, confidence (0-1), reasoning (<80 words)}."
   ),
   "macro": (
       "You are the Macro analyst. You only consider interest rates, inflation, crude oil, currency, and global cues. Respond ONLY with JSON: {stance, confidence (0-1), reasoning (<80 words)}."
   ),
   "institutional_flow": (
       "You are the Institutional Flow analyst -- your ENTIRE job is to argue whether this move is driven by FII/DII flow, index rebalancing, or block deals, as a distinct hypothesis competing with fundamentals-driven moves. Respond ONLY with JSON: {stance, confidence (0-1), reasoning (<80 words)}."
   ),
   "contrarian": (
       "You are the Contrarian. You will be shown the other four analysts' conclusions. Your job is to find the WEAKEST link in their reasoning and argue the opposite conclusion, or argue their shared confidence is unjustified given how thin the causal evidence actually is. You must identify at least one concrete flaw -- do not simply agree because the majority agrees. Respond ONLY with JSON: {stance, confidence (0-1), reasoning (<80 words)}."
   ),
}

SYNTHESIZER_PROMPT = (
   "You are the Synthesizer. You will be given four specialist opinions and one contrarian challenge. Produce a final judgment. CRITICAL RULE: if the specialists disagree with each other or the contrarian raises a concrete, unaddressed flaw, your confidence must be LOWER than the average specialist confidence, not higher. Do not simply average opinions -- weigh the strength of evidence each cites. Respond ONLY with JSON: {final_stance, confidence (0-1), plain_english_explanation (<150 words), key_disagreement (<40 words or null)}."
)


def run_debate(evidence: Dict[str, List[str]]) -> Dict:
   transcript = {}
   for role in ["fundamentals", "sentiment", "macro", "institutional_flow"]:
       ev = evidence.get(role, ["No significant causal evidence found for this agent's domain."])
       user_prompt = "Evidence:\n" + "\n".join(f"- {e}" for e in ev)
       raw = _call_claude(AGENT_PROMPTS[role], user_prompt)
       transcript[role] = raw

   contrarian_input = "Other analysts concluded:\n" + "\n".join(
       f"{role}: {result}" for role, result in transcript.items()
   )
   transcript["contrarian"] = _call_claude(AGENT_PROMPTS["contrarian"], contrarian_input)

   synth_input = "Full debate:\n" + "\n".join(f"{role}: {result}" for role, result in transcript.items())
   transcript["synthesis"] = _call_claude(SYNTHESIZER_PROMPT, synth_input)
   return transcript


if __name__ == "__main__":
   example = build_live_evidence_for_ticker("RELIANCE")
   print(json.dumps(example, indent=2))
