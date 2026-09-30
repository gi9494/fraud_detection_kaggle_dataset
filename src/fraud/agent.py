"""Investigator agent: explains in plain words why the model flagged a purchase.

The agent has two tools (plain Python: they only read the data and the model, they never invent anything):
  - purchase_facts(transaction_id): the readable facts of the purchase, its score and the threshold
  - score_drivers(transaction_id):  which features push the score up. For each feature: what would the
    score be if that feature had the value of a typical honest purchase? A big drop = a big reason.

The LLM (local, through Ollama and LangChain) decides which tools to call, then writes a short note.
At the end every number in the note is checked against the tool results: a number that is not there
is reported as "not verified" (the LLM must not invent numbers).

Usage (from the project folder, with Ollama running):
    python src/fraud/agent.py --n 3            # the 3 most suspicious purchases of the test month
    python src/fraud/agent.py --id <TransactionID>   # one purchase
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).resolve().parents[2] / ".env")   # only the .env of this project
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def chat_model() -> ChatOllama:
    """The local LLM. trust_env=False: Ollama is on this computer, so the connection must not go through
    a company proxy set in the environment variables."""
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0,
                      client_kwargs={"trust_env": False})


def check_ollama() -> None:
    """Stop with a clear message if Ollama is not running or the model is not downloaded."""
    no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        tags = json.load(no_proxy.open(f"{OLLAMA_BASE_URL}/api/tags", timeout=5))
    except OSError:
        raise SystemExit(f"Ollama is not reachable at {OLLAMA_BASE_URL}: open the Ollama app (or run `ollama serve`).")
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n == LLM_MODEL or n.split(":")[0] == LLM_MODEL for n in names):
        raise SystemExit(f"Model {LLM_MODEL} not downloaded. Run: ollama pull {LLM_MODEL}  (you have: {names})")

# ------------------------------------------------------------------ model and data (made by notebook 05)
INFO = json.load(open("models/model_info.json"))
FEATURES, CATEGORIES, THRESHOLD = INFO["features"], INFO["categories"], INFO["threshold"]
MODEL = joblib.load("models/fraud_model.joblib")
TEST = pd.read_parquet("data/processed/test_scores.parquet").set_index("TransactionID")

# the value of a typical HONEST purchase for every feature (from the train months): used by score_drivers
_full = pd.read_parquet("data/processed/features.parquet", columns=FEATURES + ["split", "isFraud"])
_honest = _full[(_full["split"] == "train") & (_full["isFraud"] == 0)]
TYPICAL = {c: (_honest[c].mode().iloc[0] if c in CATEGORIES else _honest[c].median()) for c in FEATURES}
del _full, _honest

READABLE = {
    "TransactionAmt": "amount ($)", "amt_log": "amount (log scale)", "hour": "hour of the day",
    "weekday": "day of the week", "ProductCD": "product type", "card4": "card network",
    "card6": "card type (debit/credit)", "card1_freq": "how common the card code is",
    "foreign_card": "card from a foreign country", "foreign_address": "billing address in a foreign country",
    "addr1_missing": "billing address missing", "dist1": "distance (billing to shipping)",
    "dist2_present": "second distance present", "R_email_present": "receiver email present",
    "P_email": "buyer email provider", "R_email": "receiver email provider", "same_email": "buyer and receiver same email",
    "amt_odd_cents": "amount with more than 2 decimals (currency conversion)",
    "amt_vs_usual": "amount compared to the usual amount of this card code",
    "has_identity": "device information available", "device": "device brand", "browser": "browser",
    "system": "operating system", "DeviceType": "device type (mobile/desktop)",
    "n_missing": "number of empty fields", "D1": "days since the card was first seen",
}
KEY_FACTS = ["TransactionAmt", "hour", "ProductCD", "card4", "card6", "foreign_card", "foreign_address",
             "addr1_missing", "P_email", "R_email", "device", "browser", "amt_odd_cents", "D1"]


def readable(col: str) -> str:
    return READABLE.get(col, f"{col} (anonymous feature)")


def fmt(value):
    """A value ready to be read: rounded numbers, text for categories, 'empty' for missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "empty"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 2)
    return str(value)


# ------------------------------------------------------------------ tools
@tool
def purchase_facts(transaction_id: int) -> str:
    """Readable facts about one purchase (amount, hour, card, email, device), its fraud score and the threshold."""
    row = TEST.loc[transaction_id]
    facts = {readable(c): fmt(row[c]) for c in KEY_FACTS if c in row}
    facts.update({"model score": round(float(row["score"]), 3), "threshold": THRESHOLD,
                  "flagged": bool(row["score"] >= THRESHOLD)})
    return json.dumps(facts)


@tool
def score_drivers(transaction_id: int) -> str:
    """The features that push the fraud score of one purchase up. For each: its value, the value of a typical
    honest purchase, and how much the score drops if the feature had that typical value."""
    row = TEST.loc[[transaction_id], FEATURES]
    base = float(MODEL.predict_proba(row)[:, 1][0])
    usable = [c for c in FEATURES if not (c not in CATEGORIES and pd.isna(TYPICAL[c]))]
    variants = pd.concat([row] * len(usable), ignore_index=True)
    for i, c in enumerate(usable):                       # one copy per feature, with that feature "made typical"
        variants.at[i, c] = TYPICAL[c]
    drops = base - MODEL.predict_proba(variants)[:, 1]
    top = sorted(zip(usable, drops), key=lambda x: -x[1])[:5]
    return json.dumps({
        "score": round(base, 3),
        "drivers": [{"feature": readable(c), "value": fmt(row.iloc[0][c]), "typical honest value": fmt(TYPICAL[c]),
                     "score if typical": round(base - d, 3), "score drop": round(float(d), 3)}
                    for c, d in top if d > 0.005],
    })


TOOLS = {t.name: t for t in (purchase_facts, score_drivers)}

SYSTEM = """You help a fraud analyst. You explain why the fraud model flagged a purchase.
First call the tools to get the facts: never guess. Then write a short note in English:
1. One line: the model score and the threshold.
2. "Why it looks suspicious": 3-4 bullets from score_drivers, each with the value and the score drop.
3. "Points against fraud": 1-2 bullets if some facts look normal, otherwise "none found".
4. "Next step": one concrete check (for example: contact the customer about this amount).
Use only numbers that appear in the tool results. The model can be wrong: never say that the purchase IS a fraud."""


# ------------------------------------------------------------------ agent loop
def investigate(transaction_id: int) -> tuple[str, list[str]]:
    """Run the agent on one purchase. Returns the note and the tool results it used."""
    llm = chat_model().bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(SYSTEM), HumanMessage(f"Explain why purchase {transaction_id} was flagged.")]
    results: list[str] = []
    reply = None
    for _ in range(6):                                   # the LLM asks for tools, we run them, we give back the results
        reply = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            output = TOOLS[call["name"]].invoke(call["args"])
            results.append(output)
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))
    if not results:                                      # small models sometimes skip the tools: we call them ourselves
        results = [TOOLS[name].invoke({"transaction_id": transaction_id}) for name in TOOLS]
        messages.append(HumanMessage("Tool results:\n" + "\n".join(results) + "\nNow write the note."))
        reply = chat_model().invoke(messages)
    return reply.content, results


def unverified_numbers(note: str, results: list[str], transaction_id: int) -> list[str]:
    """Numbers in the note that do not appear in the tool results (also accepted: a score written as a %)."""
    known = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", " ".join(results))] + [float(transaction_id)]
    bad = []
    for text in re.findall(r"-?\d+(?:[.,]\d+)?", note):
        n = float(text.replace(",", ""))
        if n.is_integer() and 0 <= n <= 5:              # list numbers "1.", "2." ...
            continue
        if not any(abs(n - k) < 0.006 or abs(n - 100 * k) < 0.6 for k in known):
            bad.append(text)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, help="one TransactionID of the test month")
    ap.add_argument("--n", type=int, default=3, help="how many of the most suspicious purchases to explain")
    args = ap.parse_args()
    check_ollama()
    ids = [args.id] if args.id else list(TEST.sort_values("score", ascending=False).index[:args.n])

    for tid in ids:
        note, results = investigate(int(tid))
        bad = unverified_numbers(note, results, int(tid))
        print(f"\n{'=' * 80}\npurchase {tid}\n{'=' * 80}\n{note}\n")
        print("numbers check:", "all found in the tool results ✔" if not bad else f"NOT VERIFIED: {bad}")


if __name__ == "__main__":
    main()