
"""
collect_data.py — FIXED VERSION
Collects ~300 geopolitical headlines from Guardian, Reuters (cc_news), and Finnhub.
Saves to data/geo_candidates.csv for manual labeling.

Usage:
    python collect_data.py --guardian_key YOUR_KEY --finnhub_key YOUR_KEY
"""

import requests
import pandas as pd
import argparse
import time
import os
import re
from rapidfuzz import fuzz

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--guardian_key", required=True)
parser.add_argument("--finnhub_key",  required=False)
parser.add_argument("--out", default="data/geo_candidates.csv")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)

# ── Keywords ──────────────────────────────────────────────────────────────────
GUARDIAN_KEYWORDS = (
    # Sanctions
    "sanctions OR \"economic sanctions\" OR \"financial sanctions\" "
    "OR \"asset freeze\" OR \"oil sanctions\" OR \"arms embargo\" "
    "OR \"secondary sanctions\" "
    # Trade
    "OR tariffs OR \"trade war\" OR \"import duties\" OR \"trade dispute\" "
    "OR \"WTO ruling\" OR \"trade agreement\" OR \"trade restrictions\" "
    "OR \"market access\" "
    # Technology
    "OR \"export controls\" OR \"chip ban\" OR \"chip war\" "
    "OR \"CHIPS Act\" OR \"technology transfer\" OR \"tech decoupling\" "
    "OR \"critical technology\" OR \"dual use\" "
    # Supply chain
    "OR \"supply chain\" OR \"friend-shoring\" OR reshoring "
    "OR nearshoring OR decoupling OR \"industrial policy\" "
    "OR \"supply chain resilience\" OR \"critical goods\" "
    "OR \"strategic reserves\" "
    # Energy
    "OR \"energy security\" OR \"gas pipeline\" OR \"energy sanctions\" "
    "OR \"grain corridor\" OR \"food security\" "
    "OR \"critical minerals\" OR \"rare earth\" "
    # Coercion / autonomy
    "OR \"economic coercion\" OR \"strategic autonomy\" "
    "OR \"weaponised interdependence\" OR \"economic statecraft\" "
    "OR \"geopolitical risk\" OR \"economic security\" "
    # COVID
    "OR \"vaccine nationalism\" OR \"PPE export\" "
    "OR \"pandemic supply chain\" OR \"medical supplies\" "
    # Regional architecture
    "OR \"US-China trade\" OR \"transatlantic trade\" "
    "OR \"AUKUS\" OR \"QUAD\" OR \"Belt and Road\" "
    "OR \"defence procurement\" OR \"security partnership\""
)

# Supplementary pull for underrepresented themes
SUPPLEMENTARY_KEYWORDS = (
    "\"supply chain resilience\" OR \"friend-shoring\" OR nearshoring "
    "OR reshoring OR \"industrial policy\" OR \"strategic autonomy\" "
    "OR \"economic statecraft\" OR \"chip war\" OR \"CHIPS Act\" "
    "OR \"semiconductor export\" OR \"energy security\" "
    "OR \"critical minerals\" OR \"rare earth\""
)

# Regex for Reuters filtering (defined before collection)
KW_PATTERN = re.compile(
    r"sanction|tariff|export.control|decoupling|friend.shor|"
    r"supply.chain|economic.coercion|strategic.autonom|trade.war|"
    r"industrial.policy|chip.ban|chip.war|CHIPS|rare.earth|"
    r"critical.mineral|energy.security|nearshoring|reshoring|"
    r"geopolitical|economic.security|vaccine.national|Belt.and.Road|"
    r"AUKUS|QUAD|energy.sanction|grain.corridor|food.security|arms.embargo|"
    r"asset.freeze|technology.transfer|defence.procurement|"
    r"weaponised.interdependence|economic.statecraft|dual.use|"
    r"strategic.reserve|secondary.sanction|medical.suppli|PPE",
    re.IGNORECASE
)

# Drop patterns for explainers and non-financial content
DROP_PATTERN = re.compile(
    r"explained in \d+|"
    r"^What (is|are) .{0,50}\?$|"
    r"war briefing:.{0,80}(drone|missile|troops|attack|"
    r"frontline|evacuat|civilian|soldier|bomb|shelling)",
    re.IGNORECASE
)

# Theme classification (defined before use in step 6)
THEME_PATTERNS = {
    'tariffs_trade':     r'tariff|trade.war|import.duti|WTO|trade.deal|trade.dispute',
    'sanctions':         r'sanction|asset.freeze|arms.embargo|secondary.sanction',
    'ukraine_russia':    r'ukraine|russia|putin|zelenskyy|nord.stream',
    'us_china':          r'china|taiwan|huawei|tiktok|sino|us.china',
    'supply_chain':      r'supply.chain|reshoring|nearshoring|friend.shor|decoupling',
    'semiconductors':    r'chip|semiconductor|CHIPS|export.control|dual.use',
    'energy':            r'energy.security|gas.pipeline|energy.sanction|oil.sanction',
    'critical_minerals': r'rare.earth|critical.mineral|strategic.reserve',
    'covid_security':    r'vaccine|COVID|pandemic|PPE|medical.suppli',
    'regional_arch':     r'AUKUS|QUAD|Belt.and.Road|NATO|G7|transatlantic',
    'industrial_policy': r'industrial.policy|strategic.autonom|economic.statecraft|'
                         r'weaponised.interdependence|economic.coercion',
}

def assign_theme(headline):
    """Assign first matching theme. Returns 'other' if no match."""
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, str(headline), re.IGNORECASE):
            return theme
    return 'other'

headlines = []  # collector list — defined before collection steps

# ── 1. GUARDIAN ───────────────────────────────────────────────────────────────
print("Fetching Guardian...")
for page in range(1, 7):          # 6 pages x 50 = up to 300 raw candidates
    r = requests.get(
        "https://content.guardianapis.com/search",
        params={
            "q":           KEYWORDS,
            "section":     "world|business|politics|environment",
            "from-date":   "2016-02-01",
            "to-date":     "2026-02-01",
            "page-size":   50,
            "page":        page,
            "show-fields": "headline",
            "api-key":     args.guardian_key,
        }
    ).json()

    results = r.get("response", {}).get("results", [])
    if not results:
        print(f"  Guardian: no results on page {page}, stopping.")
        break

    for item in results:
        headline = item.get("fields", {}).get("headline") or item.get("webTitle", "")
        headlines.append({
            "headline": headline.strip(),
            "source":   "guardian",
            "date":     item.get("webPublicationDate", "")[:10],
        })
    time.sleep(0.5)

print(f"  Guardian raw: {sum(1 for h in headlines if h['source']=='guardian')}")

# ── 2. REUTERS via cc_news ────────────────────────────────────────────────────
print("Fetching Reuters via cc_news (this may take ~2 min)...")
from datasets import load_dataset

cc = load_dataset("cc_news", split="train", streaming=True)
reuters_hits = []
scanned = 0

for row in cc:
    scanned += 1
    if scanned % 50000 == 0:
        print(f"  Scanned {scanned} rows, found {len(reuters_hits)} Reuters hits...")
    if "reuters.com" not in row.get("domain", ""):
        continue
    title = row.get("title", "")
    if title and KW_PATTERN.search(title):
        reuters_hits.append({
            "headline": title.strip(),
            "source":   "reuters",
            "date":     str(row.get("publish_date", ""))[:10],
        })
    if len(reuters_hits) >= 150:   # collect 150 raw to survive dedup at 100
        break

headlines.extend(reuters_hits)
print(f"  Reuters raw: {len(reuters_hits)}")

# ── 4. DEDUPLICATE ────────────────────────────────────────────────────────────
print("Deduplicating...")
df = pd.DataFrame(headlines).dropna(subset=["headline"])
df = df[df["headline"].str.len() > 20].copy()
df = df.drop_duplicates(subset=["headline"])   # exact dedup first

keep = []
seen = []
for _, row in df.iterrows():
    if all(fuzz.ratio(row["headline"], s) < 85 for s in seen):
        keep.append(row)
        seen.append(row["headline"])

df_clean = pd.DataFrame(keep).reset_index(drop=True)
print(f"  After dedup: {len(df_clean)} headlines")
print(df_clean["source"].value_counts())

# ── 5. BALANCED SAMPLE ───────────────────────────────────────────────────────
# 100 per source where possible, random_state=42 for reproducibility
sampled = (
    df_clean.groupby("source", group_keys=False)
    .apply(lambda x: x.sample(n=min(250, len(x)), random_state=42),
           include_groups=False)
    .reset_index(drop=True)
)

# Restore source column if dropped
if "source" not in sampled.columns:
    sampled = (
        df_clean.groupby("source")
        .apply(lambda x: x.sample(n=min(250, len(x)), random_state=42))
        .reset_index(drop=True)
    )

sampled["label"] = ""   # fill manually: positive / negative / neutral

sampled.to_csv(args.out, index=False)

print(f"\nSaved {len(sampled)} headlines to {args.out}")
print(sampled["source"].value_counts())
print("\nFirst 5 rows:")
print(sampled[["headline", "source", "date"]].head())



 
