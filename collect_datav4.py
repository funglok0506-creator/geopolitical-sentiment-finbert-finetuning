"""
collect_data.py — DEFINITIVE VERSION
Collects 300 high-quality geopolitical headlines: 150 Guardian and 150 Reuters

Filters enforce:
  1. Macro-level events only (no firm-level, no domestic opinion)
  2. Major power / institution reference required
  3 US-centric framing (events with clear US market perspective)
  4.Temporal balance: 2018-2021 | 2022-2023 | 2024-2026
  5. Thematic balance across 10 securitisation clusters
  6. Zero duplicates (enforced via exact & fuzzy)

Usage:
    python collect_data.py --guardian_key YOUR_KEY
"""

import requests
import pandas as pd
import argparse
import time
import os
import re
from rapidfuzz import fuzz
from datasets import load_dataset

# CLI ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--guardian_key", required = True)
parser.add_argument("--out", default="data/geo_candidates.csv")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)

#  Keyword definitions

# Guardian API query: broad  for volume, refined by filters below
GUARDIAN_KEYWORDS = (
    '"sanctions" OR "trade war" OR "tariffs" OR "export controls" '
    'OR "chip ban" OR "chip war" OR "CHIPS Act" OR "technology transfer" '
    'OR "supply chain" OR "friend-shoring" OR "reshoring" OR "nearshoring" '
    'OR "industrial policy" OR "strategic autonomy" OR "economic coercion" '
    'OR "weaponised interdependence" OR "economic statecraft" '
    'OR "energy security" OR "critical minerals" OR "rare earth" '
    'OR "vaccine nationalism" OR "pandemic supply chain" '
    'OR "Belt and Road" OR "AUKUS" OR "QUAD alliance" '
    'OR "US-China trade" OR "transatlantic trade" '
    'OR "defence procurement" OR "arms embargo" OR "asset freeze"'
)

# Reuters via cc_news API: precise word-boundary patterns with no false positives
REUTERS_PATTERN = re.compile(
    r'\bsanction(s|ed|ing)?\b(?!.*misconduct)(?!.*employee)|'  # sanctions but NOT corporate misconduct
    r'\btariff(s)?\b|'
    r'export\s+control(s)?\b|'
    r'trade\s+war\b|'
    r'chip\s+(ban|war)\b|'                    # chip ban/chip war, NOT blue chip
    r'\bCHIPS\s+Act\b|'
    r'rare\s+earth(s)?\b|'
    r'critical\s+mineral(s)?\b|'
    r'energy\s+security\b|'
    r'friend.shor(ing)?\b|'
    r'\bnearshoring\b|\breshoring\b|'
    r'industrial\s+policy\b|'
    r'strategic\s+autonom\w+\b|'
    r'economic\s+coercion\b|'
    r'weaponi[sz]ed\s+interdependence\b|'
    r'economic\s+statecraft\b|'
    r'vaccine\s+nationalism\b|'
    r'pandemic\s+supply\s+chain\b|'
    r'\bAUKUS\b|'
    r'\bQUAD\b(?!\w)|'                        # QUAD, not quadruple
    r'Belt\s+and\s+Road\b|'
    r'arms\s+embargo\b|'
    r'asset\s+freeze\b|'
    r'technology\s+transfer\b|'
    r'defence\s+procurement\b|'
    r'secondary\s+sanction(s)?\b|'
    r'dual.use\s+(technology|goods|items)\b',
    re.IGNORECASE
)

# SECTION 2: Quality filters

# Drop filters: Remove noise, non-news, sports, military-only content
DROP_PATTERN = re.compile(
    # Explainers and reader engagement
    r'explained\s+in\s+\d+\s+(second|minute)|'
    r'^(What|Who|How|Why)\s+(is|are|does|did|was)\s+.{0,60}\?$|'
    r'Share\s+how\s+.{0,40}(affect|impact)|'
    r'(quiz|crossword|puzzle|nerdiest|take\s+our)\b|'
    
    # Wire brief formats (low editorial quality)
    r'^\s*BRIEF-|^\s*RPT-EXPLAINER|^\s*RPT-UPDATE\s+\d+|'
    r'^\s*METALS-|^\s*COLUMN-|^\s*UPDATE\s+\d+-|'
    r'^\s*Breakingviews\s+-|'
    
    # Sports
    r'\b(cricket|rugby|football|soccer|golf|tennis|olympics|'
    r'squad\s+for|match\s+report|scored|wicket|batting|'
    r'bowling|quadruple.bogey|figure\s+skating|ice\s+skating)\b|'
    # Lifestyle/domestic
    r'(recipe|horoscope|fashion|beauty|lifestyle|dog\s+treat|'
    r'noodles|insect.topped|households\s+opt\s+for|'
    r'motorists\s+at\s+the\s+pumps|green\s+energy\s+tariffs)\b|'
    # Purely military war briefings with no economic content
    r'war\s+briefing:.{0,120}(drone\s+alert|drones\s+hunt|'
    r'missile\s+strike|air\s+strike|airstrike|troops\s+advance|'
    r'troops\s+deploy|frontline\s+update|civilian\s+casualt|'
    r'soldier\s+killed|bomb\s+explod|shelling|'
    r'recaptur|territorial\s+gain|battlefield)|'
    # Domestic opinion / think tank proposals (not events)
    r'thinktank\s+urges|think\s+tank\s+urges|'
    r'urges\s+UK\s+ministers|calls\s+on\s+UK\s+government|'
    # Firm-level operational (not macro)
    r'(hack|cybersecurity|data\s+breach|supply\s+chain\s+fraud|'
    r'honey\s+fraud|beekeep|dog\s+treat)\b|'
    # German corporate sanctions (not geopolitical)
    r'sanctions\s+on\s+firms\s+in\s+cases\s+of\s+employee',
    re.IGNORECASE
)

# Filtering for major powers/countries only:
    # Must reference a recognised major actor 
    # Excludes weak-state-only articles while retaining Iran/Venezuela/North korea
    # Countries that can produce commodity/proliferation market implications
MAJOR_POWERS = re.compile(
    # North America
    r'\bUS\b|\bU\.S\.|\bUSA\b|United\s+States|American?\b|'
    r'Washington\b|White\s+House\b|Pentagon\b|Congress\b|Senate\b|'
    r'Treasury\b|Federal\s+Reserve\b|Trump\b|Biden\b|'
    r'\bCanada\b|\bCanadian\b|Ottawa\b|Trudeau\b|Carney\b|Freeland\b|'
    r'\bMexico\b|\bMexican\b|'
    # Europe
    r'\bEU\b|European\s+Union\b|Brussels\b|European\b|'
    r'\bECB\b|European\s+Central\s+Bank\b|'
    r'\bUK\b|Britain\b|British\b|London\b|Starmer\b|Sunak\b|Johnson\b|'
    r'Germany\b|German\b|Berlin\b|Scholz\b|Merkel\b|'
    r'France\b|French\b|Paris\b|Macron\b|'
    r'Italy\b|Italian\b|Rome\b|Meloni\b|'
    r'\bNATO\b|\bG7\b|\bG20\b|'
    
    # Asia-Pacific
    r'China\b|Chinese\b|Beijing\b|\bXi\b|'
    r'Japan\b|Japanese\b|Tokyo\b|Kishida\b|'
    r'South\s+Korea\b|Korean\b|Seoul\b|'
    r'Taiwan\b|Taipei\b|'
    r'India\b|Indian\b|Modi\b|New\s+Delhi\b|'
    r'Australia\b|Australian\b|Canberra\b|'
    
    # Russia
    r'Russia[n]?\b|Putin\b|Kremlin\b|Moscow\b|'
    
    # Middle East with macro impact
    r'\bIran\b|\bIranian\b|Tehran\b|'       # Iran included for oil sanctions
    r'\bIsrael\b|\bIsraeli\b|'                  # Israel included for Middle East regional stability
    r'Saudi\b|Riyadh\b|\bOPEC\b|'               # Saudi included for oil market influence
    r'\bUAE\b|Abu\s+Dhabi\b|Dubai\b|'           # Gulf included for Middle East/global finance flows
    r'Qatar\b|Qatari\b|'                        # Qatar for its large LNG market
    
    # Multilateral institutions
    r'\bIMF\b|World\s+Bank\b|\bWTO\b|\bUN\b|'
    r'United\s+Nations\b|'
    
    # Specifically retained: North Korea/Venezuela for commodity/proliferation signal
    r'North\s+Korea\b|Venezuela\b',
    re.IGNORECASE
)

# EXCLUDE Weak states with negligible market impact
# Headlines ONLY about these actors with no major power involvement are dropped
WEAK_STATE_ONLY = re.compile(
    r'^.{0,20}(Central\s+African\s+Republic|Sudan(?!\s+sanction)|'
    r'South\s+Sudan|Mali\b|Niger\b|Burkina\s+Faso|'
    r'Myanmar\b|Cambodia\b|Laos\b|'
    r'Giuliani|Hezbollah(?!\s+sanction))\b.{0,60}$',
    re.IGNORECASE
)

# US-CENTRIC FILTER: Prefer headlines with US market perspective
# Not a hard filter, used in scoring for sampling priority
US_CENTRIC = re.compile(
    r'\bUS\b|\bU\.S\b|American?\b|Washington\b|'
    r'White\s+House\b|Trump\b|Biden\b|Congress\b|'
    r'Wall\s+Street\b|US\s+market|US\s+economy|'
    r'Federal\s+Reserve\b|\bIMF\b|\bWTO\b|\bG7\b',
    re.IGNORECASE
)

# THEMATIC CLASSIFICATION


THEME_PATTERNS = {
    'tariffs_trade': (r'\btariff(s)?\b|trade\s+war\b|import\s+duti|'
                          r'\bWTO\b|trade\s+deal\b|trade\s+dispute\b|'
                          r'trade\s+restriction|market\s+access\b'),
    'sanctions': (r'\bsanction(s|ed)?\b|asset\s+freeze\b|'
                          r'arms\s+embargo\b|secondary\s+sanction'),
    'ukraine_russia': (r'ukraine\b|russia[n]?\b|putin\b|zelenskyy\b|'
                          r'nord\s+stream\b|kremlin\b|shadow\s+fleet'),
    'us_china': (r'us.china\b|china.us\b|sino.american|'
                          r'huawei\b|tiktok\b'),
    'semiconductors': (r'chip\s+(ban|war)\b|\bCHIPS\s+Act\b|'
                          r'semiconductor.{0,20}(export|sanction|restrict)|'
                          r'export\s+control.{0,20}(chip|tech)|'
                          r'dual.use\s+technolog'),
    'supply_chain': (r'supply.chain\s+(resilience|security|'
                          r'disruption|crisis|relocation)\b|'
                          r'reshoring\b|nearshoring\b|friend.shor'),
    'energy_commodities':(r'energy\s+security\b|energy\s+sanction|'
                          r'oil\s+sanction\b|gas\s+pipeline\b|'
                          r'\bOPEC\b.{0,30}(cut|deal|sanction)|'
                          r'grain\s+corridor\b|food\s+security\b'),
    'critical_minerals': (r'rare\s+earth(s)?\b|critical\s+mineral(s)?\b|'
                          r'strategic\s+reserve(s)?\b|'
                          r'mineral.{0,20}(deal|alliance|security)'),
    'covid_securitisation':(r'vaccine\s+nationalism\b|PPE\s+export\b|'
                            r'pandemic\s+supply\s+chain\b|'
                            r'medical\s+suppli.{0,20}(export|ban|restrict)'),
    'industrial_policy': (r'industrial\s+policy\b|strategic\s+autonom\b|'
                          r'economic\s+statecraft\b|economic\s+coercion\b|'
                          r'weaponi[sz]ed\s+interdependence\b|'
                          r'friend.shor|Belt\s+and\s+Road\b|'
                          r'\bAUKUS\b|\bQUAD\b(?!\w)|'
                          r'defence\s+procurement\b'),
    # Taiwan and China tech as distinct from general US-China
    'taiwan_tech':       (r'\btaiwan\b.{0,60}(chip|strait|security|'
                          r'military|tension|tsmc|semiconductor)|'
                          r'tsmc\b|taiwan\s+strait'),
}

# Theme sampling targets for enforcing theme diversity
THEME_TARGETS = {
    'tariffs_trade': 40,   # cap, theme is already overrepresented
    'sanctions': 40,   # cap, already overrepresented
    'ukraine_russia': 25,
    'us_china': 20,
    'semiconductors': 25,
    'supply_chain': 20,
    'energy_commodities': 25,
    'critical_minerals': 20,
    'covid_securitisation': 20,
    'industrial_policy': 25,
    'taiwan_tech': 20,
}
# Total target: 300

def assign_theme(headline):
    for theme, pattern in THEME_PATTERNS.items():
        if re.search(pattern, str(headline), re.IGNORECASE):
            return theme
    return 'other'

def us_centric_score(headline):
    """1 if headline has US-centric framing, 0 otherwise."""
    return 1 if US_CENTRIC.search(str(headline)) else 0

def passes_quality(headline):
    """Returns True only if headline passes all hard quality filters."""
    h = str(headline).strip()
    if len(h) < 30 or len(h) > 280:
        return False
    if DROP_PATTERN.search(h):
        return False
    if not MAJOR_POWERS.search(h):
        return False
    if WEAK_STATE_ONLY.search(h):
        return False
    return True


# COLLECT HEADLINES: 3 DATE BANDS FOR TEMPORAL BALANCE


DATE_BANDS = [
    ('2018-01-01', '2021-12-31', '2018-2021'),  # COVID, Brexit era and US-China trade war
    ('2022-01-01', '2023-12-31', '2022-2023'), # chip war, Ukraine sanctions, supply chain decoupling
    ('2024-01-01', "2026-02-01", "2024-2026"), # Trump tariff era and hostile foreign policy
]

all_headlines = []

# Guardian News API: pull each date band separately

print('=' * 65)
print('Fetching Guardian headlines - 3 date bands x 4 pages... wait.')
print('=' * 65)

for from_date, to_date, band_label in DATE_BANDS:
    band_count = 0
    for page in range(1, 5): # 4 pages x 50 = 200 per band
        try:
            r = requests.get(
                "https://content.guardianapis.com/search",
                params = {
                    "q": GUARDIAN_KEYWORDS,
                    "section":  "world|business|politics",
                    "from-date": from_date,
                    "to-date": to_date,
                    "page-size": 50,
                    "page": page,
                    "show-fields": "headline",
                    "api-key": args.guardian_key,
                },
                timeout=10
            ).json()
        except Exception as e:
            print(f" [{band_label}] Page {page} error: {e}")
            break

        results = r.get("response", {}).get("results", [])
        if not results:
            break

        for item in results:
            h = (item.get("fields", {}).get("headline") or
                 item.get("webTitle", "")).strip()
            if h:
                all_headlines.append({
                    'headline': h,
                    'source': 'guardian',
                    'date': item.get('webPublicationDate', "")[:10],
                    "date_band": band_label,
                })
                band_count += 1
        time.sleep(0.5)

    print(f" [{band_label}] Guardian raw: {band_count}")

g_total = sum(1 for h in all_headlines if h["source"] == "guardian")
print(f"\n Guardian total raw: {g_total}")

# Reuters — stream cc_news with date-band tagging 
print("\n" + "=" * 65)
print("Fetching Reuters via cc_news... Please Wait.")
print("=" * 65)

cc = load_dataset("cc_news", split="train", streaming=True)
reuters_hits = []
scanned = 0

def get_band(date_str):
    if not date_str:
        return "unknown"
    try:
        yr = int(str(date_str)[:4])
        if 2018 <= yr <= 2021: return '2018-2021'
        if 2022 <= yr <= 2023: return '2022-2023'
        if yr >= 2024: return "2024-2026"
    except:
        pass
    return "unknown"

for row in cc:
    scanned += 1
    if scanned % 50000 == 0:
        print(f' Scanned {scanned:,} rows | Reuters hits: {len(reuters_hits)}')

    if "reuters.com" not in row.get("domain", ""):
        continue

    title = str(row.get("title", "")).strip()
    date_str = str(row.get("publish_date", ""))[:10]

    if len(title) < 30:
        continue

    if REUTERS_PATTERN.search(title):
        reuters_hits.append({
            'headline':  title,
            'source': 'reuters',
            "date": date_str,
            "date_band": get_band(date_str),
        })

    if len(reuters_hits) >= 500: # large raw pool to survive all filters
        break

all_headlines.extend(reuters_hits)
print(f"\n  Reuters raw: {len(reuters_hits)}")


# CLEAN & DEDUPLICATE BEFORE SAMPLING

print("\n" + "=" * 65)
print('Deduplication and quality filtering...Please Wait.)
print("=" * 65)

df = pd.DataFrame(all_headlines)
print(f"  Combined raw: {len(df)}")

# Exact dedup on headline text
df = df.drop_duplicates(subset = ["headline"]).reset_index(drop=True)
print(f"  After exact dedup: {len(df)}")

# Quality filter
df["passes"] = df["headline"].apply(passes_quality)
df_q = df[df["passes"]].drop(columns = ["passes"]).reset_index(drop = True)
print(f'  After quality filter: {len(df_q)}')
print(f' Guardian: {(df_q['source']=='guardian').sum()}')
print(f' Reuters:  {(df_q['source']=='reuters').sum()}')

# Using Fuzzy dedup with 85% similarity threshold
print("  Running fuzzy dedup (this may take 30-60 seconds)...")
keep, seen = [], []
for _, row in df_q.iterrows():
    h = row["headline"]
    if all(fuzz.ratio(h, s) < 85 for s in seen):
        keep.append(row)
        seen.append(h)

df_clean = pd.DataFrame(keep).reset_index(drop=True)
print(f"  After fuzzy dedup: {len(df_clean)}")
print(f' Guardian: {(df_clean['source']=='guardian').sum()}')
print(f' Reuters:  {(df_clean['source']=='reuters').sum()}')


# ASSIGNING THEMES AND US-CENTRIC SCORE


df_clean["theme"] = df_clean["headline"].apply(assign_theme)
df_clean["us_score"] = df_clean["headline"].apply(us_centric_score)

print("\nPre-sampling theme distribution:")
print(df_clean.groupby(["theme","source"]).size().unstack(fill_value = 0).to_string())
print("\nDate band distribution:")
print(df_clean.groupby(["date_band","source"]).size().unstack(fill_value = 0).to_string())


# STRATIFIED SAMPLING with THEME * DATE BAND * SOURCE BALANCE


print("\n" + "=" * 65)
print("STEP 4: Stratified sampling...")
print("=" * 65)

def sample_stratum(pool, target, prefer_us_centric=True):
    """
    Sample `target` headlines from pool
    If prefer_us_centric is true, sort US-centric headlines first
    """
    if len(pool) == 0:
        return pd.DataFrame()
    if prefer_us_centric:
        pool = pool.sort_values("us_score", ascending = False)
    n = min(target, len(pool))
    return pool.head(n)  # deterministic — no random_state needed after sort

# Split by source
guard_clean  = df_clean[df_clean["source"] == "guardian"].copy()
reut_clean = df_clean[df_clean["source"] == "reuters"].copy()

guard_selected = []
reut_selected  = []

# Per-source per-theme targets (half of overall target each)
for theme, total_target in THEME_TARGETS.items():
    per_source = max(1, total_target // 2)

    g_pool = guard_clean[guard_clean["theme"] == theme]
    r_pool = reut_clean[reut_clean["theme"]  == theme]

    g_sel = sample_stratum(g_pool, per_source)
    r_sel = sample_stratum(r_pool, per_source)

    guard_selected.append(g_sel)
    reut_selected.append(r_sel)

    print(f' {theme:<25} Guardian: {len(g_sel):3} | Reuters: {len(r_sel):3}')

guard_sampled = pd.concat(guard_selected, ignore_index = True).drop_duplicates(
    subset = ['headline'])
reut_sampled = pd.concat(reut_selected,  ignore_index = True).drop_duplicates(
    subset = ['headline'])

# Fill to 150 per source from remaining themed articles if under target
def top_up(sampled, full_pool, target=150):
    shortfall = target - len(sampled)
    if shortfall <= 0:
        return sampled.head(target)
    used = set(sampled["headline"].tolist())
    remaining = full_pool[
        (~full_pool["headline"].isin(used)) &
        (full_pool["theme"] != "other")
    ].sort_values("us_score", ascending=False)
    extra = remaining.head(shortfall)
    return pd.concat([sampled, extra], ignore_index=True).head(target)

guard_final = top_up(guard_sampled, guard_clean, 150)
reut_final = top_up(reut_sampled,  reut_clean,  150)

print(f'\n  Guardian final: {len(guard_final)}')
print(f' Reuters final:  {len(reut_final)}')

if len(guard_final) < 150:
    print(f'Short of expected number - please increasing page count or loosening keywords.')
if len(reut_final) < 150:
    print(f'Short of expected number - please increasing Reuters raw cap beyond 500')

# FINAL OUTPUT
final = pd.concat([guard_final, reut_final], ignore_index=True)

# Final exact dedup check (safety net)
final = final.drop_duplicates(subset=["headline"]).reset_index(drop=True)

# Shuffle so labeling session is not source-ordered
final = final.sample(frac=1, random_state=42).reset_index(drop=True)

# Add empty label column
final['label'] = "" # fill: positive/negative/neutral

# Keep useful columns only
final = final[["headline", "source", "date", "date_band", "theme", "label"]]

final.to_csv(args.out, index = False)

print("\n" + "=" * 65)
print("Final Dataset Summary")
print("=" * 65)
print(f'Total: {len(final)}')
print(f'\nSource breakdown:')
print(final["source"].value_counts().to_string())
print(f'\nTheme breakdown:')
print(final["theme"].value_counts().to_string())
print(f"\nDate band breakdown:")
print(final["date_band"].value_counts().to_string())
print(f'\nUS-centric headlines: {final['headline'].apply(us_centric_score).sum()} '
      f'/ {len(final)}'
      f'({final['headline'].apply(us_centric_score).mean()*100:.0f}%)')
print(f'\nDate range: {final['date'].min()} → {final['date'].max()};)
print(f"\nSample headlines (first 15):")
for _, row in final.head(15).iterrows():
    print(f' [{row['source']:8}][{row['theme']:22}][{row['date_band']}]'
          f'{row['headline'][:75]}')
print(f'\nSaved to: {args.out}')
