import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Local Business Opportunity Scout",
    page_icon="📍",
    layout="wide",
)

st.title("Local Business Opportunity Scout")
st.caption("A dynamic RAG prototype for evidence-based local business opportunity analysis using open public data.")

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
FALLBACK_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
]

BUSINESS_CATEGORY_TERMS = {
    "Food & beverage": ["restaurant", "cafe", "bar", "fast_food", "bakery", "food_court", "ice_cream", "deli", "pub"],
    "Fitness, health & wellness": ["gym", "fitness_centre", "yoga", "sports_centre", "spa", "massage", "clinic", "pharmacy", "doctors"],
    "Pet services": ["veterinary", "pet", "animal", "dog", "grooming"],
    "Personal care & services": ["hairdresser", "beauty", "laundry", "dry_cleaning", "tailor", "barber", "nails"],
    "Retail & convenience": ["clothes", "convenience", "supermarket", "department_store", "mall", "hardware", "electronics", "books", "gift", "florist"],
    "Home, repair & maintenance": ["hardware", "doityourself", "furniture", "garden_centre", "car_repair", "bicycle", "car_wash"],
    "Child, family & education": ["school", "kindergarten", "childcare", "playground", "library", "tutoring", "music_school"],
    "Professional & financial services": ["bank", "atm", "office", "coworking", "lawyer", "accountant", "insurance", "real_estate"],
    "Entertainment, culture & tourism": ["cinema", "theatre", "museum", "gallery", "attraction", "hotel", "arts_centre", "nightclub"],
    "Mobility & transportation": ["fuel", "charging_station", "bicycle_rental", "car_rental", "parking", "taxi", "bus_station", "station"],
}

ANCHOR_TAGS = {
    "transit": ["bus_station", "station", "subway_entrance", "tram_stop"],
    "education": ["school", "university", "college", "kindergarten"],
    "health": ["hospital", "clinic", "doctors", "pharmacy"],
    "office/employment": ["office"],
    "visitor/culture": ["hotel", "museum", "theatre", "attraction", "gallery", "arts_centre"],
    "public space": ["park", "playground", "library", "community_centre"],
}

NEWS_SIGNAL_TOPICS = {
    "business openings and closures": "business OR restaurant OR retail OR shop OR opening OR closure",
    "development and commercial change": "development OR construction OR permit OR redevelopment OR commercial",
    "tourism and events demand": "tourism OR event OR festival OR hotel OR visitor",
    "housing and affordability pressure": "rent OR housing OR affordability OR apartment OR real estate",
}

# -----------------------------
# Utility helpers
# -----------------------------
def clean_api_key(key: str) -> str:
    if not key:
        return ""
    return key.strip().replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")


def get_anthropic_key() -> str:
    key = ""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = st.session_state.get("ANTHROPIC_API_KEY", "")
    return clean_api_key(key)


def call_claude(prompt: str, max_tokens: int = 1400, temperature: float = 0.2) -> str:
    api_key = get_anthropic_key()
    if not api_key:
        return "Anthropic API key not found. Add it in Streamlit secrets as ANTHROPIC_API_KEY or enter it in the sidebar."
    if any(ord(ch) > 127 for ch in api_key):
        return "The Anthropic API key appears to contain non-ASCII characters. Re-copy it directly from the Anthropic Console."

    client = Anthropic(api_key=api_key)
    last_error = None
    for model in FALLBACK_MODELS:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            last_error = exc
            continue
    return f"Claude call failed after trying available model candidates. Last error: {last_error}"


def safe_request_json(url: str, params=None, headers=None, timeout=30):
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"_error": str(exc), "_url": url}


def normalize_text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def get_nested(dct, keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

# -----------------------------
# Geocoding and public data retrieval
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def geocode_area(area: str) -> dict:
    """Use Census geocoder first because it returns county/state geography. Fall back to Nominatim."""
    census_url = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    params = {
        "address": area,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    data = safe_request_json(census_url, params=params, timeout=30)
    matches = get_nested(data, ["result", "addressMatches"], [])
    if matches:
        m = matches[0]
        coords = m.get("coordinates", {})
        counties = get_nested(m, ["geographies", "Counties"], []) or []
        county = counties[0] if counties else {}
        states = get_nested(m, ["geographies", "States"], []) or []
        state_geo = states[0] if states else {}
        return {
            "input": area,
            "formatted": m.get("matchedAddress", area),
            "lat": coords.get("y"),
            "lon": coords.get("x"),
            "state_fips": state_geo.get("STATE") or county.get("STATE"),
            "county_fips": county.get("COUNTY"),
            "county_name": county.get("NAME"),
            "source": "Census Geocoder",
        }

    # fallback: Nominatim, no FIPS
    headers = {"User-Agent": "local-business-opportunity-scout-demo/1.0"}
    n_url = "https://nominatim.openstreetmap.org/search"
    n_data = safe_request_json(n_url, params={"q": area, "format": "json", "limit": 1}, headers=headers, timeout=30)
    if isinstance(n_data, list) and n_data:
        m = n_data[0]
        return {
            "input": area,
            "formatted": m.get("display_name", area),
            "lat": float(m.get("lat")),
            "lon": float(m.get("lon")),
            "state_fips": None,
            "county_fips": None,
            "county_name": None,
            "source": "OpenStreetMap Nominatim",
        }
    return {"input": area, "formatted": area, "lat": None, "lon": None, "state_fips": None, "county_fips": None, "county_name": None, "source": "unresolved"}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_acs_county_context(state_fips: str, county_fips: str) -> pd.DataFrame:
    if not state_fips or not county_fips:
        return pd.DataFrame([{
            "source": "Census ACS",
            "layer": "Demand volume and pricing power",
            "text": "Census county context unavailable because the selected area could not be matched to a county FIPS code.",
            "metric": "unavailable",
        }])

    url = "https://api.census.gov/data/2023/acs/acs5/profile"
    vars_ = [
        "NAME",
        "DP05_0001E",  # total population
        "DP03_0062E",  # median household income
        "DP04_0134E",  # median gross rent
        "DP04_0046PE", # renter occupied pct
        "DP03_0021PE", # unemployment rate
        "DP03_0025PE", # workers public transit
        "DP03_0024PE", # workers drove alone
    ]
    params = {"get": ",".join(vars_), "for": f"county:{county_fips}", "in": f"state:{state_fips}"}
    data = safe_request_json(url, params=params, timeout=30)
    if isinstance(data, dict) and "_error" in data:
        return pd.DataFrame([{"source": "Census ACS", "layer": "Demand volume and pricing power", "text": f"ACS retrieval failed: {data['_error']}", "metric": "error"}])
    try:
        header, row = data[0], data[1]
        rec = dict(zip(header, row))
        name = rec.get("NAME", "selected county")
        pop = rec.get("DP05_0001E")
        income = rec.get("DP03_0062E")
        rent = rec.get("DP04_0134E")
        renter = rec.get("DP04_0046PE")
        unemp = rec.get("DP03_0021PE")
        transit = rec.get("DP03_0025PE")
        drive = rec.get("DP03_0024PE")
        records = [
            {
                "source": "Census ACS 5-year Profile",
                "layer": "Demand volume",
                "metric": "population",
                "text": f"{name} has an estimated population of {pop}. This is a demand-volume proxy for local businesses and services.",
            },
            {
                "source": "Census ACS 5-year Profile",
                "layer": "Pricing power",
                "metric": "income and rent",
                "text": f"{name} has median household income of ${income} and median gross rent of ${rent}. Higher income and rent levels can support premium positioning but may also imply higher operating costs.",
            },
            {
                "source": "Census ACS 5-year Profile",
                "layer": "Demand mix",
                "metric": "renter share",
                "text": f"{name} has renter-occupied housing share of {renter}%. Higher renter share can indicate demand for convenience, services, flexible retail, and lower-commitment household offerings.",
            },
            {
                "source": "Census ACS 5-year Profile",
                "layer": "Labor and commute context",
                "metric": "employment and commute",
                "text": f"{name} reports unemployment of {unemp}%, public-transit commuting of {transit}%, and drive-alone commuting of {drive}%. These are proxies for labor availability and foot-traffic/commuting patterns.",
            },
        ]
        return pd.DataFrame(records)
    except Exception as exc:
        return pd.DataFrame([{"source": "Census ACS", "layer": "Demand volume and pricing power", "text": f"ACS parsing failed: {exc}", "metric": "error"}])


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_cbp_competition_context(state_fips: str, county_fips: str) -> pd.DataFrame:
    if not state_fips or not county_fips:
        return pd.DataFrame([{
            "source": "Census County Business Patterns",
            "layer": "Competitive intensity",
            "text": "County Business Patterns context unavailable because the selected area could not be matched to a county FIPS code.",
            "metric": "unavailable",
        }])

    # Broad categories that map reasonably to local business opportunity analysis.
    naics = {
        "44-45": "Retail trade",
        "722": "Food services and drinking places",
        "713": "Amusement, gambling, and recreation industries",
        "812": "Personal and laundry services",
        "541": "Professional, scientific, and technical services",
        "621": "Ambulatory health care services",
        "624": "Social assistance",
    }
    rows = []
    url = "https://api.census.gov/data/2022/cbp"
    for code, label in naics.items():
        params = {
            "get": "NAME,NAICS2017_LABEL,ESTAB,EMP,PAYANN",
            "for": f"county:{county_fips}",
            "in": f"state:{state_fips}",
            "NAICS2017": code,
        }
        data = safe_request_json(url, params=params, timeout=30)
        try:
            if isinstance(data, list) and len(data) > 1:
                header, row = data[0], data[1]
                rec = dict(zip(header, row))
                estab = rec.get("ESTAB", "unknown")
                emp = rec.get("EMP", "unknown")
                pay = rec.get("PAYANN", "unknown")
                rows.append({
                    "source": "Census County Business Patterns",
                    "layer": "Competitive intensity",
                    "metric": label,
                    "text": f"County Business Patterns reports {estab} establishments, {emp} employees, and annual payroll of {pay} thousand dollars for {label} in the selected county. This is a competition and category-density proxy.",
                })
        except Exception:
            continue
    if not rows:
        rows.append({
            "source": "Census County Business Patterns",
            "layer": "Competitive intensity",
            "metric": "fallback",
            "text": "County Business Patterns data could not be retrieved for the selected geography during this run. Competitive intensity should be interpreted from OSM supply density instead.",
        })
    return pd.DataFrame(rows)


def make_overpass_query(lat: float, lon: float, radius_m: int = 4000) -> str:
    return f"""
    [out:json][timeout:25];
    (
      node(around:{radius_m},{lat},{lon})[shop];
      node(around:{radius_m},{lat},{lon})[amenity];
      node(around:{radius_m},{lat},{lon})[tourism];
      node(around:{radius_m},{lat},{lon})[leisure];
      node(around:{radius_m},{lat},{lon})[office];
      node(around:{radius_m},{lat},{lon})[public_transport];
      way(around:{radius_m},{lat},{lon})[shop];
      way(around:{radius_m},{lat},{lon})[amenity];
      way(around:{radius_m},{lat},{lon})[tourism];
      way(around:{radius_m},{lat},{lon})[leisure];
      way(around:{radius_m},{lat},{lon})[office];
      way(around:{radius_m},{lat},{lon})[public_transport];
    );
    out center tags 500;
    """


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def get_osm_records(lat: float, lon: float, radius_m: int = 4000) -> pd.DataFrame:
    if lat is None or lon is None:
        return pd.DataFrame([{"name": "unavailable", "category": "unavailable", "text": "OpenStreetMap records unavailable because geocoding failed."}])
    url = "https://overpass-api.de/api/interpreter"
    query = make_overpass_query(lat, lon, radius_m)
    try:
        resp = requests.post(url, data={"data": query}, timeout=45)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return pd.DataFrame([{"name": "retrieval failed", "category": "error", "text": f"OpenStreetMap / Overpass retrieval failed: {exc}"}])

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("brand") or "unnamed place"
        category = tags.get("amenity") or tags.get("shop") or tags.get("tourism") or tags.get("leisure") or tags.get("office") or tags.get("public_transport") or "other"
        lat_val = el.get("lat") or get_nested(el, ["center", "lat"])
        lon_val = el.get("lon") or get_nested(el, ["center", "lon"])
        rows.append({
            "name": name,
            "category": category,
            "amenity": tags.get("amenity", ""),
            "shop": tags.get("shop", ""),
            "tourism": tags.get("tourism", ""),
            "leisure": tags.get("leisure", ""),
            "office": tags.get("office", ""),
            "public_transport": tags.get("public_transport", ""),
            "lat": lat_val,
            "lon": lon_val,
            "text": f"OpenStreetMap record: {name}; category={category}; tags={json.dumps({k:v for k,v in tags.items() if k in ['amenity','shop','tourism','leisure','office','public_transport','name','brand']})}",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame([{"name": "no OSM records", "category": "none", "text": "No nearby OpenStreetMap points of interest were returned for this radius."}])
    return df.drop_duplicates(subset=["name", "category"]).head(700)


def classify_osm_records(osm_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if osm_df.empty:
        empty = pd.DataFrame(columns=["source", "layer", "metric", "text"])
        return empty, empty, empty

    # OpenStreetMap results vary by area. Some locations return only shop tags,
    # only amenities, or sparse records. Ensure all expected tag columns exist
    # so the app does not crash when one evidence type is missing.
    for col in ["category", "amenity", "shop", "tourism", "leisure", "office", "public_transport", "name", "text"]:
        if col not in osm_df.columns:
            osm_df[col] = ""

    category_counts = osm_df["category"].fillna("unknown").value_counts().head(25)
    supply_rows = []
    for cat, count in category_counts.items():
        examples = ", ".join(osm_df.loc[osm_df["category"] == cat, "name"].dropna().astype(str).head(5).tolist())
        supply_rows.append({
            "source": "OpenStreetMap",
            "layer": "Local business supply",
            "metric": cat,
            "text": f"OpenStreetMap returned {int(count)} nearby records in category '{cat}'. Examples include: {examples}. This is a local supply and density signal.",
        })

    anchor_rows = []
    for anchor_label, terms in ANCHOR_TAGS.items():
        mask = osm_df["category"].fillna("").isin(terms) | osm_df["amenity"].fillna("").isin(terms) | osm_df["tourism"].fillna("").isin(terms) | osm_df["leisure"].fillna("").isin(terms) | osm_df["office"].fillna("").isin(terms) | osm_df["public_transport"].fillna("").isin(terms)
        subset = osm_df[mask]
        if len(subset) > 0:
            examples = ", ".join(subset["name"].dropna().astype(str).head(5).tolist())
            anchor_rows.append({
                "source": "OpenStreetMap",
                "layer": "Foot-traffic and demand anchors",
                "metric": anchor_label,
                "text": f"OpenStreetMap returned {len(subset)} nearby {anchor_label} anchors. Examples include: {examples}. These anchors can support walk-in demand, daypart traffic, or recurring local visits.",
            })

    # Derived competitive/pricing notes from category crowding and anchor mix.
    derived_rows = []
    total = max(len(osm_df), 1)
    for label, terms in BUSINESS_CATEGORY_TERMS.items():
        mask = osm_df["category"].fillna("").isin(terms) | osm_df["shop"].fillna("").isin(terms) | osm_df["amenity"].fillna("").isin(terms) | osm_df["leisure"].fillna("").isin(terms)
        count = int(mask.sum())
        if count > 0:
            share = count / total
            derived_rows.append({
                "source": "Derived from OpenStreetMap density",
                "layer": "Competitive intensity and category saturation",
                "metric": label,
                "text": f"For {label}, OpenStreetMap returned {count} nearby relevant records, equal to about {share:.1%} of retrieved local POIs. Higher counts suggest stronger existing supply and potentially higher competitive intensity.",
            })

    return pd.DataFrame(supply_rows), pd.DataFrame(anchor_rows), pd.DataFrame(derived_rows)



@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def get_gdelt_news_signals(area: str, county_name: str | None = None) -> pd.DataFrame:
    """Pull open news/article signals from GDELT. This adds a broader business-context layer
    without requiring a paid API key. It is intentionally summarized as directional evidence,
    not treated as a complete local news census.
    """
    place_terms = [area]
    if county_name and county_name not in area:
        place_terms.append(county_name)
    place_query = " OR ".join([f'"{x}"' for x in place_terms if x])
    if not place_query:
        return pd.DataFrame([{
            "source": "GDELT news/article search",
            "layer": "Local business news and market momentum",
            "metric": "unavailable",
            "text": "Article signal unavailable because the selected area could not be resolved.",
        }])

    rows = []
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    for topic, topic_query in NEWS_SIGNAL_TOPICS.items():
        query = f"({place_query}) ({topic_query})"
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 20,
            "sort": "HybridRel",
            "sourcelang": "english",
        }
        data = safe_request_json(url, params=params, timeout=30)
        if isinstance(data, dict) and "_error" in data:
            rows.append({
                "source": "GDELT news/article search",
                "layer": "Local business news and market momentum",
                "metric": topic,
                "text": f"GDELT article search failed for {topic}: {data['_error']}",
            })
            continue
        articles = data.get("articles", []) if isinstance(data, dict) else []
        examples = []
        domains = []
        for article in articles[:5]:
            title = normalize_text(article.get("title"))
            domain = normalize_text(article.get("domain"))
            date = normalize_text(article.get("seendate"))[:8]
            if title:
                examples.append(f"{title} ({domain}, {date})" if domain else title)
            if domain:
                domains.append(domain)
        domain_sample = ", ".join(pd.Series(domains).drop_duplicates().head(5).tolist()) if domains else "no domains returned"
        headline_sample = "; ".join(examples) if examples else "no example headlines returned"
        rows.append({
            "source": "GDELT open news/article search",
            "layer": "Local business news and market momentum",
            "metric": topic,
            "text": f"Open article search returned {len(articles)} relevant article matches for '{topic}' near {area}. Example sources: {domain_sample}. Example headlines: {headline_sample}. Treat this as a directional signal for market momentum, local concerns, events, openings, construction, affordability pressure, or demand shifts.",
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def build_evidence_corpus(area: str, radius_m: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    geo = geocode_area(area)
    acs = get_acs_county_context(geo.get("state_fips"), geo.get("county_fips"))
    cbp = get_cbp_competition_context(geo.get("state_fips"), geo.get("county_fips"))
    osm_raw = get_osm_records(geo.get("lat"), geo.get("lon"), radius_m)
    supply, anchors, derived = classify_osm_records(osm_raw)
    news = get_gdelt_news_signals(area, geo.get("county_name"))

    frames = [acs, cbp, supply, anchors, derived, news]
    corpus = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    corpus["text"] = corpus["text"].fillna("").astype(str)
    corpus["doc_id"] = [f"DOC-{i+1:03d}" for i in range(len(corpus))]
    corpus["area"] = area
    corpus["created_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return corpus, geo, osm_raw

# -----------------------------
# Retrieval and prompts
# -----------------------------
def retrieve(query: str, corpus: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if corpus.empty:
        return corpus
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    doc_vectors = vectorizer.fit_transform(corpus["text"].fillna(""))
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, doc_vectors).flatten()

    # Source/layer diversity: take top candidates, then avoid one layer dominating.
    candidate_idx = np.argsort(scores)[::-1][: min(len(scores), max(top_k * 4, top_k))]
    selected = []
    layer_counts = {}
    for idx in candidate_idx:
        layer = corpus.iloc[idx].get("layer", "unknown")
        if layer_counts.get(layer, 0) < 4 or len(selected) < 5:
            selected.append(idx)
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        if len(selected) >= top_k:
            break
    results = corpus.iloc[selected].copy()
    results["retrieval_score"] = scores[selected]
    return results


def make_rag_prompt(area, budget, complexity, categories, question, retrieved_docs):
    evidence = "\n\n".join(
        f"[{row.doc_id}] Source: {row.source}; Layer: {row.layer}; Metric: {row.metric}\n{row.text}"
        for row in retrieved_docs.itertuples()
    )
    return f"""
You are Local Business Opportunity Scout, a customer-facing market-entry advisor.

User inputs:
- Area: {area}
- Founder budget: {budget}
- Operating complexity tolerance: {complexity}
- Categories of interest: {categories}

User question:
{question}

Retrieved local evidence:
{evidence}

Answer requirements:
- Give a practical, evidence-backed recommendation.
- Explicitly discuss demand volume, pricing power, and competitive intensity when the evidence supports it.
- Mention where the evidence is strong and where it is incomplete.
- Use retrieved evidence IDs such as [DOC-001] when making factual claims.
- Do not fabricate exact prices, revenue, rents, review counts, or customer demand if they are not in the evidence.
- Write for a customer who is considering a local business, not for a class instructor.
"""


def make_baseline_prompt(area, budget, complexity, categories, question):
    return f"""
You are advising a founder about local business opportunities.
Area: {area}
Founder budget: {budget}
Operating complexity tolerance: {complexity}
Categories of interest: {categories}
Question: {question}

Answer using your general knowledge only. Do not use retrieved evidence.
"""


def final_brief_prompt(area, budget, complexity, categories, findings):
    return f"""
Create a concise customer-facing opportunity brief for a founder evaluating {area}.

Founder inputs:
- Budget: {budget}
- Complexity tolerance: {complexity}
- Categories of interest: {categories}

Use these RAG-generated findings:
{findings}

The brief should include:
1. Executive summary
2. Top 3 business opportunities
3. One category to avoid or approach carefully
4. Evidence base: demand volume, pricing power, competitive intensity, supply, and anchors
5. Next validation steps before investing

Be honest about limitations, but do not bury the recommendation in caveats.
"""

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    api_key_input = st.text_input("Anthropic API key", type="password", help="Optional locally. In Streamlit Cloud, use Secrets instead.")
    if api_key_input:
        st.session_state["ANTHROPIC_API_KEY"] = api_key_input

    area = st.text_input("Area", value="Seattle, WA")
    radius_miles = st.slider("Search radius around selected area center (miles)", min_value=0.5, max_value=10.0, value=2.5, step=0.5, help="Distance from the geocoded location center, measured in miles.")
    radius_m = int(radius_miles * 1609.34)
    budget = st.selectbox("Founder budget", ["Under $50,000", "$50,000-$100,000", "$100,000-$250,000", "$250,000+"], index=1)
    complexity = st.selectbox("Operating complexity tolerance", ["Low", "Medium", "High"], index=1)
    categories = st.multiselect(
        "Areas of interest (optional)",
        list(BUSINESS_CATEGORY_TERMS.keys()),
        default=[],
        help="Leave this blank to let the tool evaluate the market broadly across all supported categories.",
    )
    run_button = st.button("Build opportunity brief", type="primary")

with st.expander("What this prototype does"):
    st.markdown(
        """
This app builds a local RAG corpus for the selected area from six open-data evidence layers:

1. **Demand volume** from Census ACS population and local market context  
2. **Pricing power proxies** from Census income, rent, and housing indicators  
3. **Competitive intensity** from Census County Business Patterns and local category density  
4. **Local business supply** from OpenStreetMap points of interest  
5. **Foot-traffic and demand anchors** from OpenStreetMap transit, schools, parks, offices, health, and cultural amenities  
6. **Local news and market momentum** from open GDELT article search, summarized quantitatively by article-match counts and example headlines

The goal is not to predict business success. The goal is to produce a fast, grounded shortlist of local business opportunities and risks that a founder could validate further. Users can either select specific areas of interest or leave the selection blank for a broader market scan.
        """
    )

# -----------------------------
# Main app
# -----------------------------
if run_button:
    category_context = ", ".join(categories) if categories else "No specific category selected; evaluate the market broadly across all supported local business categories."

    with st.spinner("Geocoding area and building local evidence corpus..."):
        corpus, geo, osm_raw = build_evidence_corpus(area, radius_m)

    st.subheader("1. Evidence corpus built")
    col1, col2, col3 = st.columns(3)
    col1.metric("Evidence records", len(corpus))
    col2.metric("OSM raw POIs", len(osm_raw))
    col3.metric("Evidence layers", corpus["layer"].nunique() if not corpus.empty else 0)

    st.write(f"Matched area: **{geo.get('formatted', area)}**")
    st.write(f"Search distance from selected location center: **{radius_miles:.1f} miles** ({radius_m:,} meters)")
    if geo.get("county_name"):
        st.write(f"County context: **{geo.get('county_name')} County**")

    layer_counts = corpus["layer"].value_counts().reset_index()
    layer_counts.columns = ["Evidence layer", "Records"]
    st.dataframe(layer_counts, use_container_width=True, hide_index=True)

    with st.expander("View sample retrieved evidence records"):
        st.dataframe(corpus[["doc_id", "source", "layer", "metric", "text"]].head(20), use_container_width=True, hide_index=True)

    questions = [
        "What are the strongest local business opportunities in this area?",
        "Which categories look oversaturated or risky?",
        f"What business would fit a founder with {budget} and {complexity.lower()} operating complexity tolerance?",
        "Where does the evidence suggest pricing power, and where is price sensitivity a risk?",
        "What should a founder avoid in this market, and what should they validate next?",
    ]

    st.subheader("2. Customer-facing opportunity analysis")
    outputs = []
    for i, question in enumerate(questions, start=1):
        st.markdown(f"### {i}. {question}")
        retrieved = retrieve(question, corpus, top_k=10)
        prompt = make_rag_prompt(area, budget, complexity, category_context, question, retrieved)
        with st.spinner("Generating evidence-backed answer..."):
            answer = call_claude(prompt, max_tokens=1300)
        st.markdown(answer)
        with st.expander("Retrieved evidence used for this answer"):
            st.dataframe(retrieved[["doc_id", "source", "layer", "metric", "retrieval_score", "text"]], use_container_width=True, hide_index=True)
        outputs.append({"question": question, "answer": answer})

    st.subheader("3. Final local business opportunity brief")
    findings = "\n\n".join([f"Question: {o['question']}\nAnswer: {o['answer']}" for o in outputs])
    with st.spinner("Synthesizing final customer brief..."):
        final_brief = call_claude(final_brief_prompt(area, budget, complexity, category_context, findings), max_tokens=1700)
    st.markdown(final_brief)

else:
    st.info("Enter an area, set founder constraints, and click **Build opportunity brief**.")
