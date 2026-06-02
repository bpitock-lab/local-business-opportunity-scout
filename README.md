# Local Business Opportunity Scout

This project is a dynamic RAG prototype for evidence-based local business opportunity analysis.

The app allows a user to enter a local area and generate a customer-facing business opportunity brief using open public data. The system builds a local evidence corpus across five market evidence layers:

1. Demand volume
2. Pricing power proxies
3. Competitive intensity
4. Local business supply
5. Foot-traffic and demand anchors

The prototype uses open public data sources such as Census ACS, Census County Business Patterns, and OpenStreetMap-derived local points of interest. Claude is used for synthesis and customer-facing recommendation generation.

## What the app does

The user enters an area, founder budget, operating complexity tolerance, and business categories of interest. The app then:

1. Geocodes the area.
2. Retrieves public market context.
3. Builds a five-layer evidence corpus.
4. Creates a lightweight retrieval index.
5. Generates customer-facing answers and a final opportunity brief.
6. Shows the retrieved evidence used for each answer.

## How to run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## API key

The app requires an Anthropic API key for Claude generation.

For local development, you can either enter the key in the app sidebar or create:

```text
.streamlit/secrets.toml
```

with:

```toml
ANTHROPIC_API_KEY = "your_key_here"
```

Do not commit API keys to GitHub.

For Streamlit Community Cloud, add the same key in the app's Secrets settings.

## Data sources

This prototype uses open public information:

- U.S. Census ACS 5-year Profile data
- U.S. Census County Business Patterns data
- OpenStreetMap / Overpass API points of interest

The app uses these sources to generate proxies for demand volume, pricing power, competitive intensity, business supply, and foot-traffic anchors. It does not claim to predict business success. It is intended to produce a grounded shortlist of opportunities and risks for further validation.

## Limitations

This is a prototype. Data availability varies by geography, OpenStreetMap coverage is uneven, and the current retrieval approach uses TF-IDF lexical retrieval rather than a production vector database. A production version would add embedding retrieval, richer data connectors, local review/customer pain-point data, and more formal opportunity scoring.
