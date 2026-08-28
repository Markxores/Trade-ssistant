import dbnomics
import pandas as pd

# Set a generous timeout tolerance since DBnomics can be slower than FRED
CANDIDATES = [
    {
        "currency": "EUR",
        "provider": "Eurostat",
        "dataset": "une_rt_m",
        "series_code": "M.SA.TOTAL.PC_ACT.EA20",  # seasonally adjusted, % of active population, Euro Area 20
        "note": "Eurostat harmonized unemployment, monthly, seasonally adjusted"
    },
    {
        "currency": "CHF",
        "provider": "Eurostat",
        "dataset": "une_rt_m",
        "series_code": "M.SA.TOTAL.PC_ACT.CH",  # Eurostat DOES publish comparison data for non-EU Switzerland
        "note": "Testing if Eurostat's comparison series covers Switzerland"
    },
    {
        "currency": "NZD",
        "provider": "OECD",
        "dataset": "MEI",
        "series_code": "NZL.LRHUTTTT.STSA.M",
        "note": "Fallback test — likely dead given prior OECD MEI findings, testing anyway for completeness"
    },
]

print("Testing dbnomics library import...")
try:
    import dbnomics
    print(f"✅ dbnomics imported successfully\n")
except ImportError as e:
    print(f"❌ dbnomics not installed. Run: pip install dbnomics --break-system-packages")
    print(f"Error: {e}")
    exit()

for candidate in CANDIDATES:
    print(f"{'='*70}")
    print(f"TESTING: {candidate['currency']} — {candidate['note']}")
    print(f"Provider: {candidate['provider']}  Dataset: {candidate['dataset']}  Series: {candidate['series_code']}")
    print(f"{'='*70}")

    try:
        df = dbnomics.fetch_series(
            provider_code=candidate["provider"],
            dataset_code=candidate["dataset"],
            series_code=candidate["series_code"]
        )

        if df is None or df.empty:
            print("⚠️ Returned empty DataFrame — series code likely wrong or unavailable\n")
            continue

        # DBnomics DataFrames have 'period' and 'value' columns among others
        df_clean = df[['period', 'value']].dropna().sort_values('period')

        if df_clean.empty:
            print("⚠️ All values are NaN after cleaning\n")
            continue

        latest_row = df_clean.iloc[-1]
        latest_period = latest_row['period']
        latest_value = latest_row['value']
        point_count = len(df_clean)

        print(f"✅ SUCCESS: {point_count} points")
        print(f"   Latest: {latest_value} on {latest_period}")
        print(f"   Earliest: {df_clean.iloc[0]['value']} on {df_clean.iloc[0]['period']}\n")

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)[:150]}\n")

print("\nDone. Compare 'latest' dates above against today's date to judge true freshness —")
print("a successful fetch with a stale latest date is still a dead end, same as the FRED findings.")