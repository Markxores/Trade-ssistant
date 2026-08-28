import dbnomics

CANDIDATES = [
    {
        "currency": "EUR",
        "series_code": "M.SA.TOTAL.PC_ACT.T.EA21",  # corrected: 6 dimensions, current Euro area code
        "note": "Eurostat, corrected dimension count + current EA21 code"
    },
    {
        "currency": "EUR_fallback",
        "series_code": "M.SA.TOTAL.PC_ACT.T.EA20",  # in case EA21 isn't populated yet
        "note": "Fallback to EA20 in case the new code has no data yet"
    },
    {
        "currency": "CHF",
        "series_code": "M.SA.TOTAL.PC_ACT.T.CH",  # corrected: 6 dimensions, CH confirmed valid
        "note": "Eurostat's own CH comparator series"
    },
]

for c in CANDIDATES:
    print(f"{'='*70}")
    print(f"TESTING: {c['currency']} — {c['note']}")
    print(f"Series: {c['series_code']}")
    print(f"{'='*70}")

    try:
        df = dbnomics.fetch_series(
            provider_code="Eurostat",
            dataset_code="une_rt_m",
            series_code=c["series_code"]
        )

        if df is None or df.empty:
            print("⚠️ Empty result\n")
            continue

        df_clean = df[['period', 'value']].dropna().sort_values('period')
        if df_clean.empty:
            print("⚠️ All NaN after cleaning\n")
            continue

        latest = df_clean.iloc[-1]
        print(f"✅ SUCCESS: {len(df_clean)} points")
        print(f"   Latest: {latest['value']} on {latest['period']}\n")

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)[:150]}\n")