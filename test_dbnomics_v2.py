import dbnomics as db
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def test_dbnomics_v2():
    print("--- DBnomics V2 Diagnostic (Dictionary Routing) ---\n")
    
    # 1. Eurostat (EUR)
    print("1. Fetching Euro Area (EUR) Unemployment...")
    try:
        df_eur = db.fetch_series(
            'Eurostat', 'une_rt_m',
            dimensions={
                "geo": ["EA20"],     # Euro Area 20 countries
                "s_adj": ["SA"],     # Seasonally Adjusted
                "age": ["TOTAL"],    # All ages
                "unit": ["PC_ACT"],  # Percentage of active population
                "sex": ["T"],        # Total (Men + Women)
                "freq": ["M"]        # Monthly
            }
        )
        latest = df_eur.dropna(subset=['value']).iloc[-1]
        print(f"✅ SUCCESS: {latest['period']} = {latest['value']}%\n")
    except Exception as e:
        print(f"❌ EUR Error: {e}\n")

    # 2. Switzerland (CHF) via OECD MEI
    print("2. Fetching Switzerland (CHF) Unemployment...")
    try:
        df_chf = db.fetch_series(
            'OECD', 'MEI',
            dimensions={
                "LOCATION": ["CHE"],     # Switzerland
                "SUBJECT": ["LRHUTTTT"], # Harmonized Unemployment Rate
                "MEASURE": ["STSA"],     # Level, Seasonally Adjusted
                "FREQUENCY": ["M"]       # Monthly
            }
        )
        latest = df_chf.dropna(subset=['value']).iloc[-1]
        print(f"✅ SUCCESS: {latest['period']} = {latest['value']}%\n")
    except Exception as e:
        print(f"❌ CHF Error: {e}\n")

    # 3. New Zealand (NZD) via OECD MEI
    print("3. Fetching New Zealand (NZD) Unemployment...")
    try:
        df_nzd = db.fetch_series(
            'OECD', 'MEI',
            dimensions={
                "LOCATION": ["NZL"],     # New Zealand
                "SUBJECT": ["LRHUTTTT"], # Harmonized Unemployment Rate
                "MEASURE": ["STSA"],     # Level, Seasonally Adjusted
                "FREQUENCY": ["Q"]       # Quarterly (NZ reports Qtrly, not monthly)
            }
        )
        latest = df_nzd.dropna(subset=['value']).iloc[-1]
        print(f"✅ SUCCESS: {latest['period']} = {latest['value']}%\n")
    except Exception as e:
        print(f"❌ NZD Error: {e}\n")

if __name__ == "__main__":
    test_dbnomics_v2()