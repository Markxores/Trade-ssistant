import pandas_datareader.data as web
import datetime

end = datetime.datetime.now()
start = end - datetime.timedelta(days=365 * 5)

# --- Candidate series IDs to test ---
# Unemployment rate: FRED's "harmonized unemployment rate" family (LRHUTTTT series)
# is designed for exactly this — standardized, cross-country comparable, monthly
UNEMPLOYMENT_CANDIDATES = {
    "USD": "LRHUTTTTUSM156S",
    "EUR": "LRHUTTTTEZM156S",
    "GBP": "LRHUTTTTGBM156S",
    "JPY": "LRHUTTTTJPM156S",
    "CAD": "LRHUTTTTCAM156S",
    "AUD": "LRHUTTTTAUM156S",
    "NZD": "LRHUTTTTNZM156S",
    "CHF": "LRHUTTTTCHM156S",
}

# Manufacturing PMI: OECD/FRED business confidence proxies are inconsistent —
# testing a mix of ISM (US) and available international PMI-adjacent series
PMI_CANDIDATES = {
    "USD": "MANEMP",  # placeholder — US ISM Manufacturing PMI isn't directly on FRED under a simple ID; testing alternates below
    "EUR": "BSCICP03EZM665S",  # OECD Business Confidence Indicator, Eurozone
    "GBP": "BSCICP03GBM665S",
    "JPY": "BSCICP03JPM665S",
    "CAD": "BSCICP03CAM665S",
    "AUD": "BSCICP03AUM665S",
    "NZD": "BSCICP03NZM665S",
    "CHF": "BSCICP03CHM665S",
}

def check_series(name_dict, category_label):
    print(f"\n{'='*60}")
    print(f"CHECKING: {category_label}")
    print(f"{'='*60}")
    for currency, series_id in name_dict.items():
        try:
            df = web.DataReader(series_id, 'fred', start, end)
            valid_points = df[series_id].dropna()
            if len(valid_points) >= 2:
                latest = valid_points.iloc[-1]
                latest_date = valid_points.index[-1].strftime("%Y-%m")
                print(f"  ✅ {currency} ({series_id}): {len(valid_points)} points, latest={latest:.2f} ({latest_date})")
            else:
                print(f"  ⚠️ {currency} ({series_id}): only {len(valid_points)} points — likely broken/discontinued")
        except Exception as e:
            print(f"  ❌ {currency} ({series_id}): FAILED — {type(e).__name__}: {str(e)[:80]}")

check_series(UNEMPLOYMENT_CANDIDATES, "UNEMPLOYMENT RATE")
check_series(PMI_CANDIDATES, "PMI / BUSINESS CONFIDENCE PROXY")