import pandas_datareader.data as web
import pandas as pd
import datetime

FRED_MACRO_TICKERS = {
    "USD": {"Rate": "FEDFUNDS", "CPI": "CPIAUCSL"},
    "EUR": {"Rate": "IR3TIB01EZM156N", "CPI": "CP0000EZ19M086NEST"},
    "GBP": {"Rate": "IR3TIB01GBM156N", "CPI": "GBRCPIALLMINMEI"},
    "JPY": {"Rate": "IR3TIB01JPM156N", "CPI": "JPNCPIALLMINMEI"},
    "CAD": {"Rate": "IR3TIB01CAM156N", "CPI": "CANCPIALLMINMEI"},
    "AUD": {"Rate": "IR3TIB01AUM156N", "CPI": "CPALTT01AUQ657N"},
    "NZD": {"Rate": "IR3TIB01NZM156N", "CPI": "CPALTT01NZQ657N"},
    "CHF": {"Rate": "IR3TIB01CHM156N", "CPI": "CHECPIALLMINMEI"}
}

end = datetime.datetime.now()
start = end - datetime.timedelta(days=365 * 5)

all_tickers = []
for data in FRED_MACRO_TICKERS.values():
    all_tickers.extend([data["Rate"], data["CPI"]])

print("Downloading FRED data...")
df = web.DataReader(all_tickers, 'fred', start, end)
print(f"✅ Downloaded. Shape: {df.shape}\n")

rate_changes = []
cpi_changes = []

for currency, data in FRED_MACRO_TICKERS.items():
    rate_col, cpi_col = data["Rate"], data["CPI"]
    print(f"--- {currency} ---")

    try:
        rate_series = df[rate_col].dropna()
        print(f"  Rate ({rate_col}): {len(rate_series)} valid points")
        if len(rate_series) >= 2:
            rate_diffs = rate_series.diff().dropna()
            rate_changes.extend(rate_diffs.tolist())
        else:
            print(f"  ⚠️ Skipping rate diff for {currency} — insufficient data")
    except Exception as e:
        print(f"  ❌ Rate error for {currency}: {e}")

    try:
        cpi_series = df[cpi_col].dropna()
        print(f"  CPI ({cpi_col}): {len(cpi_series)} valid points")
        if len(cpi_series) >= 2:
            cpi_pct_changes = (cpi_series.pct_change() * 100).dropna()
            cpi_changes.extend(cpi_pct_changes.tolist())
        else:
            print(f"  ⚠️ Skipping CPI pct_change for {currency} — insufficient data (this is likely the crash cause)")
    except Exception as e:
        print(f"  ❌ CPI error for {currency}: {e}")

    print()

rate_changes = pd.Series(rate_changes)
cpi_changes = pd.Series(cpi_changes)

print("--- RATE CHANGE (monthly, raw) ---")
if len(rate_changes) > 0:
    print(f"n={len(rate_changes)}  min={rate_changes.min():.3f}  max={rate_changes.max():.3f}  "
          f"mean={rate_changes.mean():.3f}  median={rate_changes.median():.3f}  std={rate_changes.std():.3f}")
else:
    print("No valid rate change data collected.")

print("\n--- CPI CHANGE % (monthly, raw) ---")
if len(cpi_changes) > 0:
    print(f"n={len(cpi_changes)}  min={cpi_changes.min():.3f}  max={cpi_changes.max():.3f}  "
          f"mean={cpi_changes.mean():.3f}  median={cpi_changes.median():.3f}  std={cpi_changes.std():.3f}")
else:
    print("No valid CPI change data collected.")

if len(rate_changes) > 0 and len(cpi_changes) > 0:
    print("\n--- COMBINED SCORE CLIP-RATE CHECK (rate_mult, cpi_mult) ---")
    for rate_mult, cpi_mult in [(15, 10), (50, 30), (80, 50), (120, 80), (150, 100)]:
        combined = (rate_changes.sample(len(cpi_changes), replace=True).reset_index(drop=True) * rate_mult) - \
                   (cpi_changes.reset_index(drop=True) * cpi_mult)
        scaled = combined.clip(-100, 100)
        pct_clipped = (scaled.abs() >= 99.5).mean() * 100
        print(f"  rate×{rate_mult}, cpi×{cpi_mult}: median={combined.median():.1f}  "
              f"std={combined.std():.1f}  pct_at_100_ceiling={pct_clipped:.1f}%")