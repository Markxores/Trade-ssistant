import yfinance as yf
import pandas as pd
import requests

# --- Spoofed browser session (same as your main app) ---
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
})

# All 11 ETF proxies currently in your INDEX_ETF_MAPPING
INDEX_ETF_MAPPING = {
    "US 500 (S&P 500)": "SPY",
    "US Tech 100 (Nasdaq)": "QQQ",
    "US 30 (Dow Jones)": "DIA",
    "US 2000 (Russell 2000)": "IWM",
    "UK 100 (FTSE)": "EWU",
    "Germany 40 (DAX)": "EWG",
    "France 40 (CAC)": "EWQ",
    "Europe 50 (Euro Stoxx)": "FEZ",
    "Japan 225 (Nikkei)": "EWJ",
    "Hong Kong 50 (Hang Seng)": "EWH",
    "Australia 200 (ASX)": "EWA",
}

results = []

for name, ticker in INDEX_ETF_MAPPING.items():
    print(f"\n=== {name} ({ticker}) ===")
    try:
        asset = yf.Ticker(ticker, session=session)
        expirations = asset.options

        if not expirations:
            print("  ❌ No options expirations returned (possibly blocked or no options market)")
            results.append({
                "Instrument": name, "Ticker": ticker, "Expiry": None,
                "OI_Put": None, "OI_Call": None, "PCR_OI": None,
                "Vol_Put": None, "Vol_Call": None, "PCR_Vol": None,
                "Status": "No expirations"
            })
            continue

        # Use nearest expiry, same as your main app
        nearest_expiry = expirations[0]
        chain = asset.option_chain(nearest_expiry)

        oi_put = chain.puts.get('openInterest', pd.Series(dtype=float)).fillna(0).sum()
        oi_call = chain.calls.get('openInterest', pd.Series(dtype=float)).fillna(0).sum()
        vol_put = chain.puts.get('volume', pd.Series(dtype=float)).fillna(0).sum()
        vol_call = chain.calls.get('volume', pd.Series(dtype=float)).fillna(0).sum()

        pcr_oi = (oi_put / oi_call) if oi_call > 0 else None
        pcr_vol = (vol_put / vol_call) if vol_call > 0 else None

        print(f"  Expiry used: {nearest_expiry}")
        print(f"  OI  -> Put: {oi_put:.0f}  Call: {oi_call:.0f}  PCR_OI: {pcr_oi}")
        print(f"  Vol -> Put: {vol_put:.0f}  Call: {vol_call:.0f}  PCR_Vol: {pcr_vol}")

        results.append({
            "Instrument": name, "Ticker": ticker, "Expiry": nearest_expiry,
            "OI_Put": oi_put, "OI_Call": oi_call, "PCR_OI": pcr_oi,
            "Vol_Put": vol_put, "Vol_Call": vol_call, "PCR_Vol": pcr_vol,
            "Status": "OK"
        })

    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {e}")
        results.append({
            "Instrument": name, "Ticker": ticker, "Expiry": None,
            "OI_Put": None, "OI_Call": None, "PCR_OI": None,
            "Vol_Put": None, "Vol_Call": None, "PCR_Vol": None,
            "Status": f"Error: {e}"
        })

# --- Save to CSV for easy pasting / re-analysis ---
df = pd.DataFrame(results)
df.to_csv("pcr_diagnostic.csv", index=False)
print("\n\n--- SUMMARY TABLE ---")
print(df.to_string(index=False))
print("\nSaved to pcr_diagnostic.csv")