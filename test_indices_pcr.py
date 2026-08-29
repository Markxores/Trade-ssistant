import yfinance as yf
import pandas as pd
import requests

INDEX_ETF_MAPPING = {
    "US 500 (S&P 500)": "SPY",
    "US Tech 100 (Nasdaq)": "QQQ",
    "US 30 (Dow Jones)": "DIA",
    "US 2000 (Russell 2000)": "IWM"
}

def test_pcr():
    print("--- Testing ETF Options Chains ---\n")
    
    # Pre-warmed session with Chrome headers
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    })
    try:
        session.get("https://finance.yahoo.com", timeout=5)
    except Exception:
        pass

    for name, etf in INDEX_ETF_MAPPING.items():
        print(f"Checking {name} [{etf}]...")
        try:
            asset = yf.Ticker(etf, session=session)
            expirations = asset.options
            
            if not expirations:
                print(f"  ❌ No options expirations returned for {etf}.\n")
                continue
                
            print(f"  ✅ Expirations found: {len(expirations)} (Front: {expirations[0]})")
            
            # Fetch front expiration chain
            chain = asset.option_chain(expirations[0])
            total_puts_oi = chain.puts['openInterest'].fillna(0).sum()
            total_calls_oi = chain.calls['openInterest'].fillna(0).sum()
            total_puts_vol = chain.puts['volume'].fillna(0).sum()
            total_calls_vol = chain.calls['volume'].fillna(0).sum()
            
            pcr_oi = total_puts_oi / (total_calls_oi + 0.0001)
            pcr_vol = total_puts_vol / (total_calls_vol + 0.0001)
            
            print(f"  📊 Put/Call OI: {pcr_oi:.2f} | Put/Call Vol: {pcr_vol:.2f}\n")
            
        except Exception as e:
            print(f"  ❌ Error fetching {etf}: {str(e)}\n")

if __name__ == "__main__":
    test_pcr()