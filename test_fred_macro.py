import pandas as pd
import pandas_datareader.data as web
import datetime

FRED_MACRO_TICKERS = {
    "USD": {"Rate": "FEDFUNDS", "CPI": "CPIAUCSL", "Jobs": "UNRATE", "Growth": "GDPC1"},
    "EUR": {"Rate": "IR3TIB01EZM156N", "CPI": "CP0000EZ19M086NEST", "Jobs": "LRHUTTTTEZM156S", "Growth": "EUNGDPRQDSMEI"},
    "GBP": {"Rate": "IR3TIB01GBM156N", "CPI": "GBRCPIALLMINMEI", "Jobs": "LRHUTTTTGBM156S", "Growth": "GBRGDPRQDSMEI"},
    "JPY": {"Rate": "IR3TIB01JPM156N", "CPI": "JPNCPIALLMINMEI", "Jobs": "LRHUTTTTJPM156S", "Growth": "JPNGDPRQDSMEI"},
    "CAD": {"Rate": "IR3TIB01CAM156N", "CPI": "CANCPIALLMINMEI", "Jobs": "LRHUTTTTCAM156S", "Growth": "CANGDPRQDSMEI"},
    "AUD": {"Rate": "IR3TIB01AUM156N", "CPI": "CPALTT01AUQ657N", "Jobs": "LRHUTTTTAUM156S", "Growth": "AUSGDPRQDSMEI"}, 
    "NZD": {"Rate": "IR3TIB01NZM156N", "CPI": "CPALTT01NZQ657N", "Jobs": "LRUNTTTTNZQ156S", "Growth": "NZLGDPRQDSMEI"},
    "CHF": {"Rate": "IR3TIB01CHM156N", "CPI": "CHECPIALLMINMEI", "Jobs": "LRHUTTTTCHA156N", "Growth": "CHEGDPRQDSMEI"}
}

def run_fred_diagnostics():
    print("--- FRED Macro Database Diagnostic ---\n")
    
    end = datetime.datetime.now()
    # Pull 2 years of data to guarantee we can look back exactly 12 months for YoY calculation
    start = end - datetime.timedelta(days=730)  
    
    # Flatten dictionary to pull all tickers simultaneously for speed
    all_tickers = []
    for data in FRED_MACRO_TICKERS.values():
        all_tickers.extend(list(data.values()))
        
    print(f"Requesting {len(all_tickers)} series from FRED...\n")
    
    try:
        df = web.DataReader(all_tickers, 'fred', start, end)
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        return
        
    for currency, indicators in FRED_MACRO_TICKERS.items():
        print(f"[{currency}]")
        
        # 1. Rate Check
        rate_series = df[indicators['Rate']].dropna()
        if not rate_series.empty:
            print(f"  - Rate: {rate_series.index[-1].strftime('%Y-%m-%d')} = {rate_series.iloc[-1]:.2f}%")
        else:
            print("  - Rate: ❌ No Recent Data")
            
        # 2. YoY Inflation Check (Derived dynamically from CPI Index)
        cpi_series = df[indicators['CPI']].dropna()
        if len(cpi_series) >= 4:
            try:
                latest_cpi_date = cpi_series.index[-1]
                target_date = latest_cpi_date - pd.DateOffset(years=1)
                
                # Fetch the closest CPI print from exactly one year ago
                past_cpi_idx = cpi_series.index.get_indexer([target_date], method='nearest')[0]
                past_cpi = cpi_series.iloc[past_cpi_idx]
                latest_cpi = cpi_series.iloc[-1]
                
                yoy_inflation = ((latest_cpi - past_cpi) / past_cpi) * 100
                print(f"  - YoY Inflation: {latest_cpi_date.strftime('%Y-%m-%d')} = {yoy_inflation:.2f}%")
            except Exception as e:
                print(f"  - YoY Inflation: ❌ Calc Error ({e})")
        else:
            print("  - YoY Inflation: ❌ Insufficient history")
            
        # 3. Jobs Check
        jobs_series = df[indicators['Jobs']].dropna()
        if not jobs_series.empty:
            print(f"  - Jobs: {jobs_series.index[-1].strftime('%Y-%m-%d')} = {jobs_series.iloc[-1]:.2f}%")
        else:
            print("  - Jobs: ❌ No Recent Data")
            
        # 4. Growth Check
        growth_series = df[indicators['Growth']].dropna()
        if not growth_series.empty:
            print(f"  - Growth: {growth_series.index[-1].strftime('%Y-%m-%d')} = {growth_series.iloc[-1]:.2f}")
        else:
            print("  - Growth: ❌ No Recent Data")
            
        print("")

if __name__ == "__main__":
    run_fred_diagnostics()