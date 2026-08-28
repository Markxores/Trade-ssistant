import os
import json
import time
import datetime
import requests
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import re

# ============================================================
# CONSTANTS & MAPPINGS
# ============================================================
FF_DB_FILE = "ff_history.json"
COOLDOWN_SECONDS = 900  # 15 minutes

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

INDEX_TO_CURRENCY = {
    "S&P 500": "USD", "Nasdaq": "USD", "Dow Jones": "USD",
    "DAX": "EUR", "FTSE 100": "GBP", "Nikkei 225": "JPY"
}

# ============================================================
# PILLAR 1: STRUCTURAL MACRO (FRED)
# ============================================================
def get_structural_macro():
    """Calculates Base Rate + Inflation Momentum (Monthly change in YoY CPI)"""
    scores = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=730)
        
        all_tickers = []
        for data in FRED_MACRO_TICKERS.values():
            all_tickers.extend(list(data.values()))
            
        df = web.DataReader(all_tickers, 'fred', start, end)
        
        for currency, indicators in FRED_MACRO_TICKERS.items():
            score = 0
            
            # Rate Level Logic
            rate_series = df[indicators['Rate']].dropna()
            if not rate_series.empty:
                current_rate = rate_series.iloc[-1]
                score += (current_rate - 2.0) * 5.0  # Above 2% is hawkish
            
            # Inflation Momentum Logic
            cpi_series = df[indicators['CPI']].dropna()
            if len(cpi_series) >= 14:
                latest_cpi = cpi_series.iloc[-1]
                past_cpi_1 = cpi_series.iloc[-13]
                past_cpi_2 = cpi_series.iloc[-14]
                
                latest_yoy = ((latest_cpi - past_cpi_1) / past_cpi_1) * 100
                prev_yoy = ((cpi_series.iloc[-2] - past_cpi_2) / past_cpi_2) * 100
                yoy_change = latest_yoy - prev_yoy
                
                # Blend Level + Momentum
                level_score = (latest_yoy - 2.0) * 5.0
                momentum_score = yoy_change * 25.0
                score += level_score + momentum_score
                
            scores[currency] = max(-100, min(100, score))
    except Exception:
        pass
    
    return scores

# ============================================================
# PILLAR 2: PROXY MOMENTUM (YFINANCE)
# ============================================================
def get_proxy_momentum():
    """Evaluates cross-asset pricing to gauge real-time currency momentum."""
    scores = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}
    try:
        # Standard Yahoo Proxy tickers
        tickers = ["^TNX", "^VIX", "GC=F", "CL=F"]
        df = yf.download(tickers, period="1mo", interval="1d")['Close']
        
        if not df.empty and len(df) >= 2:
            # Calculate % change over the last 5 days
            roc = ((df.iloc[-1] - df.iloc[-5]) / df.iloc[-5]) * 100
            
            vix_change = roc.get('^VIX', 0)
            yield_change = roc.get('^TNX', 0)
            gold_change = roc.get('GC=F', 0)
            oil_change = roc.get('CL=F', 0)
            
            # Proxy Mapping
            scores["USD"] = (yield_change * 3.0) - (vix_change * 0.5)
            scores["JPY"] = vix_change * 1.5 - (yield_change * 2.0)
            scores["CHF"] = vix_change * 1.5
            scores["CAD"] = oil_change * 2.0 + (vix_change * -1.0)
            scores["AUD"] = gold_change * 2.0 + (vix_change * -1.5)
            scores["NZD"] = gold_change * 1.5 + (vix_change * -1.5)
            scores["EUR"] = (vix_change * -0.5)
            scores["GBP"] = (vix_change * -0.5)
            
            for k in scores:
                scores[k] = max(-100, min(100, scores[k]))
    except Exception:
        pass
        
    return scores

# ============================================================
# PILLAR 3: EVENT SURPRISE (FOREXFACTORY LAZY UPDATER)
# ============================================================
def parse_econ_val(val_str):
    if val_str is None or val_str in ["", "None"]: return None
    val_clean = str(val_str).strip().replace(',', '')
    if '%' in val_clean:
        try: return float(val_clean.replace('%', ''))
        except: return None
        
    multiplier = 1.0
    if val_clean.lower().endswith('k'): multiplier, val_clean = 1000.0, val_clean[:-1]
    elif val_clean.lower().endswith('m'): multiplier, val_clean = 1000000.0, val_clean[:-1]
    elif val_clean.lower().endswith('b'): multiplier, val_clean = 1000000000.0, val_clean[:-1]
        
    try: return float(val_clean) * multiplier
    except: return None

def update_and_get_ff_surprises():
    """Maintains local JSON database of FF events and returns rolling surprise score."""
    history = {}
    
    # 1. Check Rate-Limit Cooldown
    if os.path.exists(FF_DB_FILE):
        try:
            with open(FF_DB_FILE, 'r') as f:
                history = json.load(f)
            if time.time() - os.path.getmtime(FF_DB_FILE) < COOLDOWN_SECONDS:
                live_fetch_needed = False
            else:
                live_fetch_needed = True
        except:
            live_fetch_needed = True
    else:
        live_fetch_needed = True

    # 2. Lazy Update
    if live_fetch_needed:
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if response.status_code == 200:
                for event in response.json():
                    if event.get('impact') == 'High':
                        date_str = str(event.get('date', '')).split('T')[0]
                        event_id = f"{date_str}_{event.get('country')}_{event.get('title')}"
                        # Upsert logic: Update if actual is present, or if new
                        if event_id not in history or event.get('actual'):
                            history[event_id] = event
                            
                with open(FF_DB_FILE, 'w') as f:
                    json.dump(history, f, indent=4)
        except Exception:
            pass

    # 3. Calculate Scores from Archive
    scores = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}
    event_counts = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}
    
    # Only look at the last 30 days to ensure momentum is fresh
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    
    for event_id, event in history.items():
        try:
            event_date = datetime.datetime.strptime(event_id.split('_')[0], '%Y-%m-%d')
            if event_date < thirty_days_ago:
                continue
                
            currency = event.get('country')
            if currency not in scores:
                continue
                
            actual = parse_econ_val(event.get('actual'))
            forecast = parse_econ_val(event.get('forecast'))
            
            if actual is not None and forecast is not None:
                deviation = (actual - forecast) / (abs(forecast) + 0.0001)
                surprise_score = deviation * 100
                
                title = str(event.get('title')).lower()
                if any(k in title for k in ['unemployment', 'claims', 'claimant']):
                    surprise_score = -surprise_score
                
                scores[currency] += max(-50, min(50, surprise_score))
                event_counts[currency] += 1
        except Exception:
            continue
            
    # Normalize by event count
    has_data_flags = {}
    for k in scores:
        if event_counts[k] > 0:
            scores[k] = max(-100, min(100, scores[k] / event_counts[k]))
            has_data_flags[k] = True
        else:
            has_data_flags[k] = False
            
    return scores, has_data_flags

# ============================================================
# MASTER FUNDAMENTAL ENGINE
# ============================================================
def calculate_macro_score(currency, struct_data, proxy_data, surprise_data, has_surprise_flags):
    """Calculates score for a single currency using dynamic averaging fallback."""
    c_struct = struct_data.get(currency, 0)
    c_proxy = proxy_data.get(currency, 0)
    c_surp = surprise_data.get(currency, 0)
    
    active_scores = [c_struct, c_proxy]
    if has_surprise_flags.get(currency, False):
        active_scores.append(c_surp)
        
    final_score = sum(active_scores) / len(active_scores)
    return final_score, c_struct, c_proxy, c_surp

def calculate_fundamental_score(name, asset_class):
    try:
        # Load the Triad
        struct_data = get_structural_macro()
        proxy_data = get_proxy_momentum()
        surprise_data, has_surprise_flags = update_and_get_ff_surprises()
        
        details = {}
        
        # --- FOREX LOGIC ---
        if "Forex" in asset_class:
            base, quote = name.split("/")
            base_score, b_st, b_pr, b_su = calculate_macro_score(base, struct_data, proxy_data, surprise_data, has_surprise_flags)
            quote_score, q_st, q_pr, q_su = calculate_macro_score(quote, struct_data, proxy_data, surprise_data, has_surprise_flags)
            
            final_score = base_score - quote_score
            details = {
                f"{base} Structural": round(b_st, 1),
                f"{base} Proxy": round(b_pr, 1),
                f"{base} Surprise": round(b_su, 1) if has_surprise_flags.get(base) else "Gathering Data",
                f"{quote} Structural": round(q_st, 1),
                f"{quote} Proxy": round(q_pr, 1),
                f"{quote} Surprise": round(q_su, 1) if has_surprise_flags.get(quote) else "Gathering Data",
            }
            
        # --- INDICES LOGIC ---
        elif "Index" in asset_class:
            # Map index to currency, and INVERT the score (Hawkish rates = Bearish indices)
            currency = INDEX_TO_CURRENCY.get(name, "USD")
            raw_score, c_st, c_pr, c_su = calculate_macro_score(currency, struct_data, proxy_data, surprise_data, has_surprise_flags)
            final_score = -raw_score
            
            details = {
                "Host Currency": currency,
                "Structural Macro (Inverted)": round(-c_st, 1),
                "Proxy Momentum (Inverted)": round(-c_pr, 1),
                "Event Surprise (Inverted)": round(-c_su, 1) if has_surprise_flags.get(currency) else "Gathering Data",
            }
            
        # --- COMMODITIES LOGIC ---
        elif "Commodity" in asset_class:
            # Priced globally against the USD. INVERT the USD score.
            raw_score, c_st, c_pr, c_su = calculate_macro_score("USD", struct_data, proxy_data, surprise_data, has_surprise_flags)
            final_score = -raw_score
            
            details = {
                "Pricing Base": "USD Anti-Dollar Correlation",
                "Structural Macro (Inverted)": round(-c_st, 1),
                "Proxy Momentum (Inverted)": round(-c_pr, 1),
                "Event Surprise (Inverted)": round(-c_su, 1) if has_surprise_flags.get("USD") else "Gathering Data",
            }
        else:
            return 0, {"⚠️ STATUS": "Unsupported Asset Class"}

        return max(-100, min(100, final_score)), details

    except Exception as e:
        return 0, {"⚠️ STATUS": f"Macro API Failure: {str(e)}"}