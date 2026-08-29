import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas_datareader.data as web
import datetime
import pysentiment2 as ps
from trading_ig import IGService
import json
import os
import time

# ============================================================
# 1. PAGE SETUP & PERSISTENCE
# ============================================================
st.set_page_config(page_title="Quant Trade Engine", layout="wide")

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f: return json.load(f)
        except Exception: return {}
    return {}

def save_watchlist(data):
    with open(WATCHLIST_FILE, "w") as f: json.dump(data, f, indent=4)

if "custom_watchlist" not in st.session_state:
    st.session_state.custom_watchlist = load_watchlist()

# ============================================================
# 2. DICTIONARIES & MAPPINGS
# ============================================================
INSTRUMENTS = {
    "Forex (Majors)": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", 
        "USD/CHF": "USDCHF=X", "USD/CAD": "USDCAD=X", "AUD/USD": "AUDUSD=X", "NZD/USD": "NZDUSD=X"
    },
    "Forex (Minors & Crosses)": {
        "EUR/GBP": "EURGBP=X", "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X", "EUR/CHF": "EURCHF=X", 
        "AUD/JPY": "AUDJPY=X", "EUR/AUD": "EURAUD=X", "GBP/CHF": "GBPCHF=X", "CAD/JPY": "CADJPY=X", 
        "NZD/JPY": "NZDJPY=X", "AUD/NZD": "AUDNZD=X", "AUD/CAD": "AUDCAD=X", "AUD/CHF": "AUDCHF=X", 
        "CAD/CHF": "CADCHF=X", "EUR/CAD": "EURCAD=X", "EUR/NZD": "EURNZD=X", "GBP/AUD": "GBPAUD=X", 
        "GBP/CAD": "GBPCAD=X", "GBP/NZD": "GBPNZD=X", "NZD/CAD": "NZDCAD=X", "NZD/CHF": "NZDCHF=X"
    },
    "Global Stock Indices": {
        "US Dollar Index (DXY)": "DX-Y.NYB", "US 30 (Dow Jones)": "^DJI", "US 500 (S&P 500)": "^GSPC",
        "US Tech 100 (Nasdaq)": "^NDX", "US 2000 (Russell 2000)": "^RUT", "VIX (Volatility Index)": "^VIX",
        "UK 100 (FTSE)": "^FTSE", "Germany 40 (DAX)": "^GDAXI", "France 40 (CAC)": "^FCHI",
        "Europe 50 (Euro Stoxx)": "^STOXX50E", "Japan 225 (Nikkei)": "^N225",
        "Hong Kong 50 (Hang Seng)": "^HSI", "Australia 200 (ASX)": "^AXJO"
    },
    "Precious Metals & Commodities": {
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F", "Platinum": "PL=F", 
        "Palladium": "PA=F", "Zinc": "ZNC=F", "Crude Oil (WTI)": "CL=F", 
        "Brent Crude": "BZ=F", "Natural Gas": "NG=F"
    },
    "Treasury Bonds & Notes": {
        "US 10-Year T-Note (Yield)": "^TNX", "US 10-Year T-Note (Futures)": "ZN=F",
        "US 30-Year T-Bond (Yield)": "^TYX", "US 30-Year T-Bond (Futures)": "ZB=F",
        "US 5-Year T-Note (Yield)": "^FVX", "US 5-Year T-Note (Futures)": "ZF=F", "US 2-Year T-Note (Futures)": "ZT=F"
    },
    "Crypto": {
        "BTC/USD (Bitcoin)": "BTC-USD", "ETH/USD (Ethereum)": "ETH-USD", "SOL/USD (Solana)": "SOL-USD",
        "XRP/USD (Ripple)": "XRP-USD", "ADA/USD (Cardano)": "ADA-USD", "DOGE/USD (Dogecoin)": "DOGE-USD",
        "LINK/USD (Chainlink)": "LINK-USD", "DOT/USD (Polkadot)": "DOT-USD", "LTC/USD (Litecoin)": "LTC-USD",
        "BCH/USD (Bitcoin Cash)": "BCH-USD", "AVAX/USD (Avalanche)": "AVAX-USD", "MATIC/USD (Polygon)": "MATIC-USD",
        "UNI/USD (Uniswap)": "UNI7083-USD", "XLM/USD (Stellar)": "XLM-USD", "ATOM/USD (Cosmos)": "ATOM-USD"
    }
}

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
    "US Dollar Index (DXY)": "USD", "US 30 (Dow Jones)": "USD", "US 500 (S&P 500)": "USD",
    "US Tech 100 (Nasdaq)": "USD", "US 2000 (Russell 2000)": "USD", "VIX (Volatility Index)": "USD",
    "UK 100 (FTSE)": "GBP", "Germany 40 (DAX)": "EUR", "France 40 (CAC)": "EUR",
    "Europe 50 (Euro Stoxx)": "EUR", "Japan 225 (Nikkei)": "JPY",
    "Hong Kong 50 (Hang Seng)": "USD", "Australia 200 (ASX)": "AUD"
}

# ============================================================
# 3. TECHNICAL ANALYSIS ENGINE
# ============================================================
@st.cache_data(ttl=3600)
def calculate_daily_trend_score(ticker_symbol):
    try:
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="1y")
        
        # --- THE FIX: Scrub Yahoo Finance NaN glitches ---
        if not df.empty:
            df = df.dropna(subset=['Close'])
        # -------------------------------------------------
        
        if df.empty or len(df) < 200:
            return 0, {"⚠️ STATUS": "Insufficient History"} 
            
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        # ... rest of the code remains identical ...
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))
        
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema_12 - ema_26
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        current = df.iloc[-1]
        close, ema_20, sma_50, sma_200, rsi_14, macd_hist = current['Close'], current['EMA_20'], current['SMA_50'], current['SMA_200'], current['RSI_14'], current['MACDh_12_26_9']

        score = 0
        score += 10 if close > ema_20 else -10
        score += 10 if close > sma_50 else -10
        score += 10 if close > sma_200 else -10
        
        if ema_20 > sma_50 and sma_50 > sma_200: score += 20
        elif ema_20 < sma_50 and sma_50 < sma_200: score -= 20
            
        if rsi_14 > 70: score -= 25 
        elif rsi_14 < 30: score += 25 
        elif rsi_14 > 50: score += 15 
        else: score -= 15 
            
        if macd_hist > 0: score += 25
        else: score -= 25
            
        details = {
            "Close Price": round(close, 4), "EMA 20": round(ema_20, 4), "SMA 50": round(sma_50, 4),
            "SMA 200": round(sma_200, 4), "RSI (14)": round(rsi_14, 2), "MACD Hist": round(macd_hist, 4)
        }
        return max(-100, min(100, score)), details
    except Exception:
        return 0, {"⚠️ STATUS": "Technical API Failure"}
    
@st.cache_data(ttl=1800)
def get_4h_indicators(ticker_symbol):
    try:
        asset = yf.Ticker(ticker_symbol)
        df_1h = asset.history(period="60d", interval="1h") 
        if df_1h.empty: return None

        df = df_1h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        if len(df) < 60: return None

        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))

        macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

        tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()

        return df.dropna()
    except Exception:
        return None

def calculate_4h_momentum_score(df4h, daily_direction):
    if df4h is None or len(df4h) < 5 or daily_direction == 0: return 0
    c, prev, score = df4h.iloc[-1], df4h.iloc[-2], 0

    if daily_direction > 0:
        if c['Close'] > c['EMA_20'] and c['EMA_20'] > prev['EMA_20']: score += 30
        if c['RSI_14'] > 55 and c['RSI_14'] > prev['RSI_14']: score += 30
        if c['MACD_hist'] > 0 and c['MACD_hist'] > prev['MACD_hist']: score += 40
    else:
        if c['Close'] < c['EMA_20'] and c['EMA_20'] < prev['EMA_20']: score += 30
        if c['RSI_14'] < 45 and c['RSI_14'] < prev['RSI_14']: score += 30
        if c['MACD_hist'] < 0 and c['MACD_hist'] < prev['MACD_hist']: score += 40
        score = -score
    return max(-100, min(100, score))

def calculate_4h_pullback_score(df4h, daily_direction):
    if df4h is None or len(df4h) < 5 or daily_direction == 0: return 0
    c, prev, atr = df4h.iloc[-1], df4h.iloc[-2], df4h.iloc[-1]['ATR_14']
    if atr == 0 or pd.isna(atr): return 0

    dist_from_ema20_atr = (c['Close'] - c['EMA_20']) / atr
    score = 0

    if daily_direction > 0: 
        if -1.5 <= dist_from_ema20_atr <= 0.5: score += 30
        if 30 <= c['RSI_14'] <= 50: score += 30
        if c['RSI_14'] > prev['RSI_14']: score += 20
        if c['MACD_hist'] > prev['MACD_hist']: score += 20
    else: 
        if -0.5 <= dist_from_ema20_atr <= 1.5: score += 30
        if 50 <= c['RSI_14'] <= 70: score += 30
        if c['RSI_14'] < prev['RSI_14']: score += 20
        if c['MACD_hist'] < prev['MACD_hist']: score += 20
        score = -score
    return max(-100, min(100, score))

@st.cache_data(ttl=1800)
def calculate_technical_score(ticker_symbol):
    try:
        daily_score, daily_details = calculate_daily_trend_score(ticker_symbol)
        daily_direction = 1 if daily_score > 0 else (-1 if daily_score < 0 else 0)

        df4h = get_4h_indicators(ticker_symbol)
        momentum_score = calculate_4h_momentum_score(df4h, daily_direction)
        pullback_score = calculate_4h_pullback_score(df4h, daily_direction)

        if abs(momentum_score) >= abs(pullback_score):
            four_h_score, active_mode = momentum_score, ("Momentum" if momentum_score != 0 else "Neutral")
        else:
            four_h_score, active_mode = pullback_score, "Pullback"

        final_score = (daily_score * 0.7) + (four_h_score * 0.3)
        details = {
            **daily_details, 
            "4H Trading Mode": active_mode, "4H Momentum Score": round(momentum_score, 1),
            "4H Pullback Score": round(pullback_score, 1), "4H Trigger Component": round(four_h_score, 1),
        }
        return max(-100, min(100, final_score)), details
    except Exception as e:
        return 0, {"⚠️ STATUS": f"Technical API Failure: {str(e)}"}

# ============================================================
# 4. SEASONALITY ENGINE
# ============================================================
@st.cache_data(ttl=86400)
def calculate_seasonality_score(ticker_symbol):
    try:
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="10y", interval="1mo")
        if df.empty: return 0, {"⚠️ STATUS": "No Seasonality Data"}
        
        current_month = datetime.datetime.now().month
        df['Returns'] = df['Close'].pct_change()
        monthly_data = df[df.index.month == current_month]['Returns'].dropna()
        if monthly_data.empty: return 0, {"⚠️ STATUS": "No Monthly Data"}
            
        avg_return = monthly_data.mean() * 100
        score = (avg_return / 2.0) * 100
        return max(-100, min(100, score)), {"Avg Monthly Return": round(avg_return, 2)}
    except Exception:
        return 0, {"⚠️ STATUS": "Seasonality API Failure"}

# ============================================================
# 5. SENTIMENT & COT ENGINE (4-PILLAR HYBRID WITH 5-DAY PCR)
# ============================================================
COT_MAPPING = {
    # US Stock Indices & Nikkei
    "US 500 (S&P 500)": {"code": "13874A", "invert": False},
    "US Tech 100 (Nasdaq)": {"code": "209742", "invert": False},
    "US 30 (Dow Jones)": {"code": "124603", "invert": False},
    "US 2000 (Russell 2000)": {"code": "239742", "invert": False},
    "Japan 225 (Nikkei)": {"code": "240741", "invert": False},
    
    # Commodities (COMEX / NYMEX)
    "Gold": {"code": "088691", "invert": False},
    "Silver": {"code": "084691", "invert": False},
    "Copper": {"code": "085692", "invert": False},
    "Platinum": {"code": "076651", "invert": False},
    "Palladium": {"code": "075651", "invert": False},
    "Crude Oil (WTI)": {"code": "067651", "invert": False},
    "Natural Gas": {"code": "023651", "invert": False}
}

CURRENCY_COT_MAPPING = {
    "EUR": "099741", "GBP": "096742", "JPY": "097741", "CHF": "092741", 
    "CAD": "090741", "AUD": "232741", "NZD": "112741"
}

ETF_OPTIONS_MAPPING = {
    # Stock Indices
    "US 500 (S&P 500)": "SPY", "US Tech 100 (Nasdaq)": "QQQ", "US 30 (Dow Jones)": "DIA",
    "US 2000 (Russell 2000)": "IWM", "UK 100 (FTSE)": "EWU", "Germany 40 (DAX)": "EWG",
    "France 40 (CAC)": "EWQ", "Europe 50 (Euro Stoxx)": "FEZ", "Japan 225 (Nikkei)": "EWJ",
    "Hong Kong 50 (Hang Seng)": "EWH", "Australia 200 (ASX)": "EWA",
    
    # Commodities
    "Gold": "GLD", "Silver": "SLV", "Crude Oil (WTI)": "USO", "Brent Crude": "BNO",
    "Natural Gas": "UNG", "Copper": "CPER", "Platinum": "PPLT", "Palladium": "PALL", "Zinc": "DBB"
}

IG_SENTIMENT_MAPPING = {
    "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY", "USD/CHF": "USDCHF", "USD/CAD": "USDCAD",
    "AUD/USD": "AUDUSD", "NZD/USD": "NZDUSD", "EUR/GBP": "EURGBP", "EUR/JPY": "EURJPY", "GBP/JPY": "GBPJPY",
    "EUR/CHF": "EURCHF", "AUD/JPY": "AUDJPY", "EUR/AUD": "EURAUD", "GBP/CHF": "GBPCHF", "CAD/JPY": "CADJPY",
    "NZD/JPY": "NZDJPY", "AUD/NZD": "AUDNZD", "AUD/CAD": "AUDCAD", "AUD/CHF": "AUDCHF", "CAD/CHF": "CADCHF",
    "EUR/CAD": "EURCAD", "EUR/NZD": "EURNZD", "GBP/AUD": "GBPAUD", "GBP/CAD": "GBPCAD", "GBP/NZD": "GBPNZD",
    "NZD/CAD": "NZDCAD", "NZD/CHF": "NZDCHF", "US 30 (Dow Jones)": "WALL", "US 500 (S&P 500)": "US500",
    "US Tech 100 (Nasdaq)": "USTECH", "US 2000 (Russell 2000)": "R2000", "UK 100 (FTSE)": "FT100",
    "Germany 40 (DAX)": "DE30", "France 40 (CAC)": "FR40", "Europe 50 (Euro Stoxx)": "EU50",
    "Japan 225 (Nikkei)": "JP225", "Hong Kong 50 (Hang Seng)": "HS34", "Australia 200 (ASX)": "AU200",
    "Gold": "GC", "Silver": "SI", "Copper": "HG", "Platinum": "PL", "Palladium": "PA", "Zinc": "ZNC",
    "Crude Oil (WTI)": "CL", "Brent Crude": "LCO", "Natural Gas": "NG", "BTC/USD (Bitcoin)": "BITCOIN",
    "ETH/USD (Ethereum)": "ETHER", "XRP/USD (Ripple)": "RIPPLE", "LTC/USD (Litecoin)": "LITECOIN",
    "BCH/USD (Bitcoin Cash)": "BITCOINCASH", "ADA/USD (Cardano)": "CARDANO", "DOGE/USD (Dogecoin)": "DOGECOIN",
    "LINK/USD (Chainlink)": "CHAINLINK", "DOT/USD (Polkadot)": "POLKADOT", "SOL/USD (Solana)": "SOLANA",
    "AVAX/USD (Avalanche)": "AVALANCHE", "MATIC/USD (Polygon)": "POLYGON", "UNI/USD (Uniswap)": "UNISWAP",
    "XLM/USD (Stellar)": "STELLAR", "ATOM/USD (Cosmos)": "COSMOS"
}

def get_cftc_score(cftc_code):
    try:
        url = f"https://publicreporting.cftc.gov/resource/6dca-aqww.json?cftc_contract_market_code={cftc_code}&$order=report_date_as_yyyy_mm_dd DESC&$limit=2"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) == 2:
                longs_curr, shorts_curr = float(data[0].get('noncomm_positions_long_all', 0)), float(data[0].get('noncomm_positions_short_all', 0))
                total_curr = longs_curr + shorts_curr
                if total_curr == 0: return None
                
                net_curr = longs_curr - shorts_curr
                longs_prev, shorts_prev = float(data[1].get('noncomm_positions_long_all', 0)), float(data[1].get('noncomm_positions_short_all', 0))
                net_prev = longs_prev - shorts_prev
                
                abs_score = (net_curr / total_curr) * 50.0
                momentum_score = max(-50.0, min(50.0, ((net_curr - net_prev) / total_curr) * 250.0))
                return max(-100.0, min(100.0, abs_score + momentum_score))
    except Exception: pass
    return None

@st.cache_resource
def get_yf_session():
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    })
    try:
        session.get("https://finance.yahoo.com", timeout=5)
    except Exception:
        pass
    return session

def get_put_call_ratio(etf_ticker, asset_category="Index", min_threshold=30):
    try:
        import yfinance as yf
        import json
        import os
        import datetime
        
        PCR_DB_FILE = "pcr_history.json"
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        
        session = get_yf_session()
        asset = yf.Ticker(etf_ticker, session=session)
        expirations = asset.options
        
        if not expirations:
            asset = yf.Ticker(etf_ticker)
            expirations = asset.options
            
        if not expirations:
            return None
            
        total_oi_put, total_oi_call = 0.0, 0.0
        total_vol_put, total_vol_call = 0.0, 0.0
        
        for exp in expirations[:2]:
            try:
                chain = asset.option_chain(exp)
                if hasattr(chain, 'puts') and 'openInterest' in chain.puts.columns:
                    total_oi_put += chain.puts['openInterest'].fillna(0).sum()
                    total_vol_put += chain.puts['volume'].fillna(0).sum()
                if hasattr(chain, 'calls') and 'openInterest' in chain.calls.columns:
                    total_oi_call += chain.calls['openInterest'].fillna(0).sum()
                    total_vol_call += chain.calls['volume'].fillna(0).sum()
            except Exception:
                continue

        # 1. Calculate Today's Raw PCR
        pcr_oi_today, pcr_vol_today = None, None
        if total_oi_call > 0 and (total_oi_put + total_oi_call) >= min_threshold:
            pcr_oi_today = total_oi_put / total_oi_call
        if total_vol_call > 0 and (total_vol_put + total_vol_call) >= min_threshold:
            pcr_vol_today = total_vol_put / total_vol_call

        if pcr_oi_today is None and pcr_vol_today is None:
            return None

        # 2. Local Database: 5-Day Rolling Storage
        if os.path.exists(PCR_DB_FILE):
            try:
                with open(PCR_DB_FILE, 'r') as f:
                    pcr_db = json.load(f)
            except Exception: pcr_db = {}
        else:
            pcr_db = {}

        if etf_ticker not in pcr_db:
            pcr_db[etf_ticker] = {}

        # Log today's data (defaults to neutral baseline if one metric is missing)
        pcr_db[etf_ticker][today_str] = {
            "oi": pcr_oi_today if pcr_oi_today else 0.85,
            "vol": pcr_vol_today if pcr_vol_today else 0.85
        }

        # Purge dates older than 5 trading days
        sorted_dates = sorted(pcr_db[etf_ticker].keys())
        if len(sorted_dates) > 5:
            for d in sorted_dates[:-5]:
                del pcr_db[etf_ticker][d]

        # Save to disk
        try:
            with open(PCR_DB_FILE, 'w') as f:
                json.dump(pcr_db, f, indent=4)
        except Exception: pass

        # 3. Calculate the Moving Averages
        valid_dates = sorted(pcr_db[etf_ticker].keys())
        avg_pcr_oi = sum(pcr_db[etf_ticker][d]["oi"] for d in valid_dates) / len(valid_dates)
        avg_pcr_vol = sum(pcr_db[etf_ticker][d]["vol"] for d in valid_dates) / len(valid_dates)

        # 4. Dynamic Dead-Band Scoring
        if asset_category == "Commodity":
            lower_bound, upper_bound = 0.55, 0.95
        else:
            lower_bound, upper_bound = 0.70, 1.10

        def score_contrarian_band(pcr_sma):
            if lower_bound <= pcr_sma <= upper_bound:
                return 0.0
            elif pcr_sma > upper_bound:
                # Extreme Fear -> Contrarian Bullish
                return min(100.0, (pcr_sma - upper_bound) * 150.0)
            else:
                # Extreme Greed -> Contrarian Bearish
                return max(-100.0, -(lower_bound - pcr_sma) * 285.0)

        score_oi = score_contrarian_band(avg_pcr_oi) if pcr_oi_today else None
        score_vol = score_contrarian_band(avg_pcr_vol) if pcr_vol_today else None

        if score_oi is not None and score_vol is not None:
            return (score_oi * 0.50) + (score_vol * 0.50)
        elif score_oi is not None:
            return score_oi
        elif score_vol is not None:
            return score_vol
            
        return None
        
    except Exception as e:
        print(f"[{etf_ticker}] Options PCR Error: {e}")
        return None

@st.cache_resource(ttl=43200)
def get_ig_session():
    try:
        ig_service = IGService(
            st.secrets["ig_markets"]["username"], st.secrets["ig_markets"]["password"],
            st.secrets["ig_markets"]["api_key"], st.secrets["ig_markets"]["acc_type"]
        )
        ig_service.create_session()
        return ig_service
    except Exception: return None

@st.cache_data(ttl=3600)
def get_ig_retail_sentiment(instrument_name, _ig_service):
    try:
        if _ig_service is None: return None
        market_id = IG_SENTIMENT_MAPPING.get(instrument_name)
        if not market_id: return None

        sentiment = _ig_service.fetch_client_sentiment_by_instrument(market_id)
        long_pct, short_pct = float(sentiment.get('longPositionPercentage', 0)), float(sentiment.get('shortPositionPercentage', 0))

        if long_pct == 0 and short_pct == 0: return None
        return -(long_pct - short_pct) 
    except Exception: return None

@st.cache_data(ttl=3600)
def calculate_sentiment_score(ticker_symbol, name):
    try:
        # 1. News Sentiment
        news_score = None
        try:
            clean_name = name.split("(")[0].strip()
            rss_url = f"https://news.google.com/rss/search?q={clean_name.replace(' ', '+')}+market+news&hl=en-US&gl=US&ceid=US:en"
            soup = BeautifulSoup(requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).content, features="xml")
            headlines = soup.find_all("title")
            
            lm = ps.LM()
            total_polarity, count, matched_count, seen_headlines = 0, 0, 0, set()
            
            for headline in headlines[1:16]:
                text = headline.text
                if text.lower() in seen_headlines: continue
                seen_headlines.add(text.lower())
                
                score_dict = lm.get_score(lm.tokenize(text))
                if score_dict.get('Positive', 0) > 0 or score_dict.get('Negative', 0) > 0: matched_count += 1
                total_polarity += score_dict.get('Polarity', 0)
                count += 1
                
            if count > 0:
                scaled_score = (total_polarity / count) * 225.0
                density_ratio = matched_count / count
                if density_ratio < 0.15: scaled_score *= (density_ratio / 0.15)
                news_score = max(-100, min(100, scaled_score))
        except Exception: pass

        # 2. Smart Money (COT Positioning)
        cot_score = None
        if "/" in name:
            base, quote = name.split("/")
            b_code, q_code = CURRENCY_COT_MAPPING.get(base), CURRENCY_COT_MAPPING.get(quote)
            b_score, q_score = get_cftc_score(b_code) if b_code else None, get_cftc_score(q_code) if q_code else None
            
            if quote == "USD" and b_score is not None: cot_score = b_score
            elif base == "USD" and q_score is not None: cot_score = -q_score 
            elif b_score is not None and q_score is not None: cot_score = (b_score - q_score) / 2 
        elif name in COT_MAPPING:
            cftc_info = COT_MAPPING[name]
            raw_score = get_cftc_score(cftc_info["code"])
            if raw_score is not None: cot_score = -raw_score if cftc_info["invert"] else raw_score

        # 3. Smart Money (Put/Call Ratio with Dead-Band & 5-Day SMA)
        pcr_score = None
        if name in ETF_OPTIONS_MAPPING:
            category = "Commodity" if any(c in name for c in ["Gold", "Silver", "Crude Oil", "Brent Crude", "Natural Gas", "Copper", "Platinum", "Palladium", "Zinc"]) else "Index"
            pcr_score = get_put_call_ratio(ETF_OPTIONS_MAPPING[name], asset_category=category)

        # 4. Retail Sentiment (IG Contrarian)
        retail_score = get_ig_retail_sentiment(name, get_ig_session())

        # Master 4-Pillar Dynamic Averaging
        available_scores = [s for s in (news_score, cot_score, pcr_score, retail_score) if s is not None]
        final_score = sum(available_scores) / len(available_scores) if available_scores else 0 

        details = {
            "News (Loughran-McDonald)": round(news_score, 2) if news_score is not None else "No Data",
            "Smart Money (COT)": round(cot_score, 2) if cot_score is not None else "No Data",
            "Smart Money (Put/Call)": round(pcr_score, 2) if pcr_score is not None else "No Data",
            "Retail Sentiment (IG)": round(retail_score, 2) if retail_score is not None else "No Data"
        }
        return max(-100, min(100, final_score)), details
    except Exception:
        return 0, {"⚠️ STATUS": "Sentiment API Failure"}

# ============================================================
# 6. FUNDAMENTALS ENGINE (MACRO TRIAD HYBRID)
# ============================================================
FF_DB_FILE = "ff_history.json"
COOLDOWN_SECONDS = 900

@st.cache_data(ttl=3600)
def get_structural_macro():
    scores = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=730)
        all_tickers = [val for data in FRED_MACRO_TICKERS.values() for val in data.values()]
        df = web.DataReader(all_tickers, 'fred', start, end)
        
        for currency, indicators in FRED_MACRO_TICKERS.items():
            score = 0
            rate_series = df[indicators['Rate']].dropna()
            if not rate_series.empty: score += (rate_series.iloc[-1] - 2.0) * 5.0
            
            cpi_series = df[indicators['CPI']].dropna()
            if len(cpi_series) >= 14:
                latest_yoy = ((cpi_series.iloc[-1] - cpi_series.iloc[-13]) / cpi_series.iloc[-13]) * 100
                prev_yoy = ((cpi_series.iloc[-2] - cpi_series.iloc[-14]) / cpi_series.iloc[-14]) * 100
                score += (latest_yoy - 2.0) * 5.0 + (latest_yoy - prev_yoy) * 25.0
            scores[currency] = max(-100, min(100, score))
    except Exception: pass
    return scores

@st.cache_data(ttl=3600)
def get_proxy_momentum():
    scores = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}
    try:
        df = yf.download(["^TNX", "^VIX", "GC=F", "CL=F"], period="1mo", interval="1d", progress=False)['Close'].ffill().bfill()
        if len(df) >= 5:
            roc = ((df.iloc[-1] - df.iloc[-5]) / df.iloc[-5]) * 100
            v, y, g, o = roc.get('^VIX', 0), roc.get('^TNX', 0), roc.get('GC=F', 0), roc.get('CL=F', 0)
            
            scores.update({
                "USD": (y * 3.0) - (v * 0.5), "JPY": (v * 1.5) - (y * 2.0), "CHF": v * 1.5,
                "CAD": (o * 2.0) - v, "AUD": (g * 2.0) - (v * 1.5), "NZD": (g * 1.5) - (v * 1.5),
                "EUR": -(v * 0.5), "GBP": -(v * 0.5)
            })
            for k in scores: scores[k] = max(-100, min(100, scores[k]))
    except Exception: pass
    return scores

def update_and_get_ff_surprises():
    history, live_fetch_needed = {}, True
    if os.path.exists(FF_DB_FILE):
        try:
            with open(FF_DB_FILE, 'r') as f: history = json.load(f)
            if time.time() - os.path.getmtime(FF_DB_FILE) < COOLDOWN_SECONDS: live_fetch_needed = False
        except: pass

    if live_fetch_needed:
        try:
            resp = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if resp.status_code == 200:
                for event in resp.json():
                    if event.get('impact') == 'High':
                        eid = f"{str(event.get('date', '')).split('T')[0]}_{event.get('country')}_{event.get('title')}"
                        if eid not in history or event.get('actual'): history[eid] = event
                with open(FF_DB_FILE, 'w') as f: json.dump(history, f, indent=4)
        except Exception: pass

    scores, counts, has_data = {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}, {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}, {}
    thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
    
    def parse_econ_val(val_str):
        if val_str in [None, "", "None"]: return None
        v = str(val_str).strip().replace(',', '').lower()
        if '%' in v:
            try: return float(v.replace('%', ''))
            except: return None
        mult = 1.0
        if v.endswith('k'): mult, v = 1000.0, v[:-1]
        elif v.endswith('m'): mult, v = 1000000.0, v[:-1]
        elif v.endswith('b'): mult, v = 1000000000.0, v[:-1]
        try: return float(v) * mult
        except: return None

    for eid, event in history.items():
        try:
            if datetime.datetime.strptime(eid.split('_')[0], '%Y-%m-%d') < thirty_days_ago: continue
            curr = event.get('country')
            if curr not in scores: continue
            actual, forecast = parse_econ_val(event.get('actual')), parse_econ_val(event.get('forecast'))
            if actual is not None and forecast is not None:
                surprise = ((actual - forecast) / (abs(forecast) + 0.0001)) * 100
                if any(k in str(event.get('title')).lower() for k in ['unemployment', 'claims', 'claimant']): surprise = -surprise
                scores[curr] += max(-50, min(50, surprise))
                counts[curr] += 1
        except: continue
            
    for k in scores:
        if counts[k] > 0: scores[k], has_data[k] = max(-100, min(100, scores[k] / counts[k])), True
        else: has_data[k] = False
    return scores, has_data

def calculate_macro_score(currency, struct_data, proxy_data, surprise_data, has_surprise_flags):
    active = [struct_data.get(currency, 0), proxy_data.get(currency, 0)]
    if has_surprise_flags.get(currency, False): active.append(surprise_data.get(currency, 0))
    return sum(active) / len(active), struct_data.get(currency, 0), proxy_data.get(currency, 0), surprise_data.get(currency, 0)

@st.cache_data(ttl=3600)
def calculate_fundamental_score(name, asset_class):
    try:
        struct_data, proxy_data, (surprise_data, has_flags) = get_structural_macro(), get_proxy_momentum(), update_and_get_ff_surprises()
        
        if "Forex" in asset_class:
            b, q = name.split("/")
            b_sc, b_st, b_pr, b_su = calculate_macro_score(b, struct_data, proxy_data, surprise_data, has_flags)
            q_sc, q_st, q_pr, q_su = calculate_macro_score(q, struct_data, proxy_data, surprise_data, has_flags)
            return max(-100, min(100, b_sc - q_sc)), {
                f"{b} Struct.": round(b_st, 1), f"{b} Proxy": round(b_pr, 1), f"{b} Surp.": round(b_su, 1) if has_flags.get(b) else "Gathering",
                f"{q} Struct.": round(q_st, 1), f"{q} Proxy": round(q_pr, 1), f"{q} Surp.": round(q_su, 1) if has_flags.get(q) else "Gathering"
            }
        elif "Indices" in asset_class or "Index" in name:
            c = INDEX_TO_CURRENCY.get(name, "USD")
            sc, st, pr, su = calculate_macro_score(c, struct_data, proxy_data, surprise_data, has_flags)
            return max(-100, min(100, -sc)), {
                "Host Currency": c, "Struct. (Inv)": round(-st, 1), "Proxy (Inv)": round(-pr, 1), "Surp. (Inv)": round(-su, 1) if has_flags.get(c) else "Gathering"
            }
        else:
            sc, st, pr, su = calculate_macro_score("USD", struct_data, proxy_data, surprise_data, has_flags)
            return max(-100, min(100, -sc)), {
                "Pricing Base": "USD Anti-Dollar", "Struct. (Inv)": round(-st, 1), "Proxy (Inv)": round(-pr, 1), "Surp. (Inv)": round(-su, 1) if has_flags.get("USD") else "Gathering"
            }
    except Exception as e:
        return 0, {"⚠️ STATUS": f"Macro API Failure: {str(e)}"}

# ============================================================
# 7. UI NAVIGATION & SETUP ICON HELPER
# ============================================================
with st.sidebar:
    st.title("⚙️ Trading Engine")
    universe_options = ["⭐ Custom Watchlist"] + list(INSTRUMENTS.keys())
    asset_class = st.selectbox("Select Asset Class", universe_options)
    if asset_class == "⭐ Custom Watchlist": target_instruments = st.session_state.custom_watchlist
    else: target_instruments = INSTRUMENTS[asset_class]
    st.info(f"Target universe contains **{len(target_instruments)}** equities.")
    
    st.markdown("---")
    st.caption("**Setup Icon Legend**")
    st.caption("🚀 Bullish Momentum &nbsp;&nbsp; 🎯 Bullish Pullback")
    st.caption("⚡ Bearish Momentum &nbsp;&nbsp; 🎯 Bearish Pullback")
    st.caption("➖ No Clear 4H Setup")

st.title(f"📊 Market Screener: {asset_class}")
st.divider()

def get_setup_icon(tech_score, tech_details):
    mode = tech_details.get("4H Trading Mode", "Neutral")
    if mode == "Pullback": return "🎯"
    elif tech_score > 0 and mode == "Momentum": return "🚀"
    elif tech_score < 0 and mode == "Momentum": return "⚡"
    else: return "➖"

# ============================================================
# 8. LIVE DATA SCANNER (The Loop)
# ============================================================
if "last_scanned_asset" not in st.session_state or st.session_state.last_scanned_asset != asset_class:
    
    if len(target_instruments) == 0:
        st.warning("Your Custom Watchlist is empty. Select a standard sector to scan and add some instruments!")
        st.session_state.scanned_data, st.session_state.breakdown_data = [], {}
    else:
        scanned_data, breakdown_data = [], {}
        my_bar = st.progress(0, text="Scanning live markets...")

        for i, (name, ticker) in enumerate(target_instruments.items()):
            tech_score, tech_details = calculate_technical_score(ticker)
            seas_score, seas_details = calculate_seasonality_score(ticker)
            sent_score, sent_details = calculate_sentiment_score(ticker, name)
            fund_score, fund_details = calculate_fundamental_score(name, asset_class)
            
            master_score = (tech_score * 0.30) + (fund_score * 0.30) + (sent_score * 0.30) + (seas_score * 0.10)
            
            if master_score >= 50: bias_label = "🔥 Very Bullish"
            elif master_score >= 15: bias_label = "📈 Bullish"
            elif master_score > -15: bias_label = "⚖️ Neutral"
            elif master_score > -50: bias_label = "📉 Bearish"
            else: bias_label = "❄️ Very Bearish"

            scanned_data.append({
                "Setup": get_setup_icon(tech_score, tech_details),
                "Instrument": name,
                "Master Score": round(master_score, 1),
                "Bias Status": bias_label,
                "Technicals (30%)": int(tech_score),
                "Fundamentals (30%)": int(fund_score),
                "Sentiment (30%)": int(sent_score),
                "Seasonality (10%)": int(seas_score)
            })
            
            breakdown_data[name] = {
                "Ticker": ticker, "Technicals": tech_details, "Fundamentals": fund_details,
                "Sentiment": sent_details, "Seasonality": seas_details
            }
            
            my_bar.progress((i + 1) / len(target_instruments))
            time.sleep(0.8) 

        my_bar.empty()
        st.session_state.scanned_data, st.session_state.breakdown_data = scanned_data, breakdown_data
    
    st.session_state.last_scanned_asset = asset_class

# ============================================================
# 9. DATAFRAME & UI DRILL-DOWN
# ============================================================
df = pd.DataFrame(st.session_state.scanned_data)
if not df.empty: df = df.sort_values(by="Master Score", ascending=False).reset_index(drop=True)
breakdown_data = st.session_state.breakdown_data

def color_scores(val):
    if isinstance(val, (int, float)):
        if val > 0: return 'color: #00FF00; font-weight: bold;' 
        elif val < 0: return 'color: #FF4136; font-weight: bold;' 
        else: return 'color: gray;' 
    return ''

score_cols = ["Master Score", "Technicals (30%)", "Fundamentals (30%)", "Sentiment (30%)", "Seasonality (10%)"]

if not df.empty:
    styled_df = df.style.map(color_scores, subset=score_cols).format("{:.1f}", subset=["Master Score"]) 
    event = st.dataframe(styled_df, width="stretch", on_select="rerun", selection_mode="single-row")
    selected_rows = event.selection.rows
else: selected_rows = []

if selected_rows:
    selected_instrument = df.iloc[selected_rows[0]]["Instrument"]
    details = breakdown_data[selected_instrument]
    ticker = details["Ticker"]
    
    st.divider()
    col_title, col_button = st.columns([3, 1])
    with col_title: st.subheader(f"🔍 Deep Dive: {selected_instrument}")
    with col_button:
        if selected_instrument in st.session_state.custom_watchlist:
            if st.button("❌ Remove from Watchlist", use_container_width=True):
                del st.session_state.custom_watchlist[selected_instrument]
                save_watchlist(st.session_state.custom_watchlist)
                if asset_class == "⭐ Custom Watchlist": st.session_state.last_scanned_asset = None
                st.rerun()
        else:
            if st.button("⭐ Add to Watchlist", type="primary", use_container_width=True):
                st.session_state.custom_watchlist[selected_instrument] = ticker
                save_watchlist(st.session_state.custom_watchlist)
                st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📈 Technical Analysis", expanded=True):
            t_cols = st.columns(3)
            for idx, (key, val) in enumerate(details["Technicals"].items()): 
                t_cols[idx % 3].metric(label=key, value=val)
                
        with st.expander("🌍 Fundamental Macro"):
            f_cols = st.columns(2)
            for idx, (key, val) in enumerate(details["Fundamentals"].items()): 
                f_cols[idx % 2].metric(label=key, value=val)
                
    with col2:
        with st.expander("🧠 Sentiment & COT", expanded=True):
            s_cols = st.columns(4)
            for idx, (key, val) in enumerate(details["Sentiment"].items()): 
                s_cols[idx % 4].metric(label=key, value=val)
                
        with st.expander("📅 Seasonality"):
            st.metric(label="Average Monthly Return", value=f"{details['Seasonality'].get('Avg Monthly Return', 0)}%")