import streamlit as st
import pandas as pd
import random
import yfinance as yf

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas_datareader.data as web
import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="Quant Trade Engine", layout="wide")

# 2. THE EXHAUSTIVE VANTAGE MARKETS DICTIONARY
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
        "US Dollar Index (DXY)": "DX-Y.NYB",
        "US 30 (Dow Jones)": "^DJI",
        "US 500 (S&P 500)": "^GSPC",
        "US Tech 100 (Nasdaq)": "^NDX",
        "US 2000 (Russell 2000)": "^RUT",
        "VIX (Volatility Index)": "^VIX",
        "UK 100 (FTSE)": "^FTSE",
        "Germany 40 (DAX)": "^GDAXI",
        "France 40 (CAC)": "^FCHI",
        "Europe 50 (Euro Stoxx)": "^STOXX50E",
        "Japan 225 (Nikkei)": "^N225",
        "Hong Kong 50 (Hang Seng)": "^HSI",
        "Australia 200 (ASX)": "^AXJO"
    },
    "Precious Metals & Commodities": {
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F", "Platinum": "PL=F", 
        "Palladium": "PA=F", "Zinc": "ZNC=F", "Crude Oil (WTI)": "CL=F", 
        "Brent Crude": "BZ=F", "Natural Gas": "NG=F"
    },
    "Treasury Bonds & Notes": {
        "US 10-Year T-Note (Yield)": "^TNX",
        "US 10-Year T-Note (Futures)": "ZN=F",
        "US 30-Year T-Bond (Yield)": "^TYX",
        "US 30-Year T-Bond (Futures)": "ZB=F",
        "US 5-Year T-Note (Yield)": "^FVX",
        "US 5-Year T-Note (Futures)": "ZF=F",
        "US 2-Year T-Note (Futures)": "ZT=F"
    },
    "Crypto": {
        "BTC/USD (Bitcoin)": "BTC-USD",
        "ETH/USD (Ethereum)": "ETH-USD",
        "SOL/USD (Solana)": "SOL-USD",
        "XRP/USD (Ripple)": "XRP-USD",
        "ADA/USD (Cardano)": "ADA-USD",
        "DOGE/USD (Dogecoin)": "DOGE-USD",
        "LINK/USD (Chainlink)": "LINK-USD",
        "DOT/USD (Polkadot)": "DOT-USD",
        "LTC/USD (Litecoin)": "LTC-USD",
        "BCH/USD (Bitcoin Cash)": "BCH-USD",
        "AVAX/USD (Avalanche)": "AVAX-USD",
        "MATIC/USD (Polygon)": "MATIC-USD",
        "UNI/USD (Uniswap)": "UNI7083-USD", 
        "XLM/USD (Stellar)": "XLM-USD",
        "ATOM/USD (Cosmos)": "ATOM-USD"
    }
}

# THE GLOBAL MACRO FRED DICTIONARY
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


# 3. THE TECHNICAL ANALYSIS ENGINE
@st.cache_data(ttl=3600)
def calculate_technical_score(ticker_symbol):
    try:
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="1y")
        if df.empty or len(df) < 200:
            return 0, {"⚠️ STATUS": "Insufficient History"} 
            
        # --- NATIVE PANDAS TECHNICAL INDICATORS ---
        # 1. Moving Averages
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 2. RSI (14-Period Wilder's Smoothing)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema_12 - ema_26
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        # ------------------------------------------
        
        current = df.iloc[-1]
        close = current['Close']
        ema_20 = current['EMA_20']
        sma_50 = current['SMA_50']
        sma_200 = current['SMA_200']
        rsi_14 = current['RSI_14']
        macd_line = current['MACD_12_26_9']
        macd_signal = current['MACDs_12_26_9']
        macd_hist = current['MACDh_12_26_9']

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
            
        # MACD (25 Points) - Consolidating the duplicate checks
        if macd_hist > 0: score += 25
        else: score -= 25
            
        details = {
            "Close Price": round(close, 4),
            "EMA 20": round(ema_20, 4),
            "SMA 50": round(sma_50, 4),
            "SMA 200": round(sma_200, 4),
            "RSI (14)": round(rsi_14, 2),
            "MACD Hist": round(macd_hist, 4)
        }
        return max(-100, min(100, score)), details
    except Exception:
        return 0, {"⚠️ STATUS": "Technical API Failure"}

# 4. THE SEASONALITY ENGINE
@st.cache_data(ttl=86400)
def calculate_seasonality_score(ticker_symbol):
    try:
        import datetime
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="10y", interval="1mo")
        if df.empty: return 0, {"⚠️ STATUS": "No Seasonality Data"}
        
        current_month = datetime.datetime.now().month
        df['Returns'] = df['Close'].pct_change()
        monthly_data = df[df.index.month == current_month]['Returns'].dropna()
        
        if monthly_data.empty: return 0, {"⚠️ STATUS": "No Monthly Data"}
            
        avg_return = monthly_data.mean() * 100
        score = (avg_return / 2.0) * 100 # Scoring based on a 2% monthly move threshold
        
        # --- NEW DETAILS DICTIONARY ---
        details = {"Avg Monthly Return": round(avg_return, 2)}
        return max(-100, min(100, score)), details
        
    except Exception:
        return 0, {"⚠️ STATUS": "Seasonality API Failure"}


        # 5. THE SENTIMENT ENGINE & COT MAPPING
COT_MAPPING = {
    "Gold": {"code": "088691", "invert": False},
    "Silver": {"code": "084691", "invert": False},
    "Crude Oil (WTI)": {"code": "067651", "invert": False}
}

CURRENCY_COT_MAPPING = {
    "EUR": "099741", "GBP": "096742", "JPY": "097741",
    "CHF": "092741", "CAD": "090741", "AUD": "232741", "NZD": "112741"
}

INDEX_ETF_MAPPING = {
    "US 500 (S&P 500)": "SPY",
    "US Tech 100 (Nasdaq)": "QQQ",
    "US 30 (Dow Jones)": "DIA",
    "US 2000 (Russell 2000)": "IWM"
}

# HELPER FUNCTION: Fetches the raw CFTC score for a single asset/currency
def get_cftc_score(cftc_code):
    try:
        url = f"https://publicreporting.cftc.gov/resource/6dca-aqww.json?cftc_contract_market_code={cftc_code}&$order=report_date_as_yyyy_mm_dd DESC&$limit=2"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) == 2:
                longs = float(data[0].get('noncomm_positions_long_all', 0))
                shorts = float(data[0].get('noncomm_positions_short_all', 0))
                net_current = longs - shorts
                
                longs_prev = float(data[1].get('noncomm_positions_long_all', 0))
                shorts_prev = float(data[1].get('noncomm_positions_short_all', 0))
                net_prev = longs_prev - shorts_prev
                
                score = 50 if net_current > 0 else -50
                if net_current > net_prev: score += 50
                elif net_current < net_prev: score -= 50
                return score
    except Exception:
        pass
    return None
# HELPER FUNCTION: Fetches Put/Call ratio for US Indices via ETF proxies
def get_put_call_ratio(etf_ticker):
    try:
        asset = yf.Ticker(etf_ticker)
        expirations = asset.options
        if not expirations: return None
            
        chain = asset.option_chain(expirations[0])
        put_vol = chain.puts['volume'].sum()
        call_vol = chain.calls['volume'].sum()
        
        if call_vol == 0: return None
            
        pcr = put_vol / call_vol
        
        # Contrarian Scoring: High PCR (Fear) = Bullish, Low PCR (Greed) = Bearish
        if pcr > 1.0: return 50
        elif pcr < 0.7: return -50
        else: return 0
    except Exception:
        return None

@st.cache_data(ttl=3600)
def calculate_sentiment_score(ticker_symbol, name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # --- PART A: NEWS SENTIMENT (Google News + VADER AI) ---
        news_score = None
        try:
            analyzer = SentimentIntensityAnalyzer()
            search_query = f"{name} market news".replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            
            rss_resp = requests.get(rss_url, headers=headers, timeout=5)
            soup = BeautifulSoup(rss_resp.content, features="xml")
            headlines = soup.find_all("title")
            
            sentiment_sum = 0
            count = 0
            for headline in headlines[1:16]:
                score = analyzer.polarity_scores(headline.text)['compound']
                sentiment_sum += score
                count += 1
            if count > 0:
                news_score = (sentiment_sum / count) * 100 
        except Exception:
            pass

        # --- PART B: INSTITUTIONAL SMART MONEY (COT & PCR) ---
        smart_money_score = None
        smart_money_label = "Smart Money (COT)"
        
        if "/" in name:
            base, quote = name.split("/")
            base_code = CURRENCY_COT_MAPPING.get(base)
            quote_code = CURRENCY_COT_MAPPING.get(quote)
            
            base_score = get_cftc_score(base_code) if base_code else None
            quote_score = get_cftc_score(quote_code) if quote_code else None
            
            if quote == "USD" and base_score is not None:
                smart_money_score = base_score
            elif base == "USD" and quote_score is not None:
                smart_money_score = -quote_score 
            elif base_score is not None and quote_score is not None:
                raw_cross_score = base_score - quote_score
                smart_money_score = raw_cross_score / 2 
                
        elif name in COT_MAPPING:
            cftc_info = COT_MAPPING[name]
            raw_score = get_cftc_score(cftc_info["code"])
            if raw_score is not None:
                smart_money_score = -raw_score if cftc_info["invert"] else raw_score
                
        elif name in INDEX_ETF_MAPPING:
            pcr_score = get_put_call_ratio(INDEX_ETF_MAPPING[name])
            if pcr_score is not None:
                smart_money_score = pcr_score
                smart_money_label = "Smart Money (Put/Call)"

        # --- PART C: THE MASTER SENTIMENT SCORE ---
        available_scores = []
        if news_score is not None: available_scores.append(news_score)
        if smart_money_score is not None: available_scores.append(smart_money_score)
        
        if len(available_scores) > 0:
            final_score = sum(available_scores) / len(available_scores)
        else:
            final_score = 0 

        # --- DETAILS DICTIONARY ---
        details = {
            "News (Vader AI)": round(news_score, 2) if news_score is not None else "No Data",
            smart_money_label: round(smart_money_score, 2) if smart_money_score is not None else "No Data"
        }
        return max(-100, min(100, final_score)), details

    except Exception:
        return 0, {"⚠️ STATUS": "Sentiment API Failure"}
    
    # 6. THE FUNDAMENTALS ENGINE (MACRO PROXIES + US ANCHOR + GLOBAL EXCHANGES)

# HELPER: Fetches Real Global Economic Data from the Federal Reserve (FRED)
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_global_macro_data():
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365) 
        
        # Flatten dictionary to pull all 16 data streams at once (massive speed boost)
        all_tickers = []
        for data in FRED_MACRO_TICKERS.values():
            all_tickers.extend([data["Rate"], data["CPI"]])
            
        df = web.DataReader(all_tickers, 'fred', start, end)
        scores = {}
        
        for currency, data in FRED_MACRO_TICKERS.items():
            rate_col, cpi_col = data["Rate"], data["CPI"]
            
            # Rate Momentum
            rate_series = df[rate_col].dropna()
            rate_change = rate_series.iloc[-1] - rate_series.iloc[-2] if len(rate_series) >= 2 else 0
            
            # CPI (Inflation) Momentum
            cpi_series = df[cpi_col].dropna()
            cpi_change = ((cpi_series.iloc[-1] - cpi_series.iloc[-2]) / cpi_series.iloc[-2]) * 100 if len(cpi_series) >= 2 else 0
                
            # True Macro Logic: Rising Rates = Currency Strength (+15), Rising Inflation = Currency Devaluation (-10)
            scores[currency] = (rate_change * 15) - (cpi_change * 10)
            
        return scores
    except Exception:
        return {curr: 0 for curr in FRED_MACRO_TICKERS.keys()}

   # HELPER: Fetches Live Market Proxy Data (Cached once per hour)
@st.cache_data(ttl=3600)
def get_macro_proxy_data():
    try:
        import pandas as pd
        macro_tickers = ["^TNX", "DX-Y.NYB", "^VIX", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F"]
        
        # Download the data
        df = yf.download(macro_tickers, period="2mo", progress=False)['Close']
        
        # FIX: Forward-fill first, then backward-fill for any leading NaNs. 
        # No dropna() so we preserve the row count.
        df = df.ffill().bfill()
        
        return df
    except Exception:
        import pandas as pd
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_fundamental_score(name, asset_class):
    try:
        # --- 1. TRUE MACRO (Global FRED Economic Baseline) ---
        global_macro_scores = get_global_macro_data()
        us_true_macro = max(-100, min(100, global_macro_scores.get("USD", 0)))

        # --- 2. PROXY MOMENTUM (Live Market Data) ---
        macro_data = get_macro_proxy_data()
        
        if macro_data.empty or len(macro_data) < 20: 
            return 0, {"⚠️ STATUS": "Proxy Data Failed"}
            
        current = macro_data.iloc[-1]
        past = macro_data.iloc[-20]
        
        is_risk_off = current['^VIX'] >= 30  
        
        tnx_trend = max(-30, min(30, ((current['^TNX'] - past['^TNX']) / past['^TNX']) * 100))
        dxy_trend = max(-20, min(20, ((current['DX-Y.NYB'] - past['DX-Y.NYB']) / past['DX-Y.NYB']) * 100))
        vix_trend = max(-50, min(50, ((current['^VIX'] - past['^VIX']) / past['^VIX']) * 100))
        eur_trend = max(-20, min(20, ((current['EURUSD=X'] - past['EURUSD=X']) / past['EURUSD=X']) * 100))
        gbp_trend = max(-20, min(20, ((current['GBPUSD=X'] - past['GBPUSD=X']) / past['GBPUSD=X']) * 100))
        jpy_trend = max(-20, min(20, ((current['USDJPY=X'] - past['USDJPY=X']) / past['USDJPY=X']) * 100))
        gold_trend = max(-20, min(20, ((current['GC=F'] - past['GC=F']) / past['GC=F']) * 100))
        oil_trend = max(-30, min(30, ((current['CL=F'] - past['CL=F']) / past['CL=F']) * 100))
        
        tnx_weight = tnx_trend * 1.5
        dxy_weight = dxy_trend * 3
        vix_weight = vix_trend * 1.5
        eur_weight = eur_trend * 2.0
        gbp_weight = gbp_trend * 2.0
        jpy_weight = jpy_trend * 2.0 
        gold_weight = gold_trend * 2.0
        oil_weight = oil_trend * 2.0
        
        proxy_score = 0
        true_macro_score = 0
        
        # --- DYNAMIC SCORING LOGIC (Decoupled 50/50 Blend) ---
        if "Forex" in asset_class:
            def get_currency_scores(currency):
                c_proxy = 0
                
                # Fetch actual domestic economic data directly from FRED mapping
                c_true_macro = max(-100, min(100, global_macro_scores.get(currency, 0)))
                
                if currency == "CAD":
                    c_proxy = (oil_weight * 1.25) - dxy_weight - vix_weight
                elif currency in ["AUD", "NZD"]:
                    c_proxy = (gold_weight * 1.25) - dxy_weight - vix_weight
                elif currency == "EUR":
                    c_proxy = (eur_weight * 1.5) - dxy_weight - tnx_weight
                elif currency == "GBP":
                    c_proxy = (gbp_weight * 1.5) - dxy_weight - tnx_weight
                elif currency == "JPY":
                    c_proxy = -(jpy_weight * 1.5) - tnx_weight
                elif currency == "CHF":
                    c_proxy = vix_weight - tnx_weight
                elif currency == "USD":
                    c_proxy = dxy_weight + tnx_weight
                    
                if is_risk_off:
                    if currency in ["JPY", "CHF", "USD"]: c_proxy += 50 
                    elif currency in ["AUD", "NZD", "CAD"]: c_proxy -= 50 
                    elif currency in ["EUR", "GBP"]: c_proxy -= 20 
                    
                return c_proxy, c_true_macro
                
            if "/" in name:
                base, quote = name.split("/")
                base_proxy, base_macro = get_currency_scores(base)
                quote_proxy, quote_macro = get_currency_scores(quote)
                
                proxy_score = (base_proxy - quote_proxy) / 2
                true_macro_score = (base_macro - quote_macro) / 2
                
        elif "Indices" in asset_class:
            if name == "Japan 225 (Nikkei)":
                proxy_score = (jpy_weight * 1.5) - (vix_weight * 1.2)
                true_macro_score = us_true_macro * 0.8
            elif name == "UK 100 (FTSE)":
                proxy_score = -gbp_weight - (vix_weight * 1.2)
                true_macro_score = us_true_macro * 0.8
            elif name in ["Germany 40 (DAX)", "France 40 (CAC)", "Europe 50 (Euro Stoxx)"]:
                proxy_score = -eur_weight - (tnx_weight * 0.5) - vix_weight
                true_macro_score = us_true_macro * 0.8
            elif name == "US Tech 100 (Nasdaq)":
                proxy_score = -(tnx_weight * 2.0) - vix_weight
                true_macro_score = us_true_macro * 0.8
            elif name == "US 30 (Dow Jones)":
                proxy_score = -(tnx_weight * 0.5) - vix_weight
                true_macro_score = us_true_macro * 1.2
            elif name == "US 2000 (Russell 2000)":
                proxy_score = -(tnx_weight * 1.5) - (vix_weight * 1.2)
                true_macro_score = us_true_macro * 1.5
            else:
                proxy_score = -tnx_weight - vix_weight
                true_macro_score = us_true_macro
                
            if is_risk_off: proxy_score -= 60 

        elif "Metals" in asset_class or "Commodities" in asset_class:
            if name in ["Gold", "Silver", "Platinum"]:
                proxy_score = -dxy_weight - tnx_weight
                true_macro_score = -us_true_macro
                if is_risk_off: proxy_score += 60 
            else: 
                proxy_score = -vix_weight
                true_macro_score = us_true_macro * 0.5 
                if is_risk_off: proxy_score -= 50 

        elif "Crypto" in asset_class:
            proxy_score = -dxy_weight - tnx_weight - vix_weight
            true_macro_score = -us_true_macro
            if is_risk_off: proxy_score -= 70 

        elif "Treasury" in asset_class:
            proxy_score = -tnx_weight * 2
            true_macro_score = -us_true_macro * 0.5 
            if is_risk_off: proxy_score += 50 

        # --- THE 50/50 BLEND ---
        capped_proxy = max(-100, min(100, proxy_score))
        capped_macro = max(-100, min(100, true_macro_score))
        final_score = (capped_proxy * 0.5) + (capped_macro * 0.5)

        # --- NEW DETAILS DICTIONARY ---
        details = {
            "True Macro (FRED) [50%]": round(capped_macro, 2),
            "Proxy Momentum [50%]": round(capped_proxy, 2),
            "VIX (Fear) Trend": round(vix_trend, 2),
            "DXY (USD) Trend": round(dxy_trend, 2)
        }
        
        if "Forex" in asset_class and "/" in name:
            if 'base_proxy' in locals() and 'base_macro' in locals():
                details["Base Currency Blend"] = round((base_proxy * 0.5) + (base_macro * 0.5), 2)
                details["Quote Currency Blend"] = round((quote_proxy * 0.5) + (quote_macro * 0.5), 2)

        return max(-100, min(100, final_score)), details

    except Exception:
        return 0, {"⚠️ STATUS": "Macro API Failure"}

# 5. SIDEBAR NAVIGATION (Previously Section 4)
with st.sidebar:
    st.title("⚙️ Trading Engine")
    asset_class = st.selectbox("Select Asset Class", list(INSTRUMENTS.keys()))

# 6. MAIN DASHBOARD LAYOUT (Previously Section 5)
st.title(f"📊 Market Screener: {asset_class}")
st.divider()

# 7. LIVE DATA SCANNER (The Loop)
if "last_scanned_asset" not in st.session_state or st.session_state.last_scanned_asset != asset_class:
    
    scanned_data = []
    breakdown_data = {}  
    total_instruments = len(INSTRUMENTS[asset_class])
    my_bar = st.progress(0, text="Scanning live markets...")

    for i, (name, ticker) in enumerate(INSTRUMENTS[asset_class].items()):
        
        # --- 1. CALLING ENGINES (Unpacking Tuples) ---
        tech_score, tech_details = calculate_technical_score(ticker)
        seas_score, seas_details = calculate_seasonality_score(ticker)
        sent_score, sent_details = calculate_sentiment_score(ticker, name)
        fund_score, fund_details = calculate_fundamental_score(name, asset_class)
        
        # --- 2. MASTER WEIGHTING MATH ---
        # Technicals (30%), Fundamentals (30%), Sentiment (30%), Seasonality (10%)
        master_score = (tech_score * 0.30) + (fund_score * 0.30) + (sent_score * 0.30) + (seas_score * 0.10)
        
        # --- 3. BIAS LABELING ---
        if master_score >= 50: bias_label = "🔥 Very Bullish"
        elif master_score >= 15: bias_label = "📈 Bullish"
        elif master_score > -15: bias_label = "⚖️ Neutral"
        elif master_score > -50: bias_label = "📉 Bearish"
        else: bias_label = "❄️ Very Bearish"

        # --- 4. ADD TO TABLE & SAVE BREAKDOWNS ---
        scanned_data.append({
            "Instrument": name,
            "Master Score": round(master_score, 1),
            "Bias Status": bias_label,
            "Technicals (30%)": int(tech_score),
            "Fundamentals (30%)": int(fund_score),
            "Sentiment (30%)": int(sent_score),
            "Seasonality (10%)": int(seas_score)
        })
        
        # Save the dictionary to memory for the UI expanders
        breakdown_data[name] = {
            "Technicals": tech_details,
            "Fundamentals": fund_details,
            "Sentiment": sent_details,
            "Seasonality": seas_details
        }
        
        my_bar.progress((i + 1) / total_instruments)

    my_bar.empty()
    
    # Save to session state so it survives clicks
    st.session_state.scanned_data = scanned_data
    st.session_state.breakdown_data = breakdown_data
    st.session_state.last_scanned_asset = asset_class

# Retrieve from session state for display
df = pd.DataFrame(st.session_state.scanned_data).sort_values(by="Master Score", ascending=False).reset_index(drop=True)
breakdown_data = st.session_state.breakdown_data

# --- 5. THE COLOR FORMATTING ENGINE ---
def color_scores(val):
    """Colors positive numbers green and negative numbers red."""
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #00FF00; font-weight: bold;' # Bright Green
        elif val < 0:
            return 'color: #FF4136; font-weight: bold;' # Deep Red
        else:
            return 'color: gray;' # Neutral Zero
    return ''

# Select which columns to apply the color to
score_cols = [
    "Master Score", 
    "Technicals (30%)", 
    "Fundamentals (30%)", 
    "Sentiment (30%)", 
    "Seasonality (10%)"
]

# Apply the color style AND force the Master Score to 1 decimal place
styled_df = (
    df.style
    .map(color_scores, subset=score_cols) 
    .format("{:.1f}", subset=["Master Score"]) 
)
# Note: If you get a warning about 'applymap' being deprecated, just change it to '.map(color_scores...'

# Display the interactive dataframe with row-selection enabled
event = st.dataframe(
    styled_df, 
    width="stretch",
    on_select="rerun",
    selection_mode="single-row"
)

# --- 6. THE DRILL-DOWN BREAKDOWN UI ---
selected_rows = event.selection.rows

if selected_rows:
    # Get the index and name of the clicked instrument
    selected_idx = selected_rows[0]
    selected_instrument = df.iloc[selected_idx]["Instrument"]
    
    # Retrieve the saved breakdown dictionary from memory
    details = breakdown_data[selected_instrument]
    
    st.divider()
    st.subheader(f"🔍 Deep Dive: {selected_instrument}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📈 Technical Analysis", expanded=True):
            t_cols = st.columns(3)
            idx = 0
            for key, val in details["Technicals"].items():
                t_cols[idx % 3].metric(label=key, value=val)
                idx += 1
                
        with st.expander("🌍 Fundamental Macro"):
            f_cols = st.columns(2)
            idx = 0
            for key, val in details["Fundamentals"].items():
                f_cols[idx % 2].metric(label=key, value=val)
                idx += 1
                
    with col2:
        with st.expander("🧠 Sentiment & COT", expanded=True):
            s_cols = st.columns(3)
            idx = 0
            for key, val in details["Sentiment"].items():
                s_cols[idx % 3].metric(label=key, value=val)
                idx += 1
                
        with st.expander("📅 Seasonality"):
            st.metric(label="Average Monthly Return", value=f"{details['Seasonality'].get('Avg Monthly Return', 0)}%")