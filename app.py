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
        "US Dollar Index (DXY)": "DX=F",
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

# 3. THE TECHNICAL ANALYSIS ENGINE
# 3. THE TECHNICAL ANALYSIS ENGINE
@st.cache_data(ttl=3600)
def calculate_technical_score(ticker_symbol):
    try:
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="1y")
        if df.empty or len(df) < 200:
            return 0  
            
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
            
        if macd_line > macd_signal: score += 15
        else: score -= 15
        
        if macd_hist > 0: score += 10
        else: score -= 10
            
        return max(-100, min(100, score))
    except Exception:
        return 0

# 4. THE SEASONALITY ENGINE
@st.cache_data(ttl=86400)
def calculate_seasonality_score(ticker_symbol):
    try:
        import datetime
        asset = yf.Ticker(ticker_symbol)
        df = asset.history(period="10y", interval="1mo")
        if df.empty: return 0
        
        current_month = datetime.datetime.now().month
        df['Returns'] = df['Close'].pct_change()
        monthly_data = df[df.index.month == current_month]['Returns'].dropna()
        
        if monthly_data.empty: return 0
            
        avg_return = monthly_data.mean() * 100
        score = (avg_return / 2.0) * 100 # Scoring based on a 2% monthly move threshold
        return max(-100, min(100, score))
    except Exception:
        return 0


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

@st.cache_data(ttl=3600)
def calculate_sentiment_score(ticker_symbol, name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # --- PART A: RETAIL SENTIMENT (DailyFX Mimic) ---
        retail_score = None
        try:
            url = "https://content.dailyfx.com/api/v1/sentiment"
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                clean_name = name.replace("/", "").upper()
                if clean_name in data['sentiment']:
                    long_pct = data['sentiment'][clean_name]['long_percentage']
                    retail_score = (50 - long_pct) * 2 # Contrarian flip
        except Exception:
            pass

        # --- PART B: NEWS SENTIMENT (Google News + VADER AI) ---
        news_score = 0
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

        # --- PART C: SYNTHETIC COT SMART MONEY ---
        cot_score = 0
        cot_data_found = False
        
        if "/" in name:
            base, quote = name.split("/")
            base_code = CURRENCY_COT_MAPPING.get(base)
            quote_code = CURRENCY_COT_MAPPING.get(quote)
            
            base_score = get_cftc_score(base_code) if base_code else None
            quote_score = get_cftc_score(quote_code) if quote_code else None
            
            if quote == "USD" and base_score is not None:
                cot_score = base_score
                cot_data_found = True
            elif base == "USD" and quote_score is not None:
                cot_score = -quote_score 
                cot_data_found = True
            elif base_score is not None and quote_score is not None:
                raw_cross_score = base_score - quote_score
                cot_score = raw_cross_score / 2 
                cot_data_found = True
                
        elif name in COT_MAPPING:
            cftc_info = COT_MAPPING[name]
            raw_score = get_cftc_score(cftc_info["code"])
            if raw_score is not None:
                cot_score = -raw_score if cftc_info["invert"] else raw_score
                cot_data_found = True

        if not cot_data_found:
            try:
                asset = yf.Ticker(ticker_symbol)
                df = asset.history(period="1mo")
                price_change_5d = ((df.iloc[-1]['Close'] - df.iloc[-5]['Close']) / df.iloc[-5]['Close']) * 100
                proxy = 100 if price_change_5d > 2 else (-100 if price_change_5d < -2 else price_change_5d * 50)
                cot_score = max(-100, min(100, proxy))
            except Exception:
                pass

        # --- PART D: THE MASTER SENTIMENT SCORE ---
        if retail_score is not None:
            final_score = (retail_score + news_score + cot_score) / 3 
        else:
            final_score = (news_score + cot_score) / 2 

        return max(-100, min(100, final_score))

    except Exception:
        return 0
    
    # 6. THE FUNDAMENTALS ENGINE (MACRO PROXIES + US ANCHOR + GLOBAL EXCHANGES)

# HELPER: Fetches real US Economic Data from the Federal Reserve (FRED)
@st.cache_data(ttl=86400) # Cache for 24 hours (Macro data updates slowly)
def get_us_economic_baseline():
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365) # 1 year lookback
        
        # Pull Fed Funds Rate, CPI (Inflation), and GDP
        df = web.DataReader(['FEDFUNDS', 'CPIAUCSL', 'GDP'], 'fred', start, end)
        
        # Calculate recent changes
        rate_change = df['FEDFUNDS'].dropna().iloc[-1] - df['FEDFUNDS'].dropna().iloc[-2]
        cpi_change = ((df['CPIAUCSL'].dropna().iloc[-1] - df['CPIAUCSL'].dropna().iloc[-2]) / df['CPIAUCSL'].dropna().iloc[-2]) * 100
        gdp_change = ((df['GDP'].dropna().iloc[-1] - df['GDP'].dropna().iloc[-2]) / df['GDP'].dropna().iloc[-2]) * 100
        
        # Build the US True Health Score (Uncapped so it can drive massive momentum)
        health_score = (gdp_change * 15) + (rate_change * 10) - (cpi_change * 10)
        return health_score
    except Exception:
        return 0 

@st.cache_data(ttl=3600)
def calculate_fundamental_score(name, asset_class):
    try:
        # 1. Pull Real US Economic Data Score (The Anchor)
        us_macro_score = get_us_economic_baseline()

        # 2. Pull Live Market Proxy Data (Yields, DXY, VIX, EUR, GBP, JPY, Gold, Oil)
        macro_tickers = ["^TNX", "DX=F", "^VIX", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "CL=F"]
        macro_data = yf.download(macro_tickers, period="2mo", progress=False)['Close']
        
        if macro_data.empty or len(macro_data) < 20: 
            return 0
            
        current = macro_data.iloc[-1]
        past = macro_data.iloc[-20]
        
        # --- THE MASTER RISK SWITCH TRIGGER ---
        current_vix = current['^VIX']
        is_risk_off = current_vix >= 30  # Threshold for Institutional Panic
        
        # Calculate 20-day percentage trends
        tnx_trend = max(-30, min(30, ((current['^TNX'] - past['^TNX']) / past['^TNX']) * 100))
        dxy_trend = max(-20, min(20, ((current['DX=F'] - past['DX=F']) / past['DX=F']) * 100))
        vix_trend = max(-50, min(50, ((current['^VIX'] - past['^VIX']) / past['^VIX']) * 100))
        eur_trend = max(-20, min(20, ((current['EURUSD=X'] - past['EURUSD=X']) / past['EURUSD=X']) * 100))
        gbp_trend = max(-20, min(20, ((current['GBPUSD=X'] - past['GBPUSD=X']) / past['GBPUSD=X']) * 100))
        jpy_trend = max(-20, min(20, ((current['USDJPY=X'] - past['USDJPY=X']) / past['USDJPY=X']) * 100))
        gold_trend = max(-20, min(20, ((current['GC=F'] - past['GC=F']) / past['GC=F']) * 100))
        oil_trend = max(-30, min(30, ((current['CL=F'] - past['CL=F']) / past['CL=F']) * 100))
        
        # Apply Base Multipliers
        tnx_weight = tnx_trend * 1.5
        dxy_weight = dxy_trend * 3
        vix_weight = vix_trend * 1.5
        eur_weight = eur_trend * 2.0
        gbp_weight = gbp_trend * 2.0
        jpy_weight = jpy_trend * 2.0  # Positive means USD is strong / JPY is weak
        gold_weight = gold_trend * 2.0
        oil_weight = oil_trend * 2.0
        
        score = 0
        
        # --- DYNAMIC SCORING LOGIC ---
        
        if "Forex" in asset_class:
            def get_currency_macro_score(currency):
                base_score = 0
                
                # 1. THE COMMODITY ANCHORS (AUD, NZD, CAD) - 50% Primary Driver
                if currency == "CAD":
                    base_score = (oil_weight * 1.25) + (us_macro_score * 1.0) - dxy_weight - vix_weight
                elif currency in ["AUD", "NZD"]:
                    base_score = (gold_weight * 1.25) + (us_macro_score * 0.8) - dxy_weight - vix_weight
                    
                # 2. THE MOMENTUM MAJORS (EUR, GBP, JPY)
                elif currency == "EUR":
                    base_score = (eur_weight * 1.5) - dxy_weight - tnx_weight
                elif currency == "GBP":
                    base_score = (gbp_weight * 1.5) - dxy_weight - tnx_weight
                elif currency == "JPY":
                    base_score = -(jpy_weight * 1.5) - tnx_weight - (us_macro_score * 1.0)
                    
                # 3. THE SAFE HAVEN (CHF)
                elif currency == "CHF":
                    base_score = vix_weight - tnx_weight - us_macro_score
                    
                # 4. THE BASELINE (USD)
                elif currency == "USD":
                    base_score = dxy_weight + tnx_weight + us_macro_score
                
                # --- RISK REGIME OVERRIDE FOR FOREX ---
                if is_risk_off:
                    if currency in ["JPY", "CHF", "USD"]: 
                        base_score += 50  # Capital flies into safe havens
                    elif currency in ["AUD", "NZD", "CAD"]: 
                        base_score -= 50  # Capital abandons commodity/risk currencies
                    elif currency in ["EUR", "GBP"]: 
                        base_score -= 20  # Core majors suffer moderate sell-offs
                        
                return base_score
                
            if "/" in name:
                base, quote = name.split("/")
                base_val = get_currency_macro_score(base)
                quote_val = get_currency_macro_score(quote)
                score = (base_val - quote_val) / 2
                
        elif "Indices" in asset_class:
            if name == "Japan 225 (Nikkei)":
                score = (jpy_weight * 1.5) + (us_macro_score * 0.8) - (vix_weight * 1.2)
            elif name == "UK 100 (FTSE)":
                score = (us_macro_score * 0.8) - gbp_weight - (vix_weight * 1.2)
            elif name in ["Germany 40 (DAX)", "France 40 (CAC)", "Europe 50 (Euro Stoxx)"]:
                score = (us_macro_score * 0.8) - eur_weight - (tnx_weight * 0.5) - vix_weight
            elif name == "US Tech 100 (Nasdaq)":
                score = (us_macro_score * 0.8) - (tnx_weight * 2.0) - vix_weight
            elif name == "US 30 (Dow Jones)":
                score = (us_macro_score * 1.2) - (tnx_weight * 0.5) - vix_weight
            elif name == "US 2000 (Russell 2000)":
                score = (us_macro_score * 1.5) - (tnx_weight * 1.5) - (vix_weight * 1.2)
            else:
                score = us_macro_score - tnx_weight - vix_weight
                
            # --- RISK REGIME OVERRIDE FOR INDICES ---
            if is_risk_off:
                score -= 60  # Equities are dumped globally during panic

        elif "Metals" in asset_class or "Commodities" in asset_class:
            if name in ["Gold", "Silver", "Platinum"]:
                score = -us_macro_score - dxy_weight - tnx_weight
                # --- RISK REGIME OVERRIDE ---
                if is_risk_off: score += 60  # Precious metals act as a hard hedge
            else: 
                score = -vix_weight
                # --- RISK REGIME OVERRIDE ---
                if is_risk_off: score -= 50  # Industrial growth stops, oil/copper dumped

        elif "Crypto" in asset_class:
            score = -us_macro_score - dxy_weight - tnx_weight - vix_weight
            # --- RISK REGIME OVERRIDE FOR CRYPTO ---
            if is_risk_off:
                score -= 70  # The highest beta assets get dumped the hardest

        elif "Treasury" in asset_class:
            score = -tnx_weight * 2
            # --- RISK REGIME OVERRIDE FOR BONDS ---
            if is_risk_off:
                score += 50  # Institutions buy bonds for guaranteed yield

        # Cap only the final output to ensure it fits the Master Score formula
        return max(-100, min(100, score))

    except Exception:
        return 0

# 5. SIDEBAR NAVIGATION (Previously Section 4)
with st.sidebar:
    st.title("⚙️ Trading Engine")
    asset_class = st.selectbox("Select Asset Class", list(INSTRUMENTS.keys()))

# 6. MAIN DASHBOARD LAYOUT (Previously Section 5)
st.title(f"📊 Market Screener: {asset_class}")
st.divider()

# 7. LIVE DATA SCANNER (The Loop)
scanned_data = []
total_instruments = len(INSTRUMENTS[asset_class])
my_bar = st.progress(0, text="Scanning live markets...")

for i, (name, ticker) in enumerate(INSTRUMENTS[asset_class].items()):
    
    # --- 1. CALLING ALL 4 LIVE ENGINES ---
    tech_score = calculate_technical_score(ticker)
    seas_score = calculate_seasonality_score(ticker)
    sent_score = calculate_sentiment_score(ticker, name)
    fund_score = calculate_fundamental_score(name, asset_class)
    
    # --- 2. MASTER WEIGHTING MATH ---
    # Technicals (30%), Fundamentals (30%), Sentiment (30%), Seasonality (10%)
    master_score = (tech_score * 0.30) + (fund_score * 0.30) + (sent_score * 0.30) + (seas_score * 0.10)
    
    # --- 3. BIAS LABELING ---
    if master_score >= 50: bias_label = "🔥 Very Bullish"
    elif master_score >= 15: bias_label = "📈 Bullish"
    elif master_score > -15: bias_label = "⚖️ Neutral"
    elif master_score > -50: bias_label = "📉 Bearish"
    else: bias_label = "❄️ Very Bearish"

    # --- 4. ADD TO TABLE ---
    scanned_data.append({
        "Instrument": name,
        "Master Score": round(master_score, 1),
        "Bias Status": bias_label,
        "Technicals (30%)": int(tech_score),
        "Fundamentals (30%)": int(fund_score),
        "Sentiment (30%)": int(sent_score),
        "Seasonality (10%)": int(seas_score)
    })
    
    # Update progress bar
    my_bar.progress((i + 1) / total_instruments)

# Clear the progress bar when done
my_bar.empty()

# Create the final dataframe and sort by the highest master score
# Create the final dataframe and sort by the highest master score
df = pd.DataFrame(scanned_data).sort_values(by="Master Score", ascending=False).reset_index(drop=True)

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

# Display the beautiful, color-coded dashboard
st.dataframe(styled_df, width="stretch")