import streamlit as st
import pandas as pd
import random
import yfinance as yf
import pandas_ta_classic as ta
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
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD"
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
            
        df.ta.ema(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
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

        # 2. Pull Live Market Proxy Data (Yields, DXY, VIX)
        macro_tickers = ["^TNX", "DX=F", "^VIX"]
        macro_data = yf.download(macro_tickers, period="1mo", progress=False)['Close']
        
        if macro_data.empty or len(macro_data) < 20: 
            return 0
            
        current = macro_data.iloc[-1]
        past = macro_data.iloc[-20]
        
        # Calculate 20-day percentage trends
        tnx_trend = ((current['^TNX'] - past['^TNX']) / past['^TNX']) * 100
        dxy_trend = ((current['DX=F'] - past['DX=F']) / past['DX=F']) * 100
        vix_trend = ((current['^VIX'] - past['^VIX']) / past['^VIX']) * 100
        
        # Apply Base Multipliers
        tnx_weight = tnx_trend * 5
        dxy_weight = dxy_trend * 10
        vix_weight = vix_trend * 5
        
        score = 0
        
        # --- DYNAMIC SCORING LOGIC ---
        
        if "Forex" in asset_class:
            def get_currency_macro_score(currency):
                # THE US DOLLAR ANCHOR: We inject the us_macro_score into every currency, 
                # but adjust HOW sensitive that currency is to the US economy.
                if currency in ["JPY", "CHF"]:
                    # Safe havens are crushed by strong US rates and a booming US economy.
                    return vix_weight - tnx_weight - (us_macro_score * 1.5)      
                elif currency in ["AUD", "NZD", "CAD"]:
                    # Risk currencies hate a strong USD, but LIKE strong US GDP (global growth).
                    return -vix_weight - dxy_weight + (us_macro_score * 0.5)     
                elif currency in ["EUR", "GBP"]:
                    # Core majors generally move opposite to US economic strength.
                    return -dxy_weight - us_macro_score                  
                elif currency == "USD":
                    # The Anchor itself.
                    return dxy_weight + tnx_weight + us_macro_score      
                return 0
                
            if "/" in name:
                base, quote = name.split("/")
                base_score = get_currency_macro_score(base)
                quote_score = get_currency_macro_score(quote)
                
                # Synthetic Cross Spread (Works perfectly for EUR/JPY, AUD/CAD, or EUR/USD)
                score = base_score - quote_score
                
        elif "Indices" in asset_class:
            # 1. Japanese Indices (Nikkei 225)
            if "Nikkei" in name or "JP225" in name:
                score = (dxy_weight * 0.8) - vix_weight + (us_macro_score * 0.5)
            # 2. European Indices (DAX, FTSE, CAC)
            elif name in ["FTSE 100", "DAX", "UK100", "GER40", "CAC 40"]:
                score = (us_macro_score * 0.5) - (tnx_weight * 0.5) - vix_weight
            # 3. Default US Indices (S&P 500, Nasdaq, Dow Jones)
            else:
                score = us_macro_score - tnx_weight - vix_weight

        elif "Metals" in asset_class or "Commodities" in asset_class:
            if name in ["Gold", "Silver", "Platinum"]:
                # Precious metals are an alternative to the USD and US Yields
                score = -us_macro_score - dxy_weight - tnx_weight
            else: 
                # Energy/Industrial metals rely purely on global risk appetite
                score = -vix_weight

        elif "Crypto" in asset_class:
            # Crypto is a high-beta risk asset. It hates a strong US Dollar (DXY), 
            # high Treasury yields (expensive borrowing), and market fear (VIX).
            score = -us_macro_score - dxy_weight - tnx_weight - vix_weight

        elif "Treasury" in asset_class:
            # Bond PRICES move inverse to Bond YIELDS. 
            score = -tnx_weight * 2

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