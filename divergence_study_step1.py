import yfinance as yf
import pandas as pd
import numpy as np

TICKER = "EURUSD=X"

print(f"Downloading daily data for {TICKER}...")
asset = yf.Ticker(TICKER)
df = asset.history(period="5y")  # more history = more divergence events to study
print(f"✅ Downloaded {len(df)} daily bars")

# --- Compute MACD (same settings as your live app: 12, 26, 9) ---
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# --- Compute ATR (14-period, same as your live app) ---
high_low = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift()).abs()
low_close = (df['Low'] - df['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR_14'] = tr.rolling(14).mean()

df = df.dropna()
print(f"✅ Indicators computed. {len(df)} usable bars after warm-up.")

# ============================================================
# Detect swing highs/lows on PRICE — same 7-bar fractal logic as before
# ============================================================

FRACTAL_WINDOW = 7

highs = df['High'].values
lows = df['Low'].values
n = len(df)

is_price_swing_high = np.zeros(n, dtype=bool)
is_price_swing_low = np.zeros(n, dtype=bool)

for i in range(FRACTAL_WINDOW, n - FRACTAL_WINDOW):
    if highs[i] > highs[i-FRACTAL_WINDOW:i].max() and highs[i] > highs[i+1:i+FRACTAL_WINDOW+1].max():
        is_price_swing_high[i] = True
    if lows[i] < lows[i-FRACTAL_WINDOW:i].min() and lows[i] < lows[i+1:i+FRACTAL_WINDOW+1].min():
        is_price_swing_low[i] = True

df['is_price_swing_high'] = is_price_swing_high
df['is_price_swing_low'] = is_price_swing_low

# ============================================================
# Detect swing highs/lows on the MACD LINE itself — needed to compare
# "is the indicator making a lower high while price makes a higher high"
# ============================================================

macd_vals = df['MACD'].values

is_macd_swing_high = np.zeros(n, dtype=bool)
is_macd_swing_low = np.zeros(n, dtype=bool)

for i in range(FRACTAL_WINDOW, n - FRACTAL_WINDOW):
    if macd_vals[i] > macd_vals[i-FRACTAL_WINDOW:i].max() and macd_vals[i] > macd_vals[i+1:i+FRACTAL_WINDOW+1].max():
        is_macd_swing_high[i] = True
    if macd_vals[i] < macd_vals[i-FRACTAL_WINDOW:i].min() and macd_vals[i] < macd_vals[i+1:i+FRACTAL_WINDOW+1].min():
        is_macd_swing_low[i] = True

df['is_macd_swing_high'] = is_macd_swing_high
df['is_macd_swing_low'] = is_macd_swing_low

print(f"\n✅ Price swings: {is_price_swing_high.sum()} highs, {is_price_swing_low.sum()} lows")
print(f"✅ MACD swings: {is_macd_swing_high.sum()} highs, {is_macd_swing_low.sum()} lows")

df.to_csv("daily_history_with_macd_swings.csv")
print("\n✅ Saved to daily_history_with_macd_swings.csv")

print("\n--- Last 10 rows ---")
print(df[['Close', 'MACD', 'is_price_swing_high', 'is_price_swing_low', 'is_macd_swing_high', 'is_macd_swing_low']].tail(10).to_string())