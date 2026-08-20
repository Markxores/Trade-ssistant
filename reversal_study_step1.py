import yfinance as yf
import pandas as pd
import numpy as np

TICKER = "EURUSD=X"

print(f"Downloading 1H data for {TICKER}...")
asset = yf.Ticker(TICKER)
df = asset.history(period="730d", interval="1h")  # yfinance's max for 1h data
print(f"✅ Downloaded {len(df)} hourly bars")

# --- Compute indicators we'll need (MACD, RSI, ATR — same settings as before) ---
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
df['RSI_14'] = 100 - (100 / (1 + gain / loss))

high_low = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift()).abs()
low_close = (df['Low'] - df['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR_14'] = tr.rolling(14).mean()

df = df.dropna()
print(f"✅ Indicators computed. {len(df)} usable bars.\n")

# ============================================================
# Swing detection (5-bar fractal on 1H — tighter than the 7-bar
# we used on 4H, since 1H moves are naturally smaller/faster;
# we can tune this later if it looks too choppy or too sparse)
# ============================================================

FRACTAL_WINDOW = 5

highs = df['High'].values
lows = df['Low'].values
n = len(df)

is_swing_high = np.zeros(n, dtype=bool)
is_swing_low = np.zeros(n, dtype=bool)

for i in range(FRACTAL_WINDOW, n - FRACTAL_WINDOW):
    if highs[i] > highs[i-FRACTAL_WINDOW:i].max() and highs[i] > highs[i+1:i+FRACTAL_WINDOW+1].max():
        is_swing_high[i] = True
    if lows[i] < lows[i-FRACTAL_WINDOW:i].min() and lows[i] < lows[i+1:i+FRACTAL_WINDOW+1].min():
        is_swing_low[i] = True

df['is_swing_high'] = is_swing_high
df['is_swing_low'] = is_swing_low

swings = []
for i in range(n):
    if is_swing_high[i] or is_swing_low[i]:
        confirm_idx = i + FRACTAL_WINDOW
        if confirm_idx < n:
            swings.append({
                "swing_type": "HIGH" if is_swing_high[i] else "LOW",
                "swing_time": df.index[i],
                "swing_price": highs[i] if is_swing_high[i] else lows[i],
                "confirmed_time": df.index[confirm_idx],
            })

swings_df = pd.DataFrame(swings).sort_values("confirmed_time").reset_index(drop=True)
print(f"✅ Found {len(swings_df)} confirmed swings ({(swings_df['swing_type']=='HIGH').sum()} highs, {(swings_df['swing_type']=='LOW').sum()} lows)")

swings_df.to_csv("1h_swing_points.csv", index=False)
df.to_csv("1h_history_with_indicators.csv")
print("✅ Saved 1h_history_with_indicators.csv and 1h_swing_points.csv")

print("\n--- Last 10 swings ---")
print(swings_df.tail(10).to_string(index=False))