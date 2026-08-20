import pandas as pd
import numpy as np

print("Loading 1H data...")
df = pd.read_csv("1h_history_with_indicators.csv", index_col=0, parse_dates=True)
swings_df = pd.read_csv("1h_swing_points.csv", parse_dates=["swing_time", "confirmed_time"])
swings_df['swing_time'] = pd.to_datetime(swings_df['swing_time'], utc=True).dt.tz_convert(None)
swings_df['confirmed_time'] = pd.to_datetime(swings_df['confirmed_time'], utc=True).dt.tz_convert(None)
swings_df = swings_df.sort_values("confirmed_time").reset_index(drop=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
print(f"✅ Loaded {len(df)} bars and {len(swings_df)} swings\n")

# ============================================================
# Attach the MACD value AT each price swing bar (using swing_time,
# not confirmed_time — we want the indicator reading at the moment
# the swing actually occurred)
# ============================================================

df_lookup = df[['MACD']]

price_highs = swings_df[swings_df['swing_type'] == 'HIGH'].copy()
price_highs['MACD'] = price_highs['swing_time'].map(df_lookup['MACD'])
price_highs = price_highs.dropna(subset=['MACD']).sort_values('confirmed_time').reset_index(drop=True)

price_lows = swings_df[swings_df['swing_type'] == 'LOW'].copy()
price_lows['MACD'] = price_lows['swing_time'].map(df_lookup['MACD'])
price_lows = price_lows.dropna(subset=['MACD']).sort_values('confirmed_time').reset_index(drop=True)

print(f"✅ {len(price_highs)} price swing highs with MACD attached")
print(f"✅ {len(price_lows)} price swing lows with MACD attached\n")

# ============================================================
# BEARISH DIVERGENCE: price makes a HIGHER high, MACD makes a
# LOWER high at that same swing, vs the previous swing high
# ============================================================

bearish_divergences = []
for i in range(1, len(price_highs)):
    curr = price_highs.iloc[i]
    prev = price_highs.iloc[i - 1]
    if curr['swing_price'] > prev['swing_price'] and curr['MACD'] < prev['MACD']:
        bearish_divergences.append({
            "divergence_confirmed_time": curr['confirmed_time'],
            "swing_time": curr['swing_time'],
            "price_at_swing": curr['swing_price'],
            "macd_at_swing": curr['MACD'],
            "prev_swing_price": prev['swing_price'],
            "prev_swing_macd": prev['MACD'],
        })

bearish_df = pd.DataFrame(bearish_divergences)
print(f"✅ Bearish divergence events: {len(bearish_df)}")

# ============================================================
# BULLISH DIVERGENCE: price makes a LOWER low, MACD makes a
# HIGHER low at that same swing, vs the previous swing low
# ============================================================

bullish_divergences = []
for i in range(1, len(price_lows)):
    curr = price_lows.iloc[i]
    prev = price_lows.iloc[i - 1]
    if curr['swing_price'] < prev['swing_price'] and curr['MACD'] > prev['MACD']:
        bullish_divergences.append({
            "divergence_confirmed_time": curr['confirmed_time'],
            "swing_time": curr['swing_time'],
            "price_at_swing": curr['swing_price'],
            "macd_at_swing": curr['MACD'],
            "prev_swing_price": prev['swing_price'],
            "prev_swing_macd": prev['MACD'],
        })

bullish_df = pd.DataFrame(bullish_divergences)
print(f"✅ Bullish divergence events: {len(bullish_df)}")

bearish_df.to_csv("1h_bearish_divergences.csv", index=False)
bullish_df.to_csv("1h_bullish_divergences.csv", index=False)
print("\n✅ Saved 1h_bearish_divergences.csv and 1h_bullish_divergences.csv")

print("\n--- Sample bearish divergences ---")
print(bearish_df.tail(5).to_string(index=False) if len(bearish_df) > 0 else "(none)")
print("\n--- Sample bullish divergences ---")
print(bullish_df.tail(5).to_string(index=False) if len(bullish_df) > 0 else "(none)")