import pandas as pd
import numpy as np

print("Loading data from step 1...")
df = pd.read_csv("daily_history_with_macd_swings.csv", index_col=0, parse_dates=True)
print(f"✅ Loaded {len(df)} bars\n")

# ============================================================
# Build a clean list of confirmed price swing highs and lows,
# each tagged with the MACD value AT that same price swing bar
# ============================================================

FRACTAL_WINDOW = 7

price_highs = df[df['is_price_swing_high']].copy()
price_highs['confirmed_idx'] = [df.index.get_loc(t) + FRACTAL_WINDOW for t in price_highs.index]
price_highs = price_highs[price_highs['confirmed_idx'] < len(df)]
price_highs['confirmed_time'] = [df.index[i] for i in price_highs['confirmed_idx']]

price_lows = df[df['is_price_swing_low']].copy()
price_lows['confirmed_idx'] = [df.index.get_loc(t) + FRACTAL_WINDOW for t in price_lows.index]
price_lows = price_lows[price_lows['confirmed_idx'] < len(df)]
price_lows['confirmed_time'] = [df.index[i] for i in price_lows['confirmed_idx']]

price_highs = price_highs.sort_values('confirmed_time').reset_index()
price_highs = price_highs.rename(columns={price_highs.columns[0]: 'swing_time'})

price_lows = price_lows.sort_values('confirmed_time').reset_index()
price_lows = price_lows.rename(columns={price_lows.columns[0]: 'swing_time'})

print(f"✅ {len(price_highs)} confirmed price swing highs, {len(price_lows)} confirmed price swing lows")

# ============================================================
# BEARISH DIVERGENCE: price makes a HIGHER high, but MACD makes
# a LOWER high at that same swing point, compared to the
# previous swing high
# ============================================================

bearish_divergences = []

for i in range(1, len(price_highs)):
    curr = price_highs.iloc[i]
    prev = price_highs.iloc[i - 1]

    price_higher_high = curr['Close'] > prev['Close']
    macd_lower_high = curr['MACD'] < prev['MACD']

    if price_higher_high and macd_lower_high:
        bearish_divergences.append({
            "divergence_confirmed_time": curr['confirmed_time'],
            "swing_time": curr['swing_time'],
            "price_at_swing": curr['Close'],
            "macd_at_swing": curr['MACD'],
            "prev_swing_price": prev['Close'],
            "prev_swing_macd": prev['MACD'],
        })

bearish_df = pd.DataFrame(bearish_divergences)
print(f"\n✅ Found {len(bearish_df)} bearish divergence events")

# ============================================================
# BULLISH DIVERGENCE: price makes a LOWER low, but MACD makes
# a HIGHER low at that same swing point
# ============================================================

bullish_divergences = []

for i in range(1, len(price_lows)):
    curr = price_lows.iloc[i]
    prev = price_lows.iloc[i - 1]

    price_lower_low = curr['Close'] < prev['Close']
    macd_higher_low = curr['MACD'] > prev['MACD']

    if price_lower_low and macd_higher_low:
        bullish_divergences.append({
            "divergence_confirmed_time": curr['confirmed_time'],
            "swing_time": curr['swing_time'],
            "price_at_swing": curr['Close'],
            "macd_at_swing": curr['MACD'],
            "prev_swing_price": prev['Close'],
            "prev_swing_macd": prev['MACD'],
        })

bullish_df = pd.DataFrame(bullish_divergences)
print(f"✅ Found {len(bullish_df)} bullish divergence events")

bearish_df.to_csv("bearish_divergences.csv", index=False)
bullish_df.to_csv("bullish_divergences.csv", index=False)

print("\n--- Sample bearish divergences ---")
print(bearish_df.tail(10).to_string(index=False) if len(bearish_df) > 0 else "(none found)")

print("\n--- Sample bullish divergences ---")
print(bullish_df.tail(10).to_string(index=False) if len(bullish_df) > 0 else "(none found)")