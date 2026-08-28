import yfinance as yf
import pandas as pd
import numpy as np

TICKER = "SI=F"
PREFIX = "silver"
FRACTAL_WINDOW = 5

print(f"Downloading 1H data for {TICKER}...")
asset = yf.Ticker(TICKER)
df = asset.history(period="730d", interval="1h")
print(f"✅ Downloaded {len(df)} hourly bars")

# --- Indicators ---
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
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
print(f"✅ Indicators computed. {len(df)} usable bars.\n")

# --- Swing detection ---
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
print(f"✅ Found {len(swings_df)} confirmed swings\n")

# --- Trend state machine (single source of truth) ---
state_log = []
trend = 0
previous_trend = 0
last_high, prev_high = None, None
last_low, prev_low = None, None
opposite_count = 0
current_pullback_low = None
current_pullback_high = None
current_pullback_low_id = None
current_pullback_high_id = None

for swing_idx, swing in swings_df.iterrows():
    if swing['swing_type'] == 'HIGH':
        prev_high = last_high
        last_high = swing['swing_price']
    else:
        prev_low = last_low
        last_low = swing['swing_price']

    if prev_high is not None and prev_low is not None:
        is_down_pair = (last_high < prev_high) and (last_low < prev_low)
        is_up_pair = (last_high > prev_high) and (last_low > prev_low)

        if trend == 0:
            if is_up_pair: trend = 1
            elif is_down_pair: trend = -1
        elif trend == 1 and is_down_pair:
            opposite_count += 1
            if opposite_count >= 2:
                trend = -1
                opposite_count = 0
        elif trend == -1 and is_up_pair:
            opposite_count += 1
            if opposite_count >= 2:
                trend = 1
                opposite_count = 0
        else:
            opposite_count = 0

    if trend != previous_trend and previous_trend != 0:
        current_pullback_low, current_pullback_low_id = None, None
        current_pullback_high, current_pullback_high_id = None, None
    previous_trend = trend

    if trend == 1 and swing['swing_type'] == 'LOW':
        current_pullback_low = swing['swing_price']
        current_pullback_low_id = swing_idx
    elif trend == -1 and swing['swing_type'] == 'HIGH':
        current_pullback_high = swing['swing_price']
        current_pullback_high_id = swing_idx

    state_log.append({
        "confirmed_time": swing['confirmed_time'],
        "trend": trend,
        "pullback_low_ref": current_pullback_low,
        "pullback_high_ref": current_pullback_high,
        "pullback_low_id": current_pullback_low_id,
        "pullback_high_id": current_pullback_high_id,
    })

state_df = pd.DataFrame(state_log).set_index("confirmed_time")
state_df.index = pd.to_datetime(state_df.index, utc=True).tz_convert(None)
state_df = state_df.sort_index()
df = pd.merge_asof(df, state_df, left_index=True, right_index=True, direction='backward')

# --- Two-pair flips (derived from trend column) ---
trend_change = df['trend'] != df['trend'].shift(1)
two_pair_flips_df = df[trend_change & (df['trend'] != 0)].reset_index()
two_pair_flips_df = two_pair_flips_df.rename(columns={two_pair_flips_df.columns[0]: 'event_time'})
two_pair_flips_df['event_type'] = 'TWO_PAIR_FLIP'
two_pair_flips_df['new_trend'] = two_pair_flips_df['trend']
two_pair_flips_df = two_pair_flips_df[['event_time', 'event_type', 'new_trend']]
print(f"✅ Two-pair flip events: {len(two_pair_flips_df)}")

# --- CHoCH detection ---
choch_events = [0] * len(df)
already_broken_low = False
already_broken_high = False
last_ref_low_id = None
last_ref_high_id = None

for i in range(len(df)):
    row = df.iloc[i]
    trend_now = row['trend']
    ref_low = row['pullback_low_ref']
    ref_high = row['pullback_high_ref']
    ref_low_id = row['pullback_low_id']
    ref_high_id = row['pullback_high_id']

    if ref_low_id != last_ref_low_id:
        already_broken_low = False
        last_ref_low_id = ref_low_id
    if ref_high_id != last_ref_high_id:
        already_broken_high = False
        last_ref_high_id = ref_high_id

    if trend_now == 1 and pd.notna(ref_low) and not already_broken_low:
        if row['Close'] < ref_low:
            choch_events[i] = -1
            already_broken_low = True
    elif trend_now == -1 and pd.notna(ref_high) and not already_broken_high:
        if row['Close'] > ref_high:
            choch_events[i] = 1
            already_broken_high = True

df['choch_event'] = choch_events
choch_count = (df['choch_event'] != 0).sum()
print(f"✅ CHoCH events detected: {choch_count}")

df.to_csv(f"{PREFIX}_1h_history_with_trend_choch.csv")

# --- Combine + deduplicate termination events ---
choch_events_df = df[df['choch_event'] != 0].copy()
choch_events_df = choch_events_df.reset_index().rename(columns={choch_events_df.index.name or 'index': 'event_time'})
choch_events_df['event_type'] = 'CHOCH'
choch_events_df['new_trend'] = choch_events_df['choch_event']
choch_events_df = choch_events_df[['event_time', 'event_type', 'new_trend']]

all_events = pd.concat([choch_events_df, two_pair_flips_df], ignore_index=True)
event_priority = {'CHOCH': 0, 'TWO_PAIR_FLIP': 1}
all_events['priority'] = all_events['event_type'].map(event_priority)
all_events = all_events.sort_values(['event_time', 'priority']).reset_index(drop=True)

termination_events = []
last_kept_trend = None
for _, event in all_events.iterrows():
    if event['new_trend'] != last_kept_trend:
        termination_events.append(event)
        last_kept_trend = event['new_trend']

termination_events = pd.DataFrame(termination_events).drop(columns='priority').reset_index(drop=True)
termination_events.to_csv(f"{PREFIX}_termination_events.csv", index=False)
print(f"✅ Termination events: {len(termination_events)} ({(termination_events['new_trend']==1).sum()} bullish, {(termination_events['new_trend']==-1).sum()} bearish)")

consecutive_same = (termination_events['new_trend'] == termination_events['new_trend'].shift(1)).sum()
print(f"🔍 Sanity check: {consecutive_same} (should be 0)\n")

# --- MACD divergence ---
df_lookup = df[['MACD']]
price_highs = swings_df[swings_df['swing_type'] == 'HIGH'].copy()
price_highs['MACD'] = price_highs['swing_time'].map(df_lookup['MACD'])
price_highs = price_highs.dropna(subset=['MACD']).sort_values('confirmed_time').reset_index(drop=True)

price_lows = swings_df[swings_df['swing_type'] == 'LOW'].copy()
price_lows['MACD'] = price_lows['swing_time'].map(df_lookup['MACD'])
price_lows = price_lows.dropna(subset=['MACD']).sort_values('confirmed_time').reset_index(drop=True)

bearish_divergences = []
for i in range(1, len(price_highs)):
    curr, prev = price_highs.iloc[i], price_highs.iloc[i-1]
    if curr['swing_price'] > prev['swing_price'] and curr['MACD'] < prev['MACD']:
        bearish_divergences.append({"divergence_confirmed_time": curr['confirmed_time']})
bearish_div_df = pd.DataFrame(bearish_divergences)

bullish_divergences = []
for i in range(1, len(price_lows)):
    curr, prev = price_lows.iloc[i], price_lows.iloc[i-1]
    if curr['swing_price'] < prev['swing_price'] and curr['MACD'] > prev['MACD']:
        bullish_divergences.append({"divergence_confirmed_time": curr['confirmed_time']})
bullish_div_df = pd.DataFrame(bullish_divergences)

bearish_div_df.to_csv(f"{PREFIX}_bearish_divergences.csv", index=False)
bullish_div_df.to_csv(f"{PREFIX}_bullish_divergences.csv", index=False)
print(f"✅ MACD divergences: {len(bearish_div_df)} bearish, {len(bullish_div_df)} bullish")

# --- RSI extreme + turn ---
OVERSOLD, OVERBOUGHT = 30, 70
was_oversold = df['RSI_14'].shift(1) < OVERSOLD
now_above_oversold = df['RSI_14'] >= OVERSOLD
in_downtrend = df['trend'] == -1
bullish_rsi_signal = was_oversold & now_above_oversold & in_downtrend

was_overbought = df['RSI_14'].shift(1) > OVERBOUGHT
now_below_overbought = df['RSI_14'] <= OVERBOUGHT
in_uptrend = df['trend'] == 1
bearish_rsi_signal = was_overbought & now_below_overbought & in_uptrend

bullish_rsi_events = df[bullish_rsi_signal].reset_index()
bullish_rsi_events = bullish_rsi_events.rename(columns={bullish_rsi_events.columns[0]: 'signal_time'})[['signal_time']]

bearish_rsi_events = df[bearish_rsi_signal].reset_index()
bearish_rsi_events = bearish_rsi_events.rename(columns={bearish_rsi_events.columns[0]: 'signal_time'})[['signal_time']]

bullish_rsi_events.to_csv(f"{PREFIX}_bullish_rsi_signals.csv", index=False)
bearish_rsi_events.to_csv(f"{PREFIX}_bearish_rsi_signals.csv", index=False)
print(f"✅ RSI extreme+turn: {len(bearish_rsi_events)} bearish, {len(bullish_rsi_events)} bullish")

print(f"\n✅ ALL {PREFIX.upper()} PIPELINE FILES SAVED")