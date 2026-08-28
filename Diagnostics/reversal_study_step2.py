import pandas as pd
import numpy as np

print("Loading data...")
df = pd.read_csv("1h_history_with_indicators.csv", index_col=0, parse_dates=True)
swings_df = pd.read_csv("1h_swing_points.csv", parse_dates=["swing_time", "confirmed_time"])
swings_df['swing_time'] = pd.to_datetime(swings_df['swing_time'], utc=True).dt.tz_convert(None)
swings_df['confirmed_time'] = pd.to_datetime(swings_df['confirmed_time'], utc=True).dt.tz_convert(None)
swings_df = swings_df.sort_values("confirmed_time").reset_index(drop=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
print(f"✅ Loaded {len(df)} bars and {len(swings_df)} swings\n")

# ============================================================
# Build trend state bar-by-bar (single source of truth)
# ============================================================

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

    # Reset stale references the instant trend changes
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

# ============================================================
# Two-pair flip events: derived directly from df['trend'] changes
# ============================================================

trend_change = df['trend'] != df['trend'].shift(1)
two_pair_flips_df = df[trend_change & (df['trend'] != 0)].reset_index()
two_pair_flips_df = two_pair_flips_df.rename(columns={two_pair_flips_df.columns[0]: 'event_time'})
two_pair_flips_df['event_type'] = 'TWO_PAIR_FLIP'
two_pair_flips_df['new_trend'] = two_pair_flips_df['trend']
two_pair_flips_df = two_pair_flips_df[['event_time', 'event_type', 'new_trend']]

print(f"✅ Two-pair flip events: {len(two_pair_flips_df)}")

# ============================================================
# CHoCH detection
# ============================================================

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

df.to_csv("1h_history_with_trend_choch.csv")
two_pair_flips_df.to_csv("two_pair_flip_events.csv", index=False)

print("\n✅ Saved 1h_history_with_trend_choch.csv and two_pair_flip_events.csv")
print("\n--- Sample CHoCH events ---")
print(df[df['choch_event'] != 0][['Close', 'trend', 'choch_event', 'pullback_low_ref', 'pullback_low_id', 'pullback_high_ref', 'pullback_high_id']].tail(15).to_string())