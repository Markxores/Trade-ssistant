import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

print("Loading data...")
termination_df = pd.read_csv("termination_events.csv", parse_dates=["event_time"])
bearish_div = pd.read_csv("1h_bearish_divergences.csv", parse_dates=["divergence_confirmed_time"])
bullish_div = pd.read_csv("1h_bullish_divergences.csv", parse_dates=["divergence_confirmed_time"])
df = pd.read_csv("1h_history_with_trend_choch.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
print(f"✅ Loaded {len(termination_df)} terminations, {len(bearish_div)} bearish divs, {len(bullish_div)} bullish divs\n")

bar_index = df.index

def hit_rate(event_times, target_direction, terminations, bar_index, window_bars):
    hits = 0
    valid_events = 0
    for t in event_times:
        if t not in bar_index:
            pos_candidates = bar_index[bar_index >= t]
            if len(pos_candidates) == 0:
                continue
            t = pos_candidates[0]
        pos = bar_index.get_loc(t)
        window_end_pos = min(pos + window_bars, len(bar_index) - 1)
        window_end_time = bar_index[window_end_pos]

        matching_terms = terminations[
            (terminations['new_trend'] == target_direction) &
            (terminations['event_time'] > t) &
            (terminations['event_time'] <= window_end_time)
        ]
        valid_events += 1
        if len(matching_terms) > 0:
            hits += 1
    return hits, valid_events

# ============================================================
# Sweep across a range of window sizes
# ============================================================

WINDOW_SIZES = [12, 24, 48, 72, 96, 144, 192]  # 0.5 day up to 8 days

results = []

for window_bars in WINDOW_SIZES:
    bear_hits, bear_n = hit_rate(bearish_div['divergence_confirmed_time'], -1, termination_df, bar_index, window_bars)
    bull_hits, bull_n = hit_rate(bullish_div['divergence_confirmed_time'], 1, termination_df, bar_index, window_bars)

    np.random.seed(42)
    valid_range = bar_index[:-window_bars] if window_bars < len(bar_index) else bar_index

    random_bear_times = pd.Series(valid_range).sample(n=bear_n, random_state=1).values
    random_bear_hits, random_bear_n = hit_rate(random_bear_times, -1, termination_df, bar_index, window_bars)

    random_bull_times = pd.Series(valid_range).sample(n=bull_n, random_state=2).values
    random_bull_hits, random_bull_n = hit_rate(random_bull_times, 1, termination_df, bar_index, window_bars)

    table_bear = [[bear_hits, bear_n - bear_hits], [random_bear_hits, random_bear_n - random_bear_hits]]
    _, p_bear = fisher_exact(table_bear)

    table_bull = [[bull_hits, bull_n - bull_hits], [random_bull_hits, random_bull_n - random_bull_hits]]
    _, p_bull = fisher_exact(table_bull)

    results.append({
        "window_bars": window_bars,
        "window_days": round(window_bars / 24, 1),
        "bear_div_rate": round(bear_hits / bear_n * 100, 1),
        "bear_random_rate": round(random_bear_hits / random_bear_n * 100, 1),
        "bear_p_value": round(p_bear, 4),
        "bear_significant": p_bear < 0.05,
        "bull_div_rate": round(bull_hits / bull_n * 100, 1),
        "bull_random_rate": round(random_bull_hits / random_bull_n * 100, 1),
        "bull_p_value": round(p_bull, 4),
        "bull_significant": p_bull < 0.05,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("macd_divergence_window_sensitivity.csv", index=False)

print("--- BEARISH DIVERGENCE across window sizes ---")
print(results_df[['window_bars', 'window_days', 'bear_div_rate', 'bear_random_rate', 'bear_p_value', 'bear_significant']].to_string(index=False))

print("\n--- BULLISH DIVERGENCE across window sizes ---")
print(results_df[['window_bars', 'window_days', 'bull_div_rate', 'bull_random_rate', 'bull_p_value', 'bull_significant']].to_string(index=False))

print(f"\n✅ Saved to macd_divergence_window_sensitivity.csv")