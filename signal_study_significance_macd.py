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

WINDOW_BARS = 48  # how many bars after a signal we allow termination to still "count"

# ============================================================
# Helper: for a list of event timestamps, and a target direction,
# compute what fraction are followed by a matching termination
# within WINDOW_BARS bars
# ============================================================

def hit_rate(event_times, target_direction, terminations, bar_index):
    hits = 0
    valid_events = 0
    for t in event_times:
        if t not in bar_index:
            # snap to nearest available bar if exact match not found
            pos_candidates = bar_index[bar_index >= t]
            if len(pos_candidates) == 0:
                continue
            t = pos_candidates[0]
        pos = bar_index.get_loc(t)
        window_end_pos = min(pos + WINDOW_BARS, len(bar_index) - 1)
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

bar_index = df.index

# ============================================================
# TEST 1: Bearish divergence -> bearish termination
# ============================================================

bear_hits, bear_n = hit_rate(bearish_div['divergence_confirmed_time'], -1, termination_df, bar_index)
print(f"Bearish divergence: {bear_hits}/{bear_n} led to a bearish termination within {WINDOW_BARS} bars ({bear_hits/bear_n*100:.1f}%)")

# ============================================================
# TEST 2: Bullish divergence -> bullish termination
# ============================================================

bull_hits, bull_n = hit_rate(bullish_div['divergence_confirmed_time'], 1, termination_df, bar_index)
print(f"Bullish divergence: {bull_hits}/{bull_n} led to a bullish termination within {WINDOW_BARS} bars ({bull_hits/bull_n*100:.1f}%)")

# ============================================================
# BASELINE: random timestamps during the dataset, matched in
# count to the real divergence events, tested the same way
# ============================================================

np.random.seed(42)  # reproducible randomness

# Only sample from timestamps that have enough room left for the full window
valid_range = bar_index[:-WINDOW_BARS]

random_bear_times = pd.Series(valid_range).sample(n=bear_n, random_state=1).values
random_bear_hits, random_bear_n = hit_rate(random_bear_times, -1, termination_df, bar_index)
print(f"\nRandom baseline (bearish): {random_bear_hits}/{random_bear_n} ({random_bear_hits/random_bear_n*100:.1f}%)")

random_bull_times = pd.Series(valid_range).sample(n=bull_n, random_state=2).values
random_bull_hits, random_bull_n = hit_rate(random_bull_times, 1, termination_df, bar_index)
print(f"Random baseline (bullish): {random_bull_hits}/{random_bull_n} ({random_bull_hits/random_bull_n*100:.1f}%)")

# ============================================================
# SIGNIFICANCE TEST: Fisher's exact test, divergence vs random
# ============================================================

print("\n--- SIGNIFICANCE TESTS (Fisher's exact) ---")

table_bear = [[bear_hits, bear_n - bear_hits], [random_bear_hits, random_bear_n - random_bear_hits]]
odds_ratio_bear, p_value_bear = fisher_exact(table_bear)
print(f"Bearish divergence vs random: odds ratio={odds_ratio_bear:.2f}, p-value={p_value_bear:.4f}", 
      "-> SIGNIFICANT" if p_value_bear < 0.05 else "-> not significant")

table_bull = [[bull_hits, bull_n - bull_hits], [random_bull_hits, random_bull_n - random_bull_hits]]
odds_ratio_bull, p_value_bull = fisher_exact(table_bull)
print(f"Bullish divergence vs random: odds ratio={odds_ratio_bull:.2f}, p-value={p_value_bull:.4f}",
      "-> SIGNIFICANT" if p_value_bull < 0.05 else "-> not significant")