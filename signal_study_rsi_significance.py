import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

print("Loading data...")
termination_df = pd.read_csv("termination_events.csv", parse_dates=["event_time"])
bullish_rsi = pd.read_csv("1h_bullish_rsi_signals.csv", parse_dates=["signal_time"])
bearish_rsi = pd.read_csv("1h_bearish_rsi_signals.csv", parse_dates=["signal_time"])
df = pd.read_csv("1h_history_with_trend_choch.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
print(f"✅ Loaded {len(termination_df)} terminations, {len(bullish_rsi)} bullish RSI signals, {len(bearish_rsi)} bearish RSI signals\n")

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

def run_sensitivity_sweep(bullish_times, bearish_times, window_sizes, seed_a=1, seed_b=2):
    results = []
    for window_bars in window_sizes:
        bear_hits, bear_n = hit_rate(bearish_times, -1, termination_df, bar_index, window_bars)
        bull_hits, bull_n = hit_rate(bullish_times, 1, termination_df, bar_index, window_bars)

        np.random.seed(42)
        valid_range = bar_index[:-window_bars] if window_bars < len(bar_index) else bar_index

        random_bear_times = pd.Series(valid_range).sample(n=bear_n, random_state=seed_a).values
        random_bear_hits, random_bear_n = hit_rate(random_bear_times, -1, termination_df, bar_index, window_bars)

        random_bull_times = pd.Series(valid_range).sample(n=bull_n, random_state=seed_b).values
        random_bull_hits, random_bull_n = hit_rate(random_bull_times, 1, termination_df, bar_index, window_bars)

        table_bear = [[bear_hits, bear_n - bear_hits], [random_bear_hits, random_bear_n - random_bear_hits]]
        _, p_bear = fisher_exact(table_bear)

        table_bull = [[bull_hits, bull_n - bull_hits], [random_bull_hits, random_bull_n - random_bull_hits]]
        _, p_bull = fisher_exact(table_bull)

        results.append({
            "window_bars": window_bars,
            "window_days": round(window_bars / 24, 1),
            "bear_signal_rate": round(bear_hits / bear_n * 100, 1),
            "bear_random_rate": round(random_bear_hits / random_bear_n * 100, 1),
            "bear_p_value": round(p_bear, 4),
            "bear_significant": p_bear < 0.05,
            "bull_signal_rate": round(bull_hits / bull_n * 100, 1),
            "bull_random_rate": round(random_bull_hits / random_bull_n * 100, 1),
            "bull_p_value": round(p_bull, 4),
            "bull_significant": p_bull < 0.05,
        })
    return pd.DataFrame(results)

WINDOW_SIZES = [12, 24, 48, 72, 96, 144, 192]

results_df = run_sensitivity_sweep(bullish_rsi['signal_time'], bearish_rsi['signal_time'], WINDOW_SIZES, seed_a=3, seed_b=4)
results_df.to_csv("rsi_extreme_window_sensitivity.csv", index=False)

print("--- BEARISH RSI EXTREME+TURN across window sizes ---")
print(results_df[['window_bars', 'window_days', 'bear_signal_rate', 'bear_random_rate', 'bear_p_value', 'bear_significant']].to_string(index=False))

print("\n--- BULLISH RSI EXTREME+TURN across window sizes ---")
print(results_df[['window_bars', 'window_days', 'bull_signal_rate', 'bull_random_rate', 'bull_p_value', 'bull_significant']].to_string(index=False))

print(f"\n✅ Saved to rsi_extreme_window_sensitivity.csv")