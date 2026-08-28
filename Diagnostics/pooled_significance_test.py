import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

INSTRUMENTS = [
    {"prefix": "", "label": "EUR/USD", "termination_file": "termination_events.csv",
     "bars_file": "1h_history_with_trend_choch.csv",
     "bear_div": "1h_bearish_divergences.csv", "bull_div": "1h_bullish_divergences.csv",
     "bear_rsi": "1h_bearish_rsi_signals.csv", "bull_rsi": "1h_bullish_rsi_signals.csv"},
    {"prefix": "sp500_", "label": "SP500", "termination_file": "sp500_termination_events.csv",
     "bars_file": "sp500_1h_history_with_trend_choch.csv",
     "bear_div": "sp500_bearish_divergences.csv", "bull_div": "sp500_bullish_divergences.csv",
     "bear_rsi": "sp500_bearish_rsi_signals.csv", "bull_rsi": "sp500_bullish_rsi_signals.csv"},
    {"prefix": "wti_", "label": "WTI", "termination_file": "wti_termination_events.csv",
     "bars_file": "wti_1h_history_with_trend_choch.csv",
     "bear_div": "wti_bearish_divergences.csv", "bull_div": "wti_bullish_divergences.csv",
     "bear_rsi": "wti_bearish_rsi_signals.csv", "bull_rsi": "wti_bullish_rsi_signals.csv"},
    {"prefix": "silver_", "label": "Silver", "termination_file": "silver_termination_events.csv",
     "bars_file": "silver_1h_history_with_trend_choch.csv",
     "bear_div": "silver_bearish_divergences.csv", "bull_div": "silver_bullish_divergences.csv",
     "bear_rsi": "silver_bearish_rsi_signals.csv", "bull_rsi": "silver_bullish_rsi_signals.csv"},
]

print("Loading all instruments...")
instrument_data = {}
for inst in INSTRUMENTS:
    termination_df = pd.read_csv(inst["termination_file"], parse_dates=["event_time"])
    bars_df = pd.read_csv(inst["bars_file"], index_col=0, parse_dates=True)
    bars_df.index = pd.to_datetime(bars_df.index, utc=True).tz_convert(None)

    bear_div = pd.read_csv(inst["bear_div"], parse_dates=["divergence_confirmed_time"]) if "divergence_confirmed_time" in pd.read_csv(inst["bear_div"], nrows=0).columns else pd.DataFrame(columns=["divergence_confirmed_time"])
    bull_div = pd.read_csv(inst["bull_div"], parse_dates=["divergence_confirmed_time"]) if "divergence_confirmed_time" in pd.read_csv(inst["bull_div"], nrows=0).columns else pd.DataFrame(columns=["divergence_confirmed_time"])
    bear_rsi = pd.read_csv(inst["bear_rsi"], parse_dates=["signal_time"])
    bull_rsi = pd.read_csv(inst["bull_rsi"], parse_dates=["signal_time"])

    instrument_data[inst["label"]] = {
        "termination_df": termination_df, "bar_index": bars_df.index,
        "bear_div": bear_div, "bull_div": bull_div,
        "bear_rsi": bear_rsi, "bull_rsi": bull_rsi,
    }
    print(f"  {inst['label']}: {len(termination_df)} terminations, {len(bear_div)} bear div, {len(bull_div)} bull div, {len(bear_rsi)} bear rsi, {len(bull_rsi)} bull rsi")

print()

def hit_rate_single_instrument(event_times, target_direction, terminations, bar_index, window_bars):
    hits, valid = 0, 0
    for t in event_times:
        if t not in bar_index:
            candidates = bar_index[bar_index >= t]
            if len(candidates) == 0:
                continue
            t = candidates[0]
        pos = bar_index.get_loc(t)
        window_end_pos = min(pos + window_bars, len(bar_index) - 1)
        window_end_time = bar_index[window_end_pos]
        matches = terminations[(terminations['new_trend'] == target_direction) &
                                (terminations['event_time'] > t) &
                                (terminations['event_time'] <= window_end_time)]
        valid += 1
        if len(matches) > 0:
            hits += 1
    return hits, valid

def pooled_hit_rate(signal_key, target_direction, window_bars, seed):
    total_hits, total_n = 0, 0
    total_random_hits, total_random_n = 0, 0
    for label, data in instrument_data.items():
        signal_times = data[signal_key]['divergence_confirmed_time'] if 'div' in signal_key else data[signal_key]['signal_time']
        hits, n = hit_rate_single_instrument(signal_times, target_direction, data['termination_df'], data['bar_index'], window_bars)
        total_hits += hits
        total_n += n

        if n > 0 and window_bars < len(data['bar_index']):
            np.random.seed(42)
            valid_range = data['bar_index'][:-window_bars]
            random_times = pd.Series(valid_range).sample(n=n, random_state=seed + hash(label) % 1000).values
            r_hits, r_n = hit_rate_single_instrument(random_times, target_direction, data['termination_df'], data['bar_index'], window_bars)
            total_random_hits += r_hits
            total_random_n += r_n

    return total_hits, total_n, total_random_hits, total_random_n

WINDOW_SIZES_BARS = [12, 24, 48, 72, 96]  # kept in bar terms; days will differ slightly per instrument's trading hours

print("=== POOLED MACD DIVERGENCE (all 4 instruments combined) ===")
results_macd = []
for w in WINDOW_SIZES_BARS:
    bear_hits, bear_n, rbear_hits, rbear_n = pooled_hit_rate('bear_div', -1, w, seed=10)
    bull_hits, bull_n, rbull_hits, rbull_n = pooled_hit_rate('bull_div', 1, w, seed=20)

    p_bear = fisher_exact([[bear_hits, bear_n-bear_hits],[rbear_hits, rbear_n-rbear_hits]])[1] if bear_n and rbear_n else None
    p_bull = fisher_exact([[bull_hits, bull_n-bull_hits],[rbull_hits, rbull_n-rbull_hits]])[1] if bull_n and rbull_n else None

    results_macd.append({
        "window_bars": w,
        "bear_rate": round(bear_hits/bear_n*100,1) if bear_n else None, "bear_random": round(rbear_hits/rbear_n*100,1) if rbear_n else None,
        "bear_n": bear_n, "bear_p": round(p_bear,4) if p_bear is not None else None, "bear_sig": (p_bear or 1) < 0.05,
        "bull_rate": round(bull_hits/bull_n*100,1) if bull_n else None, "bull_random": round(rbull_hits/rbull_n*100,1) if rbull_n else None,
        "bull_n": bull_n, "bull_p": round(p_bull,4) if p_bull is not None else None, "bull_sig": (p_bull or 1) < 0.05,
    })
macd_pooled_df = pd.DataFrame(results_macd)
print(macd_pooled_df.to_string(index=False))
macd_pooled_df.to_csv("pooled_macd_divergence_results.csv", index=False)

print("\n=== POOLED RSI EXTREME+TURN (all 4 instruments combined) ===")
results_rsi = []
for w in WINDOW_SIZES_BARS:
    bear_hits, bear_n, rbear_hits, rbear_n = pooled_hit_rate('bear_rsi', -1, w, seed=30)
    bull_hits, bull_n, rbull_hits, rbull_n = pooled_hit_rate('bull_rsi', 1, w, seed=40)

    p_bear = fisher_exact([[bear_hits, bear_n-bear_hits],[rbear_hits, rbear_n-rbear_hits]])[1] if bear_n and rbear_n else None
    p_bull = fisher_exact([[bull_hits, bull_n-bull_hits],[rbull_hits, rbull_n-rbull_hits]])[1] if bull_n and rbull_n else None

    results_rsi.append({
        "window_bars": w,
        "bear_rate": round(bear_hits/bear_n*100,1) if bear_n else None, "bear_random": round(rbear_hits/rbear_n*100,1) if rbear_n else None,
        "bear_n": bear_n, "bear_p": round(p_bear,4) if p_bear is not None else None, "bear_sig": (p_bear or 1) < 0.05,
        "bull_rate": round(bull_hits/bull_n*100,1) if bull_n else None, "bull_random": round(rbull_hits/rbull_n*100,1) if rbull_n else None,
        "bull_n": bull_n, "bull_p": round(p_bull,4) if p_bull is not None else None, "bull_sig": (p_bull or 1) < 0.05,
    })
rsi_pooled_df = pd.DataFrame(results_rsi)
print(rsi_pooled_df.to_string(index=False))
rsi_pooled_df.to_csv("pooled_rsi_extreme_results.csv", index=False)