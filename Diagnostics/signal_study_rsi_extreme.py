import pandas as pd
import numpy as np

print("Loading 1H data...")
df = pd.read_csv("1h_history_with_trend_choch.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
print(f"✅ Loaded {len(df)} bars\n")

OVERSOLD = 30
OVERBOUGHT = 70

# ============================================================
# BULLISH SIGNAL: RSI was below OVERSOLD, then closes back above it
# (only relevant during a downtrend — the countertrend context)
# ============================================================

was_oversold = df['RSI_14'].shift(1) < OVERSOLD
now_above_oversold = df['RSI_14'] >= OVERSOLD
in_downtrend = df['trend'] == -1

bullish_rsi_signal = was_oversold & now_above_oversold & in_downtrend

# ============================================================
# BEARISH SIGNAL: RSI was above OVERBOUGHT, then closes back below it
# (only relevant during an uptrend)
# ============================================================

was_overbought = df['RSI_14'].shift(1) > OVERBOUGHT
now_below_overbought = df['RSI_14'] <= OVERBOUGHT
in_uptrend = df['trend'] == 1

bearish_rsi_signal = was_overbought & now_below_overbought & in_uptrend

df['bullish_rsi_signal'] = bullish_rsi_signal
df['bearish_rsi_signal'] = bearish_rsi_signal

bullish_events = df[df['bullish_rsi_signal']].reset_index()
bullish_events = bullish_events.rename(columns={bullish_events.columns[0]: 'signal_time'})[['signal_time', 'RSI_14', 'Close']]

bearish_events = df[df['bearish_rsi_signal']].reset_index()
bearish_events = bearish_events.rename(columns={bearish_events.columns[0]: 'signal_time'})[['signal_time', 'RSI_14', 'Close']]

print(f"✅ Bullish RSI extreme+turn events: {len(bullish_events)}")
print(f"✅ Bearish RSI extreme+turn events: {len(bearish_events)}")

bullish_events.to_csv("1h_bullish_rsi_signals.csv", index=False)
bearish_events.to_csv("1h_bearish_rsi_signals.csv", index=False)
print("\n✅ Saved 1h_bullish_rsi_signals.csv and 1h_bearish_rsi_signals.csv")

print("\n--- Sample bullish RSI signals ---")
print(bullish_events.tail(5).to_string(index=False))
print("\n--- Sample bearish RSI signals ---")
print(bearish_events.tail(5).to_string(index=False))