import pandas as pd

print("Loading data...")
df = pd.read_csv("1h_history_with_trend_choch.csv", index_col=0, parse_dates=True)
flips_df = pd.read_csv("two_pair_flip_events.csv", parse_dates=["event_time"])
print(f"✅ Loaded {len(df)} bars, {len(flips_df)} two-pair flips\n")

choch_events = df[df['choch_event'] != 0].copy()
choch_events = choch_events.reset_index().rename(columns={choch_events.index.name or 'index': 'event_time'})
choch_events['event_type'] = 'CHOCH'
choch_events['new_trend'] = choch_events['choch_event']
choch_events = choch_events[['event_time', 'event_type', 'new_trend']]

flips_df['event_type'] = 'TWO_PAIR_FLIP'
flips_df = flips_df[['event_time', 'event_type', 'new_trend']]

all_events = pd.concat([choch_events, flips_df], ignore_index=True)

# --- Deterministic tie-break: at identical timestamps, always prefer CHOCH
# (since it's the faster/earlier-triggering rule by design) ---
event_priority = {'CHOCH': 0, 'TWO_PAIR_FLIP': 1}
all_events['priority'] = all_events['event_type'].map(event_priority)
all_events = all_events.sort_values(['event_time', 'priority']).reset_index(drop=True)

print(f"✅ Combined event list: {len(all_events)} total raw events")

# ============================================================
# Deduplicate: walk through chronologically, keep an event only
# if its direction differs from the LAST KEPT event (not just the
# immediately preceding row) — this correctly collapses same-
# timestamp or near-simultaneous conflicting events
# ============================================================

termination_events = []
last_kept_trend = None

for _, event in all_events.iterrows():
    if event['new_trend'] != last_kept_trend:
        termination_events.append(event)
        last_kept_trend = event['new_trend']

termination_events = pd.DataFrame(termination_events).drop(columns='priority').reset_index(drop=True)

print(f"✅ Deduplicated termination events: {len(termination_events)}")
print(f"   ({(termination_events['new_trend']==1).sum()} bullish terminations, {(termination_events['new_trend']==-1).sum()} bearish terminations)")

termination_events.to_csv("termination_events.csv", index=False)
print("\n✅ Saved to termination_events.csv")

print("\n--- Sample termination events ---")
print(termination_events.tail(15).to_string(index=False))

# --- Sanity check: confirm no two consecutive events share the same new_trend ---
consecutive_same = (termination_events['new_trend'] == termination_events['new_trend'].shift(1)).sum()
print(f"\n🔍 Sanity check — consecutive events with same direction: {consecutive_same} (should be 0)")