import requests
import json

url = "https://api.db.nomics.world/v22/datasets/Eurostat/une_rt_m"

print(f"Fetching: {url}\n")
resp = requests.get(url, timeout=15)
print(f"Status code: {resp.status_code}\n")

data = resp.json()

# Print the full raw structure so we can see the ACTUAL keys DBnomics uses,
# rather than guessing based on assumed naming
print("--- TOP-LEVEL KEYS ---")
print(list(data.keys()))

print("\n--- FULL RAW JSON (first 3000 chars) ---")
print(json.dumps(data, indent=2)[:3000])