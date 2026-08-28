import requests
from datetime import datetime, timedelta

def test_fmp_calendar():
    print("--- FMP Economic Calendar Paywall & Payload Diagnostic ---\n")
    api_key = input("Enter your FMP API Key: ").strip()
    if not api_key:
        print("Test aborted.")
        return
        
    # Check the last 30 days to ensure recent data exists
    today = datetime.now()
    thirty_days_ago = today - timedelta(days=30)
    
    date_to = today.strftime('%Y-%m-%d')
    date_from = thirty_days_ago.strftime('%Y-%m-%d')
    
    url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={date_from}&to={date_to}&apikey={api_key}"
    
    print(f"\nRequesting calendar data from {date_from} to {date_to}...")
    response = requests.get(url)
    
    # Check for silent paywalls
    if response.status_code == 403:
        print("❌ FAILED: HTTP 403 Forbidden. The Economic Calendar endpoint is currently behind a paid paywall for this API key.")
        return
    elif response.status_code != 200:
        print(f"❌ FAILED: HTTP {response.status_code}. Response: {response.text}")
        return
        
    data = response.json()
    if not data:
        print("⚠️ SUCCESSFUL REQUEST, but the array is empty. The data feed might be dead or restricted.")
        return
        
    print(f"✅ SUCCESS: Retrieved {len(data)} global economic events.\n")
    print("Hunting for Growth/Jobs surprise data (Actual vs Estimate)...")
    
    # Filter for Growth/Jobs indicators to verify payload integrity
    found_high_impact = 0
    for event in data:
        name = str(event.get('event', '')).lower()
        if any(keyword in name for keyword in ['gdp', 'pmi', 'payroll', 'employment', 'unemployment']):
            print(f" - [{event.get('country')}] {event.get('date')}: {event.get('event')}")
            print(f"     Actual: {event.get('actual')} | Estimate: {event.get('estimate')} | Previous: {event.get('previous')}")
            found_high_impact += 1
            if found_high_impact >= 5: # Limit output to 5 clean examples
                break
                
    if found_high_impact == 0:
        print("⚠️ No obvious GDP/PMI/Jobs data found in the 30-day sample. The feed might be missing Tier-1 events.")

if __name__ == "__main__":
    test_fmp_calendar()