import requests

def test_ff_calendar():
    print("--- ForexFactory JSON Free Feed Diagnostic ---\n")
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    
    try:
        # Spoofed headers to prevent basic bot-blocking
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Retrieved {len(data)} global economic events for this week.\n")
            
            print("Hunting for High-Impact Jobs & Growth data...")
            found = 0
            
            for event in data:
                if event.get('impact') == 'High':
                    print(f" - [{event.get('country')}] {event.get('title')}")
                    print(f"     Actual: {event.get('actual')} | Forecast: {event.get('forecast')} | Previous: {event.get('previous')}")
                    found += 1
                    
            if found == 0:
                print("   (No High-impact events scheduled for this specific week)")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_ff_calendar()