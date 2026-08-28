import requests
import dbnomics as db

def test_dbnomics():
    print("--- DBnomics Data Freshness Diagnostic ---\n")
    
    # 1. Eurostat (EUR) - Direct Fetch Test
    print("1. Checking Eurostat (EUR - Euro Area Unemployment)...")
    try:
        df_eur = db.fetch_series('Eurostat', 'une_rt_m', 'M.SA.TOTAL.PC_ACT.EA20')
        latest = df_eur.dropna(subset=['value']).iloc[-1]
        print(f"✅ SUCCESS: Latest Eurostat data is from {latest['period']} (Value: {latest['value']}%)")
    except Exception as e:
        print(f"❌ FAILED to fetch Eurostat: {e}")

    # 2. Switzerland (CHF) - Provider Search Test
    print("\n2. Checking Switzerland (CHF) coverage...")
    try:
        # Corrected API subdomain
        res = requests.get("https://api.db.nomics.world/v22/search?q=unemployment+switzerland").json()
        docs = res.get('datasets', {}).get('docs', [])
        if docs:
            print(f"✅ SUCCESS: Found datasets. Top provider is '{docs[0]['provider_code']}'")
            print(f"   Top Dataset: {docs[0]['name']} (Code: {docs[0]['code']})")
        else:
            print("⚠️ No datasets found for Switzerland.")
    except Exception as e:
        print(f"❌ API Error: {e}")

    # 3. New Zealand (NZD) - Provider Search Test
    print("\n3. Checking New Zealand (NZD) coverage...")
    try:
        # Corrected API subdomain
        res = requests.get("https://api.db.nomics.world/v22/search?q=unemployment+new+zealand").json()
        docs = res.get('datasets', {}).get('docs', [])
        if docs:
            print(f"✅ SUCCESS: Found datasets. Top provider is '{docs[0]['provider_code']}'")
            print(f"   Top Dataset: {docs[0]['name']} (Code: {docs[0]['code']})")
        else:
            print("⚠️ No datasets found for New Zealand.")
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    test_dbnomics()