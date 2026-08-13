from trading_ig import IGService

IG_USERNAME = "Xores1"
IG_PASSWORD = "DMarverick22!!"
IG_API_KEY = "71c054f96a5db51f84a0c9e8a5942bb855a23a36"
IG_ACC_TYPE = "DEMO"

ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
ig_service.create_session()
print("--- LOGIN SUCCESS ---\n")

confirmed_ids = {
    "US 30 (Dow Jones)": "WALL",
    "US 2000 (Russell 2000)": "R2000",
    "Germany 40 (DAX)": "DE30",
    "France 40 (CAC)": "FR40",
    "Hong Kong 50 (Hang Seng)": "HS34",
    "Australia 200 (ASX)": "AU200",
}

for name, market_id in confirmed_ids.items():
    try:
        result = ig_service.fetch_client_sentiment_by_instrument(market_id)
        flag = "⚠️ zero/zero" if result['longPositionPercentage'] == 0.0 and result['shortPositionPercentage'] == 0.0 else "✅ REAL DATA"
        print(f"{flag}  {name} ('{market_id}') -> {result}")
    except Exception as e:
        print(f"❌ {name} ('{market_id}') failed: {type(e).__name__}")