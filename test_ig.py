from trading_ig import IGService

# Replace with your actual demo details to test
ig_service = IGService(
    username="Xores1",
    password="DMarverick22!!",
    api_key="71c054f96a5db51f84a0c9e8a5942bb855a23a36",
    acc_type="DEMO"
)

# Connect and fetch EUR/USD sentiment
ig_service.create_session()
sentiment = ig_service.fetch_client_sentiment_by_instrument("EURUSD")

print("EUR/USD Client Sentiment:", sentiment)