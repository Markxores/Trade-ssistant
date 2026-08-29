import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import math

TEST_INSTRUMENTS = {
    "Forex": [
        ("EUR/USD", "EURUSD=X"),
        ("GBP/JPY", "GBPJPY=X"),
        ("USD/CAD", "USDCAD=X")
    ],
    "Indices": [
        ("S&P 500", "^GSPC"),
        ("Nasdaq 100", "^NDX"),
        ("DAX 40", "^GDAXI")
    ],
    "Commodities": [
        ("Gold", "GC=F"),
        ("Crude Oil", "CL=F"),
        ("Natural Gas", "NG=F")
    ],
    "Crypto": [
        ("Bitcoin", "BTC-USD"),
        ("Ethereum", "ETH-USD")
    ]
}

def get_test_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    })
    try:
        session.get("https://finance.yahoo.com", timeout=5)
    except Exception:
        pass
    return session

def run_scaling_diagnostic():
    session = get_test_session()
    analyzer = SentimentIntensityAnalyzer()
    
    print("=" * 85)
    print(f"{'INSTRUMENT':<16} | {'ARTICLES':<8} | {'RAW (1.0x)':<12} | {'FILTERED (2.0x)':<16} | {'POWER ROOT':<12}")
    print("=" * 85)

    for category, items in TEST_INSTRUMENTS.items():
        print(f"\n--- {category.upper()} ---")
        for name, ticker in items:
            try:
                asset = yf.Ticker(ticker, session=session)
                news_items = asset.news
                
                if not news_items:
                    print(f"{name:<16} | {'0':<8} | {'No Data':<12} | {'No Data':<16} | {'No Data':<12}")
                    continue

                compounds = []
                seen = set()

                for item in news_items[:15]:
                    title = item.get('title') or item.get('content', {}).get('title', '')
                    summary = item.get('summary') or item.get('content', {}).get('summary', '') or item.get('text', '')
                    
                    if not title or title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    
                    full_text = f"{title}. {summary}" if summary else title
                    score = analyzer.polarity_scores(full_text)['compound']
                    compounds.append(score)

                if not compounds:
                    continue

                # Model 1: Raw Linear (1.0x)
                raw_mean = sum(compounds) / len(compounds)
                score_raw = max(-100.0, min(100.0, raw_mean * 100.0))

                # Model 2: Filtered Non-Neutral (2.0x)
                non_neutral = [c for c in compounds if abs(c) >= 0.05]
                if non_neutral:
                    filtered_mean = sum(non_neutral) / len(non_neutral)
                    score_filtered = max(-100.0, min(100.0, filtered_mean * 200.0))
                else:
                    score_filtered = 0.0

                # Model 3: Non-Linear Power Root
                sign = 1 if raw_mean >= 0 else -1
                score_root = sign * math.sqrt(abs(raw_mean)) * 100.0
                score_root = max(-100.0, min(100.0, score_root))

                print(f"{name:<16} | {len(compounds):<8} | {score_raw:>10.2f}   | {score_filtered:>14.2f}   | {score_root:>10.2f}")

            except Exception as e:
                print(f"{name:<16} | ERROR: {e}")

    print("\n" + "=" * 85)

if __name__ == "__main__":
    run_scaling_diagnostic()