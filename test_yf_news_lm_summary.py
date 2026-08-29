import yfinance as yf
import pysentiment2 as ps
import requests

TEST_INSTRUMENTS = {
    "Forex (Majors)": {"name": "EUR/USD", "ticker": "EURUSD=X"},
    "Forex (Crosses)": {"name": "GBP/JPY", "ticker": "GBPJPY=X"},
    "Global Stock Indices": {"name": "US 500 (S&P 500)", "ticker": "^GSPC"},
    "Precious Metals": {"name": "Gold", "ticker": "GC=F"},
    "Commodities": {"name": "Crude Oil (WTI)", "ticker": "CL=F"},
    "Treasuries": {"name": "US 10-Year Yield", "ticker": "^TNX"},
    "Crypto": {"name": "BTC/USD (Bitcoin)", "ticker": "BTC-USD"}
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

def run_news_audit():
    session = get_test_session()
    lm = ps.LM()
    
    print("=" * 70)
    print("🔍 AUDITING YAHOO FINANCE NEWS: LOUGHRAN-MCDONALD + SUMMARIES")
    print("=" * 70)

    for category, item in TEST_INSTRUMENTS.items():
        name, ticker = item["name"], item["ticker"]
        print(f"\n[{category}] -> {name} ({ticker})")
        print("-" * 50)
        
        try:
            asset = yf.Ticker(ticker, session=session)
            news_items = asset.news
            
            if not news_items:
                print("  ❌ No news items returned.")
                continue
                
            print(f"  ✅ Headlines Found: {len(news_items)}")
            
            total_polarity = 0
            scored_count = 0
            matched_count = 0
            
            # Print top 3 headlines + summaries as a sample
            for idx, article in enumerate(news_items[:3]):
                title = article.get('title') or article.get('content', {}).get('title', '')
                summary = article.get('summary') or article.get('content', {}).get('summary', '') or article.get('text', '')
                publisher = article.get('publisher', article.get('provider', {}).get('displayName', 'Unknown'))
                
                print(f"   {idx+1}. [{publisher}] {title}")
                if summary:
                    trunc_summary = (summary[:100] + '...') if len(summary) > 100 else summary
                    print(f"      ↳ Summary: {trunc_summary}")
                else:
                    print("      ↳ Summary: [No summary provided by Yahoo]")
                
            # Run LM Sentiment Analysis across up to 15 articles
            for article in news_items[:15]:
                title = article.get('title') or article.get('content', {}).get('title', '')
                summary = article.get('summary') or article.get('content', {}).get('summary', '') or article.get('text', '')
                
                if not title:
                    continue
                    
                # Stitch title and summary together for equal comparison
                full_text = f"{title}. {summary}" if summary else title
                
                tokens = lm.tokenize(full_text)
                score_dict = lm.get_score(tokens)
                
                if score_dict.get('Positive', 0) > 0 or score_dict.get('Negative', 0) > 0:
                    matched_count += 1
                total_polarity += score_dict.get('Polarity', 0)
                scored_count += 1
                
            if scored_count > 0:
                scaled_score = (total_polarity / scored_count) * 225.0
                density_ratio = matched_count / scored_count
                if density_ratio < 0.15:
                    scaled_score *= (density_ratio / 0.15)
                final_score = max(-100.0, min(100.0, scaled_score))
                print(f"\n  📊 LM Scored Articles: {scored_count} | Keyword Hits: {matched_count} | Scaled Score: {round(final_score, 2)}")
            else:
                print("  ⚠️ No articles could be scored.")
                
        except Exception as e:
            print(f"  ❌ Error fetching news: {e}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_news_audit()