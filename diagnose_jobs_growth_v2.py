import pandas_datareader.data as web
import datetime

end = datetime.datetime.now()
start = end - datetime.timedelta(days=365 * 5)
STALENESS_THRESHOLD_DAYS = 120  # flag anything not updated in the last ~4 months

# ============================================================
# JOBS — replacement candidates for the two dead series + fresher
# alternates for USD/EUR in case newer editions exist
# ============================================================
UNEMPLOYMENT_CANDIDATES = {
    "USD_v1": "LRHUTTTTUSM156S",
    "USD_v2_alt": "UNRATE",  # standard US unemployment rate, near-certainly current
    "EUR_v1": "LRHUTTTTEZM156S",
    "EUR_v2_alt": "LRHUTTTTEA19M156S",  # euro area 19 variant, alternate coding
    "GBP": "LRHUTTTTGBM156S",
    "JPY": "LRHUTTTTJPM156S",
    "CAD": "LRHUTTTTCAM156S",
    "AUD": "LRHUTTTTAUM156S",
    "NZD_v1": "LRHUTTTTNZM156S",
    "NZD_v2_alt": "LMUNRRTTNZM156S",
    "CHF_v1": "LRHUTTTTCHM156S",
    "CHF_v2_alt": "LMUNRRTTCHM156S",
}

# ============================================================
# GROWTH — abandoning OECD Business Confidence entirely.
# Testing: (a) OECD Composite Leading Indicator, (b) raw GDP growth
# as a fallback, for each currency
# ============================================================
GROWTH_CANDIDATES = {
    "USD_CLI": "USALOLITONOSTSAM",
    "USD_GDP": "GDPC1",
    "EUR_CLI": "EA19OLITONOSTSAM",
    "EUR_GDP": "CLVMEURSCAB1GQEA19",
    "GBP_CLI": "GBROLITONOSTSAM",
    "GBP_GDP": "NGDPRSAXDCGBQ",
    "JPY_CLI": "JPNLOLITONOSTSAM",
    "JPY_GDP": "JPNRGDPEXP",
    "CAD_CLI": "CANLOLITONOSTSAM",
    "CAD_GDP": "NGDPRSAXDCCAQ",
    "AUD_CLI": "AUSLOLITONOSTSAM",
    "AUD_GDP": "NGDPRSAXDCAUQ",
    "NZD_CLI": "NZLLOLITONOSTSAM",
    "CHF_CLI": "CHELOLITONOSTSAM",
}


def check_series(name_dict, category_label):
    print(f"\n{'='*70}")
    print(f"CHECKING: {category_label}")
    print(f"{'='*70}")
    for label, series_id in name_dict.items():
        try:
            df = web.DataReader(series_id, 'fred', start, end)
            valid_points = df[series_id].dropna()
            if len(valid_points) >= 2:
                latest_date = valid_points.index[-1]
                days_stale = (end - latest_date.to_pydatetime().replace(tzinfo=None)).days
                latest_val = valid_points.iloc[-1]
                status = "✅ CURRENT" if days_stale <= STALENESS_THRESHOLD_DAYS else f"⚠️ STALE ({days_stale}d old)"
                print(f"  {status}  {label} ({series_id}): {len(valid_points)} pts, "
                      f"latest={latest_val:.2f} on {latest_date.strftime('%Y-%m-%d')}")
            else:
                print(f"  ⚠️ {label} ({series_id}): only {len(valid_points)} points — likely broken")
        except Exception as e:
            print(f"  ❌ {label} ({series_id}): FAILED — {type(e).__name__}: {str(e)[:80]}")


check_series(UNEMPLOYMENT_CANDIDATES, "JOBS — UNEMPLOYMENT RATE (with replacements)")
check_series(GROWTH_CANDIDATES, "GROWTH — CLI vs GDP (new candidates)")