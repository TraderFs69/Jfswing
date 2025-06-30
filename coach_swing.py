
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import time

# Récupérer la liste S&P 500 depuis Wikipedia
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(sp500_url)
sp500_df = tables[0]
tickers = sp500_df["Symbol"].tolist()
print(f"Nombre de tickers S&P500 : {len(tickers)}")

results = []

for ticker in tickers:
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 200:
            continue

        df.dropna(inplace=True)
        # ATR
        atr = ta.atr(df["High"], df["Low"], df["Close"], length=10)

        # UT Bot trail approximation
        keyValue = 0.5
        upperBand = df["Close"] + keyValue * atr
        lowerBand = df["Close"] - keyValue * atr
        trailPrice = df["Close"].copy()
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > trailPrice.iloc[i-1]:
                trailPrice.iloc[i] = max(trailPrice.iloc[i-1], lowerBand.iloc[i])
            else:
                trailPrice.iloc[i] = min(trailPrice.iloc[i-1], upperBand.iloc[i])

        utBuy = (df["Close"] > trailPrice) & (df["Close"].shift(1) <= trailPrice.shift(1))

        # MACD
        macd = ta.macd(df["Close"], fast=5, slow=13, signal=4)
        macdCond = macd["MACD_5_13_4"] > macd["MACDs_5_13_4"]

        # Stochastique
        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=8, d=5)
        stochK, stochD = stoch["STOCHk_8_5_3"], stoch["STOCHd_8_5_3"]
        stochCond = (stochK > stochD) & (stochK < 50)

        # SMA 100 & 200
        sma100 = ta.sma(df["Close"], length=100)
        sma200 = ta.sma(df["Close"], length=200)

        # ADX
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        adxCond = adx["ADX_14"] > 20

        # OBV
        obv = ta.obv(df["Close"], df["Volume"])
        obvSMA20 = ta.sma(obv, length=20)
        obvCond = obv > obvSMA20

        buySignal = utBuy & macdCond & stochCond & adxCond & obvCond

        if buySignal.iloc[-1]:
            results.append(ticker)

    except Exception as e:
        print(f"Erreur sur {ticker} : {e}")
    time.sleep(1.2)

print("Signaux détectés :", results)
pd.DataFrame(results, columns=["Ticker"]).to_csv("coach_swing_signals.csv", index=False)
