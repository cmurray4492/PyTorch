import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg

quotes = yf.download('META', start='2011-1-1', end='2021-1-1')
model = AutoReg(quotes['Close'], lags=2)
model_fit = model.fit()
print(model_fit.params)
