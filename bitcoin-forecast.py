import yfinance as yf
import pandas as pd
from autots import AutoTS
from datetime import date
import matplotlib.pyplot as plt


data = yf.download("BTC-USD",
                   start='2020-01-01',
                   end=date.today(),
                   progress=False)


data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date'])


model = AutoTS(
    forecast_length=60,
    frequency='infer',
    model_list="fast",
    validation_method="backwards",
    ensemble='simple'
)
model = model.fit(
    data,
    date_col='Date',
    value_col='Close',
    id_col=None,
)

prediction = model.predict()

forecast = prediction.forecast


print("BitCoin price forecasts")
forecast = forecast.reset_index()
print(forecast)

x1, y1 = data['Date'], data['Close']
x2, y2 = forecast['index'], forecast['Close']

plt.xlabel("date")
plt.ylabel("price")

plt.title("BitCoin price")

plt.plot(x1, y1, label='data')
plt.plot(x2, y2, color='red', label='forecast')

plt.legend()

plt.show()