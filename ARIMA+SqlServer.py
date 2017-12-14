import pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import statsmodels
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from sqlalchemy import *

#Create Connection
db = create_engine('mssql+pyodbc://BankMega')
db.connect()

#load Data From sql Server
data = db.execute("SELECT TRANSACTION_DATETIME, PRODUCT, SUM(QUANTITY) AS qty FROM query_forecase GROUP BY PRODUCT,TRANSACTION_DATETIME")
df = DataFrame(data.fetchall(),dtype=float)
df[0] = pd.to_datetime(df[0],format='%y-%m-%d')
indexed_df = df.set_index(0)
ts = indexed_df[2]
# plt.plot(ts.index.to_pydatetime(), ts.values)

#Resample to Week Data
# ts_week = ts.resample('M').mean()
# print ts
# print ts_week