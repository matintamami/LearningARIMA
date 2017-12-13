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
for my_data in data:
    print my_data
