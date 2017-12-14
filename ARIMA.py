#More tutorial https://machinelearningmastery.com/make-sample-forecasts-arima-python/
import request, pandas as pd, numpy as np
from pandas import DataFrame
from pandas import read_csv
from io import StringIO
import time,json
from datetime import date
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error

#load Data
df_fx_data = read_csv('E:/Matin/Work Programming/Practice/Learn ARIMA/BOE-XUDLERD.csv')
print type(df_fx_data)
df_fx_data['Date'] = pd.to_datetime(df_fx_data['Date'], format='%Y-%m-%d')
indexed_df = df_fx_data.set_index('Date')
print indexed_df
ts = indexed_df['Value']
# plt.plot(ts.index.to_pydatetime(), ts.values)

#Resample Data to week
ts_week = ts.resample('W').mean()
ts_month = ts.resample('M').mean()
print "Ini Data TS WEEK"
print ts_week
print
print "Ini Data TS Month"
print ts_month
print "ts_month : " + str(len(ts_month)) + " ts_week : " + str(len(ts_week))
# plt.plot(ts_week.index.to_pydatetime(), ts_week.values)

# Check for stationarity
# Calculate the moving variances, plot the results and apply the Dickey-Fuller test on the time series
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=52,center=False).mean()
    rolstd = timeseries.rolling(window=52,center=False).std()

    #Plot rolling statistics
    orig = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue', label='Original')
    mean = plt.plot(rolmean.index.to_pydatetime(), rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.index.to_pydatetime(), rolstd.values, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    #Perform Dickey-Fuller test
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries,autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statsitic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value(%s)' % key] = value
    print dfoutput
    plt.show()

# test_stationarity(ts_week)
ts_week_log = np.log(ts_week)
# print ts_week_log
# test_stationarity(ts_week_log)

#Remove trend and seasonality with differencing
# decomposition = seasonal_decompose(ts_week)
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
#
# ts_week_log_select = ts_week_log[-80:]
#
# plt.subplot(411)
# plt.plot(ts_week_log_select.index.to_pydatetime(),ts_week_log_select.values, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(ts_week_log_select.index.to_pydatetime(),trend[-80:].values, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(ts_week_log_select.index.to_pydatetime(),seasonal[-80:].values,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(ts_week_log_select.index.to_pydatetime(),residual[-80:].values,label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

#Remove trend and seasonality with differencing
# print ts_week_log.shift()
ts_week_log_diff = ts_week_log - ts_week_log.shift()
# print ts_week_log_diff
# plt.plot(ts_week_log_diff.index.to_pydatetime(), ts_week_log_diff.values)
ts_week_log_diff.dropna(inplace=True)
# test_stationarity(ts_week_log_diff)

#Find optimal Parameters and build an ARIMA model
#to view more deep about find the optimal parameters you can watch https://www.youtube.com/watch?v=-vSzKfqcTDg

#1. ACF and PACF Plots is it for p,q,d parameters in ARIMA
lag_acf = acf(ts_week_log_diff, nlags=5)
lag_pacf = pacf(ts_week_log_diff, nlags=5, method='ols')

#Plot ACF
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0, linestyle='--', color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_week_log_diff)), linestyle='--', color='gray')
# plt.title('Autocorrelation Function')
#
#Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
# plt.show()

model = ARIMA(ts_week_log, order=(2,1,1))
result_ARIMA = model.fit(disp=-1)
# plt.plot(ts_week_log_diff.index.to_pydatetime(), ts_week_log_diff.values)
# plt.plot(ts_week_log_diff.index.to_pydatetime(), result_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %4f' % sum((result_ARIMA.fittedvalues - ts_week_log_diff) **2))
# plt.show()
#

#Measure the variance between the data and the values predicted by the model
# print result_ARIMA.summary()
# residuals = DataFrame(result_ARIMA.resid)
# residuals.plot(kind='kde')
# print residuals.describe()

#Scale Predictions
predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
print predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cums = predictions_ARIMA_diff.cumsum()
print
print "Cumsumnya :"
print predictions_ARIMA_diff_cums
predictions_ARIMA_log = pd.Series(ts_week_log.iloc[0], index=ts_week_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cums, fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plot Predictions
# plt.plot(ts_week.index.to_pydatetime(), ts_week.values)
# plt.plot(ts_week.index.to_pydatetime(), predictions_ARIMA.values, color='red')
# plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts_week) ** 2)/len(ts_week)))


#Visualize ARIMA
size = int(len(ts_week_log) - 15)
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()
print "Ini Data Test"
print
print test
print
print train
print "Printing Predicted vs Expected Values"
print
for t in range(len(test)):
    model = ARIMA(history, order=(2, 0, 1))
    mode_fit = model.fit(disp=-1)
    output = mode_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print "Predicted=%f, Expected=%f" % (np.exp(yhat), np.exp(obs))

error = mean_squared_error(test,predictions)

print
print "Printing Mean Squared Error of Predictions"
print "Test MSE: %.6f" % error

predictions_series = pd.Series(predictions, index=test.index)
# fig, ax = plt.subplots()
# plt.set(title='Spot Exchange Rate, Euro into USD', xlabel='Date', ylabel='Euro into USD')
plt.plot(ts_week[-15:], color='blue', label='observed')
plt.plot(np.exp(predictions_series), color='red', label='rolling one-step out-of-sample forecast')
# legend = plt.legend(loc='upper left')
# legend.get_frame().set_facecolor('w')
plt.show()