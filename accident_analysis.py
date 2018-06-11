# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 22:06:33 2018

@author: vivek
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

dataset = pd.read_csv('PBL_Data_Final.csv')

X = dataset.iloc[:,3:].values


#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=-1,strategy="mean",axis=0)
imputer = imputer.fit(X[:,6].reshape(-1,1))
X[:,6] =  imputer.transform(X[:,6].reshape(-1,1)).reshape(1,-1)


#finding Correlation between attributes
Z = dataset.drop(['ACCIDENT_NO','ACCIDENT_DATE','ACCIDENT_TIME','LONGITUDE','LATITUDE','DAY_OF_WEEK'],axis=1).corr(method="pearson")

dataset.drop(['ACCIDENT_NO','ACCIDENT_DATE','ACCIDENT_TIME','LONGITUDE','LATITUDE','DAY_OF_WEEK'],axis=1).corr(method="pearson").style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


X_n = dataset[['LGA_NAME','TOTAL_PERSONS']]
X_n = X_n.iloc[:,:].values

imputer = Imputer(missing_values='NaN',strategy="most_frequent",axis=0)
imputer = imputer.fit(X_n[:,:])
X_n[:,:] =  imputer.transform(X_n[:,:])


plt.scatter(dataset['LONGITUDE'],dataset['LATITUDE'])
'''
#K-means clustering
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300)
    kmeans.fit(X_n)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,10),wcss)
plt.show()
'''


plt.scatter(X_n[:,0],X_n[:,1])
plt.xlabel('LGA_NAME')
plt.ylabel('Toatal_death')


time = dataset['ACCIDENT_TIME']
j=0
t=[]
for i in time:
   t.append(int(i[0:2])) 
   j+=1

import math
j=0
for i in t:
    t[j] = i//3
    j+=1
    
deathpertime={}
for i,j in zip(X_n[:,1],t):
    if j in deathpertime:
        deathpertime[j] += i
    else:
        deathpertime[j] = i
 
val=[]
for i in sorted(deathpertime):
    val.append(deathpertime[i])
    
label = ['12am-3am','3am-6am','6am-9am','9am-12pm','12pm-3pm','3pm-6pm','6pm-9pm','9pm-12am']

plt.figure(figsize=(10,5))
plt.bar([0,1,2,3,4,5,6,7],val) 
plt.xticks([0,1,2,3,4,5,6,7],label)
plt.xlabel('Time (3-hour period)')
plt.ylabel('Number of accidents')

explode=[0,0,0,0,0,0.1,0,0]
plt.pie(val,explode=explode,labels=label,autopct='%1.1f%%')
plt.axis('equal')

#accidents per month
m=[]
for i in dataset['ACCIDENT_DATE']:
    m.append(int(i[3:5]))

months={}
for i,j in zip(X_n[:,1],m):
    if j in months:
        months[j] += i
    else:
        months[j] = i
        
plt.bar(list(months.keys()),list(months.values()))


#Severity Index
totalaccidents = {}

totalaccidents["2012"] = RoadUserDf.loc[36,'Number of Total Road Accidents of Two-Wheelers - 2012']
totalaccidents["2012"] += RoadUserDf.loc[36,'Number of Total Road Accidents of Auto-Rickshaws - 2012']
totalaccidents["2012"] += RoadUserDf.loc[36,'Number of Total Road Accidents of Cars, Jeeps, Taxis - 2012']
totalaccidents["2012"] += RoadUserDf.loc[36,'Number of Total Road Accidents of Buses - 2012']
totalaccidents["2012"] += RoadUserDf.loc[36,'Number of Total Road Accidents of Trucks, Tempos, MAVs, Tractors - 2012']
totalaccidents["2012"] += RoadUserDf.loc[36,'Number of Total Road Accidents of Other Motor Vehicles - 2012']
totalaccidents["2012"] += RoadUserDf.loc[36,'Number of Total Road Accidents of Other Vehicles/Objects - 2012']

totalaccidents["2014"] = RoadUserDf.loc[36,'Two-Wheelers - Number of Road Accidents-Total - 2014']
totalaccidents["2014"] += RoadUserDf.loc[36,'Auto-Rickshaws - Number of Road Accidents-Total - 2014']
totalaccidents["2014"] += RoadUserDf.loc[36,'Cars, Jeeps,Taxis - Number of Road Accidents - Total - 2014']
totalaccidents["2014"] += RoadUserDf.loc[36,'Buses - Number of Road Accidents - Total - 2014']
totalaccidents["2014"] += RoadUserDf.loc[36,'Trucks, Tempos,MAVs,Tractors - Number of Road Accidents - Total - 2014']
totalaccidents["2014"] += RoadUserDf.loc[36,'Other Motor Vehicles - Number of Road Accidents - Total - 2014']
totalaccidents["2014"] += RoadUserDf.loc[36,'Other Vehicles/Objects - Number of Road Accidents - Total - 2014']

totalaccidents["2016"] = RoadUserDf.loc[36,'Motor Cycle/ Scooter - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Moped/Scootty - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Auto rickshaw - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Tempo - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'E-Rickshaw - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Motor Car - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Jeep - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Taxi - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Bus - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Truck/Lorry - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Articulated Vehicle/Trolly - Number of Road Accidents - Fatal - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Tractor - Number of Road Accidents - Total - 2016']
totalaccidents["2016"] += RoadUserDf.loc[36,'Other Motor Vehicles - Number of Road Accidents - Total - 2016']


totalkilled = {}

totalkilled["2012"] = RoadUserDf.loc[36,'Number of Persons Killed from accidents of Two-Wheelers - 2012']
totalkilled["2012"] += RoadUserDf.loc[36,'Number of Persons Killed from accidents of Auto-Rickshaws - 2012']
totalkilled["2012"] += RoadUserDf.loc[36,'Number of Persons Killed from accidents of Cars, Jeeps, Taxis - 2012']
totalkilled["2012"] += RoadUserDf.loc[36,'Number of Persons Killed from accidents of Buses - 2012']
totalkilled["2012"] += RoadUserDf.loc[36,'Number of Persons Killed from accidents of Trucks, Tempos, MAVs, Tractors - 2012']
totalkilled["2012"] += RoadUserDf.loc[36,'Number of Persons Killed from accidents of Other Motor Vehicles - 2012']
totalkilled["2012"] += RoadUserDf.loc[36,'Number of Persons Killed from accidents of Other Vehicles/Objects - 2012']


totalkilled["2014"] = RoadUserDf.loc[36,'Two-Wheelers - Number of Persons-Killed - 2014']
totalkilled["2014"] += RoadUserDf.loc[36,'Auto-Rickshaws - Number of Persons-Killed - 2014']
totalkilled["2014"] += RoadUserDf.loc[36,'Cars, Jeeps,Taxis - Number of Persons Killed - 2014']
totalkilled["2014"] += RoadUserDf.loc[36,'Buses - Number of Persons - Killed - 2014']
totalkilled["2014"] += RoadUserDf.loc[36,'Trucks,Tempos,MAVs,Tractors - Number of Persons - Killed - 2014']
totalkilled["2014"] += RoadUserDf.loc[36,'Other Motor Vehicles - Number of Persons - Killed - 2014']
totalkilled["2014"] += RoadUserDf.loc[36,'Other Vehicles/Objects - Number of Persons - Killed - 2014']

totalkilled["2016"] = RoadUserDf.loc[36,'Motor Cycle/ Scooter - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Moped/Scootty - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Auto rickshaw - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Tempo - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'E-Rickshaw - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Motor Car - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Jeep - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Taxi - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Bus - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Truck/Lorry - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Articulated Vehicle/Trolly - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Tractor - Number of Persons - Killed - 2016']
totalkilled["2016"] += RoadUserDf.loc[36,'Other Motor Vehicles - Number of Persons - Killed - 2016']

severity = {}
for i in totalkilled:
    severity[i] = (totalkilled[i]/totalaccidents[i])*100

plt.figure(figsize=(10,5))
plt.plot([2012,2014,2016],list(severity.values()))
plt.xticks([2012,2014,2016])
plt.yticks([28,29,30,31])
plt.title('Accident Severity Index (Total deaths per 100 accidents)')
plt.xlabel('Year')
plt.ylabel('Accident Severity Index')


### Road-user data
FaultType = pd.read_csv('datafile_4.csv')
FaultType = FaultType.drop(FaultType.index[[34,37]])


faulttype = {}
faulttype["Driver's Fault"] = FaultType.loc[36,'Fault of Driver-Total No. of Road Accidents - 2014'] 
faulttype["Cyclist's Fault"] = FaultType.loc[36,'Fault of Cyclist-Total No. of Road Accidents - 2014'] 
faulttype["Vehicle Condition"] = FaultType.loc[36,'Defect in Condition of Motor Vehicle-Total No. of Road Accidents - 2014'] 
faulttype["Road Condition"] = FaultType.loc[36,'Defect in Road Condition-Total No. of Road Accidents - 2014'] 
faulttype["Weather Condition"] = FaultType.loc[36,'Weather Condition-Total No. of Road Accidents - 2014'] 
faulttype["Passenger's Fault"] = FaultType.loc[36,'Fault of Passenger-Total No. of Road Accidents - 2014'] 
faulttype["Poor Light"] = FaultType.loc[36,'Poor light-Total No. of Road Accidents - 2014'] 
faulttype["Stray Animals"] = FaultType.loc[36,'Stray animals-Total No. of Road Accidents - 2014'] 
faulttype["Others"] = FaultType.loc[36,'Other causes/ Causes not known-Total No. of Road Accidents - 2014'] 


val = list(faulttype.values())
total = sum(val)
for i in range(0,9):
    val[i] = format(val[i]*100/total,'.2f')

plt.figure(figsize=(10,8))
plt.pie(list(faulttype.values()))
plt.axis('equal')
plt.xlabel('Accidents in 2014')

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
label = [list(pair) for pair in zip(list(faulttype.keys()),val)]
plt.legend(label,loc="best")
plt.show()

# Involvement of different types of vehicles

vehicletype = {}
vehicletype["2-Wheeler"] = RoadUserDf.loc[36,'Motor Cycle/ Scooter - Number of Road Accidents - Total - 2016']
vehicletype["2-Wheeler"] += RoadUserDf.loc[36,'Moped/Scootty - Number of Road Accidents - Total - 2016']

vehicletype["3-Wheeler"] = RoadUserDf.loc[36,'Auto rickshaw - Number of Road Accidents - Total - 2016']
vehicletype["3-Wheeler"] += RoadUserDf.loc[36,'Tempo - Number of Road Accidents - Total - 2016']
vehicletype["3-Wheeler"] += RoadUserDf.loc[36,'E-Rickshaw - Number of Road Accidents - Total - 2016']


vehicletype["4-Wheeler"] = RoadUserDf.loc[36,'Motor Car - Number of Road Accidents - Total - 2016']
vehicletype["4-Wheeler"] += RoadUserDf.loc[36,'Jeep - Number of Road Accidents - Total - 2016']
vehicletype["4-Wheeler"] += RoadUserDf.loc[36,'Taxi - Number of Road Accidents - Total - 2016']

vehicletype["Heavy Vehicle"] = RoadUserDf.loc[36,'Bus - Number of Road Accidents - Total - 2016']
vehicletype["Heavy Vehicle"] += RoadUserDf.loc[36,'Truck/Lorry - Number of Road Accidents - Total - 2016']
vehicletype["Heavy Vehicle"] += RoadUserDf.loc[36,'Articulated Vehicle/Trolly - Number of Road Accidents - Total - 2016']
vehicletype["Heavy Vehicle"] += RoadUserDf.loc[36,'Tractor - Number of Road Accidents - Total - 2016']

vehicletype["Other Vehicle"] = RoadUserDf.loc[36,'Other Motor Vehicles - Number of Road Accidents - Total - 2016']

#plot vehicle-type

plt.figure(figsize=(10,8))
plt.pie(list(vehicletype.values()),labels=list(vehicletype.keys()),autopct='%1.2f%%')
plt.axis('equal')
plt.xlabel('Types of Vehicles Involved in Accidents in 2016')

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.show()



#time series analysis
dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%y')
tdata = pd.read_csv('PBL_Data_Final.csv')
tdata = tdata[['ACCIDENT_DATE','TOTAL_PERSONS']]
tdata.to_csv('NewData.csv') 

tdata = pd.read_csv('NewData.csv', parse_dates=['ACCIDENT_DATE'],index_col='ACCIDENT_DATE',date_parser=dateparse)
tdata.head()
tdata = tdata.drop(['Unnamed: 0'],axis=1)

tk= tdata['TOTAL_PERSONS']
tk.head(10)

dictime = {}

for i in tk.index:
    if i not in dictime.keys():
        dictime[i] = pd.Series(tk[i]).sum()


ts = pd.Series(data=list(dictime.values()),index=list(dictime.keys()))
ts = ts.sort_index()

plt.plot(ts['2010'])

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(ts)

ts_log = np.log(ts)
plt.plot(ts_log)

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

ts_log_diff2 = ts_log_diff - ts_log_diff.shift()
plt.plot(ts_log_diff2)

ts_log_diff2.dropna(inplace=True)
test_stationarity(ts_log_diff2)

ts_log_diff3 = ts_log_diff2 - ts_log_diff2.shift()
plt.plot(ts_log_diff3)

ts_log_diff3.dropna(inplace=True)
test_stationarity(ts_log_diff3)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
#plt.xticks(np.arange(0,21, 0.5))
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
#plt.xticks(np.arange(0,21, 0.5))
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#p=0.7 q=0.7

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(1, 1, 1))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts['2011'])
plt.plot(predictions_ARIMA['2011'])
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))