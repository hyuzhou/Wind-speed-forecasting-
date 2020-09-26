import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time
start = time.clock()

'''

Weights are determined by minimizing the error variance of the testing dataset

'''
dta=pd.read_csv('C:/Users/dw/Desktop/OptWeights.csv')

D1 = dta['D1']
D2 = dta['D2']

ARIMA_D1,ARIMA_D2 = dta['ARIMA1'],dta['ARIMA2']
BPNN_D1, BPNN_D2 = dta['BPNN1'],dta['BPNN2']
ELM_D1, ELM_D2 = dta['ELM1'],dta['ELM2']
ENN_D1,ENN_D2 = dta['ENN1'],dta['ENN2']
GRNN_D1,GRNN_D2 = dta['GRNN1'],dta['GRNN2']

MSE_ARIMA_D1 = mean_squared_error(D1,ARIMA_D1)
MSE_BPNN_D1 = mean_squared_error(D1,BPNN_D1)
MSE_ELM_D1 = mean_squared_error(D1,ELM_D1)
MSE_ENN_D1 = mean_squared_error(D1,ENN_D1)
MSE_GRNN_D1 = mean_squared_error(D1,GRNN_D1)
MSE_ARIMA_D2 = mean_squared_error(D2,ARIMA_D2)
MSE_BPNN_D2 = mean_squared_error(D2,BPNN_D2)
MSE_ELM_D2 = mean_squared_error(D2,ELM_D2)
MSE_ENN_D2 = mean_squared_error(D2,ENN_D2)
MSE_GRNN_D2 = mean_squared_error(D2,GRNN_D2)

l1 = [MSE_ARIMA_D1,MSE_BPNN_D1,MSE_ELM_D1,MSE_ENN_D1,MSE_GRNN_D1]
l2 = [MSE_ARIMA_D2,MSE_BPNN_D2,MSE_ELM_D2,MSE_ENN_D2,MSE_GRNN_D2]

weights_D1 = []
weights_D2 = []
util1 = 0
util2 = 0

for i in l1:
    util1 += 1/i
for j in l2:
    util2 += 1/j

for k in range(5):
    weights_D1.append(round((1/l1[k])/util1,3))

for m in range(5):
    weights_D2.append(round((1/l2[m])/util2,3))

print(weights_D1)
print(weights_D2)