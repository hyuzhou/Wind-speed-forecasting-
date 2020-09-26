import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time
'''

this file takes a weighted sum of the sub prediction results of multiple models

'''
start = time.clock()
dta=pd.read_csv('C:/Users/dw/Desktop/OptWeights.csv')

D1 = np.array(dta['D1']).reshape((100,1))
D2 = np.array(dta['D2']).reshape((100,1))

ARIMA_D1,ARIMA_D2 = np.array(dta['ARIMA1']).reshape((100,1)),np.array(dta['ARIMA2']).reshape((100,1))
BPNN_D1, BPNN_D2 = np.array(dta['BPNN1']).reshape((100,1)),np.array(dta['BPNN2']).reshape((100,1))
ELM_D1, ELM_D2 = np.array(dta['ELM1']).reshape((100,1)),np.array(dta['ELM2']).reshape((100,1))
ENN_D1,ENN_D2 = np.array(dta['ENN1']).reshape((100,1)),np.array(dta['ENN2']).reshape((100,1))
GRNN_D1,GRNN_D2 = np.array(dta['GRNN1']).reshape((100,1)),np.array(dta['GRNN2']).reshape((100,1))

weights_D1 = [0.291, 0.293, 0.229, 0.055, 0.132] # from OptWeights.py
weights_D2 = [0.233, 0.162, 0.175, 0.179, 0.252]

mat_D1 = np.hstack((ARIMA_D1,BPNN_D1,ELM_D1,ENN_D1,GRNN_D1))
mat_D2 = np.hstack((ARIMA_D2,BPNN_D2,ELM_D2,ENN_D2,GRNN_D2))

mat_weights_D1 = np.array(weights_D1).reshape((5,1))
mat_weights_D2 = np.array(weights_D2).reshape((5,1))

pred_D1 = np.dot(mat_D1,mat_weights_D1)
pred_D2 = np.dot(mat_D2,mat_weights_D2)

MAE_D1 = mean_absolute_error (D1,pred_D1)
MSE_D1 = mean_squared_error (D1,pred_D1)
print(MAE_D1,MSE_D1)
MAE_D2 = mean_absolute_error (D2,pred_D2)
MSE_D2 = mean_squared_error (D2,pred_D2)
print(MAE_D2,MSE_D2)

