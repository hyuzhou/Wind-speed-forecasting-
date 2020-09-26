import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Activation
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error,mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import time
start = time.clock()

'''
this file contains object to perform point prediction for the raw data

input features: 
(1) Data#1(8inputs): 
            the prediction results of the linear and nonlinear components (x1, x8)
            the past 4 observations of the raw data (x2, x3, x4, x5)
            the past 2 observations of the nonlinear component (x6, x7)
(2) Data #2(9inputs): 
            the prediction results of the linear and nonlinear components 
            the past 4 observations of the raw data 
            the past 3 observations of the nonlinear component

parameters of the NN:
* number of hidden layer: 1
* hidden units: 12
* optimizer: Adam
* epoch: 1500
* batch_size: 128
* activation of hidden layer: ReLu

'''

# Load data
dta=pd.read_csv('C:/Users/dw/Desktop/Data1.csv')

y = dta['Wind speed']
x1 = dta['Linear_Pred'] # the prediction results of the linear component
x2,x3,x4,x5 = dta['y(t-1)'],dta['y(t-2)'],dta['y(t-3)'],dta['y(t-4)'] # the past 4 observations of the raw data
x6,x7= dta['Nonlinear(t-1)'],dta['Nonlinear(t-2)'] # the past 2 observations of the nonlinear component
x8 = dta['Nonlinear_Pred'] # the prediction results of the nonlinear component

# Construct training and testing sets
train_y,test_y=y[4:1340],y[1340:1440]
train_x1,test_x1=x1[4:1340],x1[1340:1440]
train_x2,test_x2=x2[4:1340],x2[1340:1440]
train_x3,test_x3=x3[4:1340],x3[1340:1440]
train_x4,test_x4=x4[4:1340],x4[1340:1440]
train_x5,test_x5=x5[4:1340],x5[1340:1440]
train_x6,test_x6=x6[4:1340],x6[1340:1440]
train_x7,test_x7 = x7[4:1340],x7[1340:1440]
train_x8,test_x8 = x8[4:1340],x8[1340:1440]
Train = pd.concat((train_x1,train_x2,train_x3,train_x4,train_x5,train_x6,train_x7,train_x8,train_y),axis=1)
Test = pd.concat((test_x1,test_x2,test_x3,test_x4,test_x5,test_x6,test_x7,test_x8,test_y),axis=1)
X = np.vstack([Train,Test])

# Standardize
X_scale = scale(X)
train_x_scaled = X_scale[:1336,:8]
train_y_scaled = X_scale[:1336,8]
test_x_scaled = X_scale[1336:,:8]
test_y_scaled = X_scale[1336:,8]

# Build NN model
def make_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, init="uniform",activation="relu"))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
model = make_model()

runtimes = 50
test_MAE,test_MSE = [],[]
train_MAE, train_MSE= [],[]
fit_, pred_  = np.zeros((1336,1)),np.zeros((100,1))
for i in range(runtimes):
    # Train model
    model.fit(train_x_scaled,train_y_scaled,nb_epoch = 1500, batch_size = 128, verbose = 0)
    # Predict
    pred_scaled = model.predict(test_x_scaled)
    fit_scaled = model.predict(train_x_scaled)
    pred = (pred_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]
    fit = (fit_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]
    # Forecasting metrics
    test_error_MAE = mean_absolute_error(test_y.tolist(), pred)
    test_error_MSE = mean_squared_error(test_y.tolist(), pred)
    train_error_MAE = mean_absolute_error(train_y.tolist(), fit)
    train_error_MSE = mean_squared_error(train_y.tolist(), fit)
    test_MAE.append(test_error_MAE)
    test_MSE.append(test_error_MSE)
    train_MAE.append(train_error_MAE)
    train_MSE.append(train_error_MSE)
    fit_ += fit
    pred_ += pred
    print('===================run %d times================' % (i+1))

# Print error measurements
print('Test MAE: %.3f' % np.array(test_MAE).mean())
print('Test MSE: %.3f' % np.array(test_MSE).mean())
print('Train MAE: %.3f' % np.array(train_MAE).mean())
print('Train MSE: %.3f' % np.array(train_MSE).mean())

# Save the average prediction results
pred = pred_/50
np.savetxt('C:/Users/dw/Desktop/temp.csv',pred)

# Calculate running time
end = time.clock()
print("final is in ",end-start)
