import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense,Activation
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
import time
start = time.clock()
import warnings
warnings.filterwarnings("ignore")

'''

BPNN is used to predict the raw data in Liu's model
params: 
* number of hidden units:10
* optimizer: Adam
* Epochs: 1500
* batchsize: 128

'''
# Load Data #1
dta=pd.read_csv('C:/Users/dw/Desktop/Data1.csv')
# Load the reconstructed series from EMD
N = dta['reconstruction']
x1,x2,x3,x4 = dta['y(t-1)'],dta['y(t-2)'],dta['y(t-3)'],dta['y(t-4)'] # Inputs

# Training and testing sets
train_x1,test_x1=x1[4:1340],x1[1340:1440]
train_x2,test_x2=x2[4:1340],x2[1340:1440]
train_x3,test_x3=x3[4:1340],x3[1340:1440]
train_x4,test_x4=x4[4:1340],x4[1340:1440]
train_N,test_N=N[4:1340],N[1340:1440]
Train = pd.concat((train_x1,train_x2,train_x3,train_x4,train_N),axis=1)
Test = pd.concat((test_x1,test_x2,test_x3,test_x4,test_N),axis=1)
X = np.vstack([Train,Test])
X_scale = scale(X)
train_x_scaled = X_scale[:1336,:4]
train_y_scaled = X_scale[:1336,4]
test_x_scaled = X_scale[1336:,:4]
test_y_scaled = X_scale[1336:,4]

def make_model():
    model = Sequential()
    model.add(Dense(10, input_dim=4, init="uniform",activation="relu"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
model=make_model()

runtimes = 50
test_MAE,test_MSE = [],[]
train_MAE, train_MSE= [],[]
fit_,pred_  = np.zeros((1336,1)),np.zeros((100,1))

for i in range(runtimes):
    model.fit(train_x_scaled,train_y_scaled,nb_epoch=1500,batch_size=128,verbose=0)
    pred_scaled = model.predict(test_x_scaled)
    fit_scaled = model.predict(train_x_scaled)
    pred = (pred_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]
    fit = (fit_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]
    # Error measurements
    test_error_MAE = mean_absolute_error(test_N.tolist(), pred)
    test_error_MSE = mean_squared_error(test_N.tolist(), pred)
    train_error_MAE = mean_absolute_error(train_N.tolist(), fit)
    train_error_MSE = mean_squared_error(train_N.tolist(), fit)
    test_MAE.append(test_error_MAE)
    test_MSE.append(test_error_MSE)
    train_MAE.append(train_error_MAE)
    train_MSE.append(train_error_MSE)
    fit_ += fit
    pred_ += pred
    print('===================run %d times================' % (i+1))

print('Test MAE: %.3f' % np.array(test_MAE).mean())
print('Test MSE: %.3f' % np.array(test_MSE).mean())
print('Train MAE: %.3f' % np.array(train_MAE).mean())
print('Train MSE: %.3f' % np.array(train_MSE).mean())

pred = pred_/50
np.savetxt('C:/Users/dw/Desktop/temp.csv',pred)

end = time.clock()
print("final is in ",end-start)