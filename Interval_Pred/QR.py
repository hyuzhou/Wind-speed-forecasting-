import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K
from sklearn.preprocessing import scale
import time

'''

we use BPNN to conduct quantile regression
parameters:
no. of hidden layer: 1
no. of hidden units: 10
activation of hidden layer: Relu
optimizer: Adadelta

'''
start = time.clock()

# Load data
dta=pd.read_csv('C:/Users/dw/Desktop/D1.csv')
y = dta['Wind speed']
x1 = dta['x1']
x2 = dta['x2']
x3 = dta['x3']
x4 = dta['x4']

# Construct the training and testing sets
train_y,test_y=y[4:1340],y[1340:1440]
train_x1,test_x1=x1[4:1340],x1[1340:1440]
train_x2,test_x2=x2[4:1340],x2[1340:1440]
train_x3,test_x3=x3[4:1340],x3[1340:1440]
train_x4,test_x4=x4[4:1340],x4[1340:1440]

Train = pd.concat((train_x1,train_x2,train_x3,train_x4,train_y),axis=1)
Test = pd.concat((test_x1,test_x2,test_x3,test_x4,test_y),axis=1)
X = np.vstack([Train,Test])
X_scale = scale(X)
train_x_scaled = X_scale[:1336,:4]
train_y_scaled = X_scale[:1336,4]
test_x_scaled = X_scale[1336:,:4]
test_y_scaled = X_scale[1336:,4]

# Define loss func
def loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def Model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=4, activation='relu'))
    model.add(Dense(1,activation='linear'))
    return model
x = np.arange(0,100)

quantile = [0.025,0.975]
runtimes = 50
pred_ = np.zeros((100,1))

for q in quantile:
    model = Model()
    for i in range(runtimes):
        model.compile(loss=lambda y, f: loss(q, y, f), optimizer='adadelta')
        model.fit(train_x_scaled,train_y_scaled, epochs=1500, batch_size=32, verbose=0)
        # Predict the quantile
        y_test_scaled = model.predict(test_x_scaled)
        y_pred = (y_test_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]
        # plt.plot(y_pred, label=q)
        # plt.plot(x,test_y)
        pred_ += y_pred
    print(q,pd.DataFrame(pred_/50))

end = time.clock()
print("final is in ",end-start)