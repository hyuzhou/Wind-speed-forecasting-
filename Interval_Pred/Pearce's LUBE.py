import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import scale
from math import e
import warnings
import time 
start = time.clock()

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('C:/Users/dw/Desktop/Data1.csv')

y = data['Wind speed']
x1,x2,x3,x4 = data['x1'],data['x2'],data['x3'],data['x4']

train_x1,test_x1=x1[4:1340],x1[1340:1440]
train_x2,test_x2=x2[4:1340],x2[1340:1440]
train_x3,test_x3=x3[4:1340],x3[1340:1440]
train_x4,test_x4=x4[4:1340],x4[1340:1440]
train_y,test_y = y[4:1340],y[1340:1440]

Train = pd.concat((train_x1,train_x2,train_x3,train_x4,train_y),axis=1)
Test = pd.concat((test_x1,test_x2,test_x3,test_x4,test_y),axis=1)
X = np.vstack([Train,Test])

# normalize data 
X_scale = scale(X)
train_x_scaled = X_scale[:1336,:4]
train_y_scaled = np.stack((X_scale[:1336,4],X_scale[:1336,4]),axis=1)
test_x_scaled = X_scale[1336:,:4]
test_y_scaled = X_scale[1336:,4]


lambda_ = 0.01
alpha_ = 0.05
soften_ = 160.
n_ = 100

# define loss fn
def lossf(y_true, y_pred):

    y_true = y_true[:, 0]
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]

    K_HU = tf.maximum(0., tf.sign(y_u - y_true))
    K_HL = tf.maximum(0., tf.sign(y_true - y_l))
    K_H = tf.multiply(K_HU, K_HL)

    K_SU = tf.sigmoid(soften_ * (y_u - y_true))
    K_SL = tf.sigmoid(soften_ * (y_true - y_l))
    K_S = tf.multiply(K_SU, K_SL)

    MPIW_c = tf.reduce_sum(tf.multiply((y_u - y_l), K_H)) / tf.reduce_sum(K_H)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)

    Loss_S = MPIW_c + lambda_ * n_ / (alpha_ * (1 - alpha_)) * tf.maximum(0., (1 - alpha_) - PICP_S)

    return Loss_S

def NNmodel():
    model = Sequential()
    model.add(Dense(40, input_dim=4, init="uniform",activation="relu"))
    model.add(Dense(2,activation='linear',init='uniform',bias_initializer=keras.initializers.Constant(value=[3.,-3.])))
    opt = keras.optimizers.Adam(lr=0.02, decay=0.01)
    model.compile(loss=lossf, optimizer=opt)
    return model
model = NNmodel()

result_loss = []

result = model.fit(train_x_scaled,train_y_scaled, epochs=2000, batch_size=n_, verbose=0)
result_loss.append(result.history['loss'])
y_pred_scaled = model.predict(test_x_scaled,verbose=0)
y_U_scaled = y_pred_scaled[:,0]
y_L_scaled = y_pred_scaled[:,1]
y_U = (y_U_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]
y_L = (y_L_scaled * X.std(axis=0)[-1])+X.mean(axis=0)[-1]

K_u = y_U > test_y
K_l = y_L < test_y
print('PICP:', np.mean(K_u * K_l))
print('PINAW:', np.round(np.mean(y_U - y_L),3)/8)

end = time.clock()
print("final is in ",end-start)
    

    
 



    


    