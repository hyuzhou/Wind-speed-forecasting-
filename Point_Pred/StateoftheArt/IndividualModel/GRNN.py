import numpy as np
import math
import pandas as pd
import time
start = time.clock()

'''

GRNN is used to predict the raw data in Liu's model
there no parameter needed to be set subjectively

'''
def load_data(filename):
    '''load data
    input:  file_path(string)
    output: feature(mat)
            label(mat)
    '''
    f = open(filename)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    return np.mat(feature), np.mat(label).T

def distance(X, Y):
    '''
    calculate the distance
    '''
    return np.sqrt(np.sum(np.square(X - Y), axis=1))

def distance_mat(trainX, testX):
    '''
    input:trainX(mat)
          testX(mat)
    output:Euclidean_D(mat)
    '''
    m, n = np.shape(trainX)
    p = np.shape(testX)[0]
    Euclidean_D = np.mat(np.zeros((p, m)))
    for i in range(p):
        for j in range(m):
            Euclidean_D[i, j] = distance(testX[i, :], trainX[j, :])[0, 0]
    return Euclidean_D

def Gauss(Euclidean_D, sigma):
    '''
    input:Euclidean_D(mat)
          sigma(float)
    output:Gauss(mat)
    '''
    m, n = np.shape(Euclidean_D)
    Gauss = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            Gauss[i, j] = math.exp(- Euclidean_D[i, j] / (2 * (sigma ** 2)))
    return Gauss

def sum_layer(Gauss, trY):
    m, l = np.shape(Gauss)
    n = np.shape(trY)[1]
    sum_mat = np.mat(np.zeros((m, n + 1)))
    for i in range(m):
        sum_mat[i, 0] = np.sum(Gauss[i, :], axis=1)
    for i in range(m):
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += Gauss[i, s] * trY[s, j]
            sum_mat[i, j + 1] = total
    return sum_mat

def output_layer(sum_mat):
    '''
    input:sum_mat(mat)
    output:output_mat(mat)
    '''
    m, n = np.shape(sum_mat)
    output_mat = np.mat(np.zeros((m, n - 1)))
    for i in range(n - 1):
        output_mat[:, i] = sum_mat[:, i + 1] / sum_mat[:, 0]
    return output_mat

print('------------------------1. Load Data----------------------------')
feature,label = load_data('C:/Users/dw/Desktop/D1.txt')

print('--------------------2.Train Set and Test Set--------------------')
trX = feature[0:-100,:]
trY = label[0:-100,:]
teX = feature[-100:,:]
teY = label[-100:,:]

print('---------------------3. Output of Hidden Layer------------------')
Euclidean_D = distance_mat(trX,teX)
Gauss = Gauss(Euclidean_D,0.1)

print('---------------------4. Output of Sum Layer---------------------')
sum_mat = sum_layer(Gauss,trY)

print('---------------------5. Output of Output Layer------------------')
output_mat = output_layer(sum_mat)

print(output_mat)
np.savetxt('C:/Users/dw/Desktop/temp.csv',output_mat)

end = time.clock()
print("final is in ",end-start)
