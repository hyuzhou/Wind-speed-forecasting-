import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs

import time
start = time.clock()

'''

this file contains object to extract the linear component of raw data
this method is based on the secondary decomposition using EMD and SSA

'''

# load SourceData#1.csv
dta = pd.read_csv('C:/Users/dw/Desktop/Data1.csv')
# read the wind speed series
ts = np.array(dta.iloc[:,0])

# EMD decomposition
decomposer = EMD(ts)
imfs = decomposer.decompose()
plot_imfs(ts,imfs)
arr = np.vstack((imfs,ts))
dataframe = pd.DataFrame(arr.T)
arr1 = arr.T[:,2:8]
# linear subseries: IMF3 to IMF7, and Res
linear_subseries = arr1.sum(1)
# nonlinear subseries: IMF1 and IMF2
arr2 = arr.T[:,0:2]
nonlinear_subseries = arr2.sum(1)

# Secondary decomposition of nonlinear subseries using SSA
series = nonlinear_subseries - np.mean(nonlinear_subseries)
windowLen = 12
seriesLen = len(series)
K = seriesLen - windowLen + 1
X = np.zeros((windowLen, K))
for i in range(K):
    X[:, i] = series[i:i + windowLen]
U, sigma, VT = np.linalg.svd(X, full_matrices = False)

for i in range(VT.shape[0]):
    VT[i, :] *= sigma[i]
A = VT
rec = np.zeros((windowLen, seriesLen))
for i in range(windowLen):
    for j in range(windowLen - 1):
        for m in range(j + 1):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (j + 1)
    for j in range(windowLen - 1, seriesLen - windowLen + 1):
        for m in range(windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= windowLen
    for j in range(seriesLen - windowLen + 1, seriesLen):
        for m in range(j - seriesLen + windowLen, windowLen):
            rec[i, j] += A[i, j - m] * U[m, i]
        rec[i, j] /= (seriesLen - j)

plt.figure()
for i in range(10):
    ax = plt.subplot(5, 2, i + 1)
    ax.plot(rec[i, :])

# reconstruct
main_trend = rec[0,:]+rec[1,:]+rec[2,:]+rec[3,:]+rec[4,:]+rec[5,:]

# linear component construction
linear = linear_subseries + main_trend
# print(arr.T[:,-1])
nonlinear = arr.T[:,-1] - linear
df = pd.DataFrame(np.vstack((linear,nonlinear)))
# print(df.T)
# save the linear and nonlinear components,
#  the first column is linear component and the second column is nonlinear component
df.T.to_csv('C:/Users/dw/Desktop/temp.csv')

end = time.clock()
# record the computation time
print("final is in ",end-start)

