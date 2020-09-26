import pandas as pd
import numpy as np
import time

'''

this file presents the proposed interval prediction module

params：ARIMA order: Data #1: (3,2,5)
                    Data #2: (2,3,2)
        no. of states of Markov chain: data #1: 10
                                       data #2: 10

95% PINC is taken for example

'''

start = time.clock()

# Load data
data0 = pd.read_csv("C:/Users/dw/Desktop/Data1.csv")
data = data0.iloc[:,3] # nonlinear subseries: the combined series of IMF1 and IMF2
data = np.array(data)
data_arr = np.array(data)
data_arr.sort()

min = min(data_arr)
max = max(data_arr)
print(min,max)

# Define the lower and upper of states
state_upper = [data_arr[144],data_arr[288],data_arr[3*144],data_arr[4*144],data_arr[5*144],data_arr[6*144],data_arr[7*144],data_arr[8*144],data_arr[9*144],max]
state_lower = [min,data_arr[144],data_arr[288],data_arr[3*144],data_arr[4*144],data_arr[5*144],data_arr[6*144],data_arr[7*144],data_arr[8*144],data_arr[9*144]]


state = np.zeros(data_arr.shape)

# States categorization: 10 states
for i in range(len(data_arr)):
    if (data[i] <= data_arr[144]):
        state[i] = 1
    elif (data[i] <= data_arr[288]) and (data[i] > data_arr[144]):
        state[i] = 2
    elif (data[i] <= data_arr[3*144]) and (data[i] > data_arr[288]):
        state[i] = 3
    elif (data[i] <= data_arr[4*144]) and (data[i] > data_arr[3*144]):
        state[i] = 4
    elif (data[i] <= data_arr[5*144]) and (data[i] > data_arr[4*144]):
        state[i] = 5
    elif (data[i] <= data_arr[6*144]) and (data[i] > data_arr[5*144]):
        state[i] = 6
    elif (data[i] <= data_arr[7*144]) and (data[i] > data_arr[6*144]):
        state[i] = 7
    elif (data[i] <= data_arr[8*144]) and (data[i] > data_arr[7*144]):
        state[i] = 8
    elif (data[i] <= data_arr[9*144]) and (data[i] > data_arr[8*144]):
        state[i] = 9
    elif (data[i] > data_arr[9*144]):
        state[i] = 10

# Save the states
# np.savetxt("C:/Users/dw/Desktop/Wind speed data/EX6/temp.csv",state_arr)

# Calculate the state transition matrix c
l = state.tolist()
a = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        for k in range(len(l)-1):
            if l[k:k+2] ==[i+1,j+1]:
                a[i,j] += 1
b = a.sum(axis=1)
c = np.zeros((10,10))
for i in range(a.shape[0]):
    c [i,:] = a[i,:]/b[i]

# Print the state transition matrix
# print(c)

# Calculate the probability distribution vectors
state_test = state[-101:]
l2 = state_test.tolist()
prob_vector = np.zeros((100,10))
for k in range(len(l2)-1):
    prob_vector[k, 0] = c[int(l2[k]-1), 0] *1
    prob_vector[k, 1] = c[int(l2[k]-1), 1] *1
    prob_vector[k, 2] = c[int(l2[k]-1), 2] *1
    prob_vector[k, 3] = c[int(l2[k]-1), 3] *1
    prob_vector[k, 4] = c[int(l2[k]-1), 4] *1
    prob_vector[k, 5] = c[int(l2[k] - 1), 5] * 1
    prob_vector[k, 6] = c[int(l2[k] - 1), 6] * 1
    prob_vector[k, 7] = c[int(l2[k] - 1), 7] * 1
    prob_vector[k, 8] = c[int(l2[k] - 1), 8] * 1
    prob_vector[k, 9] = c[int(l2[k] - 1), 9] * 1

# print(prob_vector)
# np.savetxt("C:/Users/dw/Desktop/Wind speed data/EX6/temp2.csv",prob_vector)

# PIs construction
nonlinear_PI_lower = np.zeros((100,1)) # the lower of PI
nonlinear_PI_upper = np.zeros((100,1))# the upper of PI

alpha = 0.05 # alpha should be set to 0.08 and 0.1 for 90% and 85% confidence level to achieve valid PICPs


for i in range(100):
    p = alpha / 2
    j = 0
    while prob_vector[i][j] < p:
        p -= prob_vector[i][j]
        j += 1
    nonlinear_PI_lower[i] = (p/prob_vector[i][j])* (state_upper[j]-state_lower[j]) + state_lower[j]

    p = alpha / 2
    j = 0

    while prob_vector[i][9-j] < p:
        p -= prob_vector[i][9-j]
        j += 1
    nonlinear_PI_upper[i] = state_upper[9-j] - (p/prob_vector[i][9-j])* (state_upper[9-j]-state_lower[9-j])

# np.savetxt("C:/Users/dw/Desktop/Wind speed data/EX6/temp.csv",nonlinear_PI_lower)
# np.savetxt("C:/Users/dw/Desktop/Wind speed data/EX6/temp1.csv",nonlinear_PI_upper)

# Read the ARIMA-PPs: the point predictions of the combined series of IMF1 and IMF2
linear_subseries_pred = data0['Linear_Subseries_Pred'][-100:]

# Combine the ARIMA-PPs and IFOMC-PIs
PI_lower = np.array(linear_subseries_pred).reshape(100,1)+ nonlinear_PI_lower
PI_upper = np.array(linear_subseries_pred).reshape(100,1)+ nonlinear_PI_upper

# Save the PIs
np.savetxt("C:/Users/dw/Desktop/temp.csv",PI_lower)
np.savetxt("C:/Users/dw/Desktop/temp1.csv",PI_upper)

end = time.clock()
print("final is in ",end-start)
