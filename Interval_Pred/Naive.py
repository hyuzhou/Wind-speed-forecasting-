import pandas as pd
import numpy as np
import time
start = time.clock()

'''

1. Naïve method generates PIs by projecting the previous value in the future.
2. Here, the previous 100, 80, 60 samples are separately used to determine 
   the lower and upper bounds of the next observation for different PINC levels. 

'''
# Load data

dta=pd.read_csv('C:/Users/dw/Desktop/Data1.csv')
y = dta['Wind speed']

# 90% PINC is taken for example
train_y,test_y=y[1260:1340],y[1340:1440]

# Output the lower and upper bounds
print(train_y.min(),train_y.max())

end = time.clock()
print("final is in ",end-start)