#dict_test = {'Thread-1': {'epoch': 0, 'cost': 0, 'updated': False}, 
#             'Thread-2': {'epoch': 0, 'cost': 0, 'updated': False}}

import threading
import numpy as np
import pandas as pd

#ThreadStatus=np.array([[]])
Threads=10
ThreadStatus= np.zeros(shape=(Threads,4))
print(ThreadStatus)

for count in range(0,10):
    ThreadStatus[count,0]=count
print(ThreadStatus)
    
    
for count in range(0,10):
    ThreadStatus[count]= [0,0,0,0]
    
    
    

for count in range(0, 10):
    ThreadStatus=np.append(ThreadStatus[count],np.array([count,0,0,0]))
    print(ThreadStatus)
        
print(ThreadStatus)


arr = np.append(arr, np.array([1,2,3]))

ThreadStatus=np.array([[0,0,0,0],
                      [1,0,0,1],
                      [2,0,0,1],
                      [3,0,0,1],
                      [4,0,0,1],
                      [5,0,0,1]]
                      )
print(ThreadStatus)

Loss=ThreadStatus.sum(axis=0)
print(Loss[2])
#AllSet=ThreadStatus==True
#x = bool

if 1 in ThreadStatus[:3]:
    print('True')
else:
    print('False')

x=len(ThreadStatus)
print(x)
Status=ThreadStatus[0,3]
print(Status)
#AverageLoss=np.sum(ThreadStatus)
ThreadStatus[0,3]=1
Status=ThreadStatus[0,3]
print(Status)

Loss=np.sum(ThreadStatus,axis=0)
StatusFlags=Loss[3]
if (StatusFlags/x)==1:
    print('All have updated - calculate the global loss')

print(Loss)

