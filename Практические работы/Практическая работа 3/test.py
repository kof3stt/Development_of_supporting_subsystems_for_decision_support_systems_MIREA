import numpy as np


A=np.array([[1,2,-1],
            [2,2,5],
            [-1,5,-3]])

ev, eW = np.linalg.eigh(A)
L=np.diag(ev) 
np.dot(eW.T, eW)
np.dot(np.dot(eW, L), eW.T)
print(np.dot(np.dot(eW, L), eW.T))