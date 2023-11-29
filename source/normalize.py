import numpy as np

def normalize(data):
    n=data.shape[0]
    after_normalize=np.zeros(n,n)
    for i in range (0,n):
        for j in range (0,n):
            after_normalize[i][j]=data[i][j]/255
    return after_normalize

    