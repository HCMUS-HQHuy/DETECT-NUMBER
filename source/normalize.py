import numpy as np

def normalize(one_img):             #one_image la mang data[1]
    n=one_img.shape[0]
    after_normalize=np.zeros(n,n)
    for i in range (0,n):
        for j in range (0,n):
            after_normalize[i][j]=one_img[i][j]/255
    return after_normalize

    