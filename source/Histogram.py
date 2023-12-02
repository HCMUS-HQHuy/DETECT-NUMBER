import numpy as np

def Histogram(img):
    n0 = img.shape[0]
    n1 = img.shape[1]
    vector = np.zeros(256)
    for i in range(n0):
        for j in range(n1):
            vector[img[i, j]] += 1
    return vector

def get(data):
    arrayVector = np.zeros((data.shape[0], 256))
    for i in range(data.shape[0]):
        arrayVector[i] = Histogram(data[i])
    return arrayVector
