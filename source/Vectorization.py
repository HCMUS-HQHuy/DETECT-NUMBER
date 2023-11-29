import numpy 

def vectorization(img): 
    vector = numpy.zeros(784)
    
    return vector

def get(data):
    arrayVector = numpy.zeros(data.shape[0])
    for i in range(data.shape[0]):
        arrayVector[i] = vectorization(data[i])
    return arrayVector