import numpy 
from Normalize import normalize

def Downsample(img): 
    n=img.shape[0]                      #28 phan tu
    vector=numpy.zeros(14*14)
    index=0
    for i in range(0,n,2):          
       for j in range(0,n,2):
          vector[index]=(img[i,j] + img[i,j+1] + img[i+1,j] + img[i+1,j+1])/4
          index=index+1
    return vector

def get(data):
   arrayVector=numpy.zeros((data.shape[0], 14*14))
   for i in range(0, data.shape[0]):
      arrayVector[i]=Downsample(normalize(data[i]))
    
   return arrayVector