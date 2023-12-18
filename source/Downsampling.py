import numpy 
from Normalize import normalize

def Downsample(img):
   reshaped_img = img.reshape(-1, 2, 2)
   averaged_values = numpy.mean(reshaped_img, axis=(1, 2))
   return averaged_values

def get(data):
   arrayVector = numpy.zeros((data.shape[0], 14*14))
   for i in range(0, data.shape[0]):
      arrayVector[i]=Downsample(normalize(data[i]))
    
   return arrayVector