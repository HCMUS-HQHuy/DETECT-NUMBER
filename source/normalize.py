import numpy as np

def normalize(one_img):             #one_image la mang data[1]
    n=one_img.shape[0]
    after_normalize=np.zeros(n*n)
    after_normalize = numpy.divide(one_img,255)
    return after_normalize

    
