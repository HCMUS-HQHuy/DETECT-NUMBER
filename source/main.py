import matplotlib.pyplot as plt

from loadData import load_mnist
import Vectorization 
import Downsampling
import Histogram
from KNN import KNN_predict
import random

data_train, lable_train = load_mnist('data/', kind = 'train')
data_test, lable_test = load_mnist('data/', kind = 't10k')

pos = random.randint(0, 10000)
# print(pos)
sample = data_test[pos]
print("CHECK: ", lable_test[pos])
print("PREDICT: ", KNN_predict(Vectorization.get(data_train), lable_train, Vectorization.vectorization(sample), 10))
