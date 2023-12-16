import matplotlib.pyplot as plt
import numpy
from Accuracy import accuracy
from loadData import load_mnist
import Vectorization 
import Downsampling
import Histogram
from KNN import KNN_predict
import random

def output(s, f, K):
    predict =  KNN_predict(f(data_train), lable_train, f(data_test), K)
    print(s + f"(K = {K}) -> Accuaracy = {accuracy(lable_test, predict) * 100 : .2f}%")

data_train, lable_train = load_mnist('data/', kind = 'train')
data_test, lable_test = load_mnist('data/', kind = 't10k')

output("Vectorization method ", Vectorization.get, 5)
output("Downsampling method ", Downsampling.get, 9)
output("Histogram method ", Histogram.get, 7)
