from loadData import load_mnist
import Vectorization 
import Downsampling
import Histogram

def outAns(data1, data2, f):
    print("vector_train: ", f(data1))
    print("vector_test : ", f(data2))

X_train, y_train = load_mnist('data/', kind = 'train')
X_test, y_test = load_mnist('data/', kind = 't10k')

print("Vectorization: ")
outAns(X_train, X_test, Vectorization.get)
print("Dowsampling: ")
outAns(X_train, X_test, Downsampling.get)
print("Histogram: ")
outAns(X_train, X_test, Histogram.get)
