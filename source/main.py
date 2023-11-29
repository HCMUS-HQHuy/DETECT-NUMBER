from loadData import load_mnist
import Vectorization 
import Downsampling
import Histogram

X_train, y_train = load_mnist('data/', kind = 'train')
X_test, y_test = load_mnist('data/', kind = 't10k')

# array = Vectorization.get(X_train)
# print(array)
# array = Vectorization.get(X_test)
