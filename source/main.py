import matplotlib.pyplot as plt
import os
import numpy as np
import gzip

def load_mnist(path, kind = 'train'):
    lables_path = os.path.join(path, '%s-lables-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(lables_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        lables = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(lables), 28, 28).astype(np.float64)
    
    return images, lables

X_train, y_train = load_mnist('data/', kind = 'train')
print('Rows: %d, colums: %d' %(X_train.shape[0], X_train.shape[1]))

fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap='Greys', interpolation = 'nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()