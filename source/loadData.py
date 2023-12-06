import os
import numpy as np

def load_mnist(path, kind = 'train'):
    lables_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind, '%s-images-idx3-ubyte' % kind)

    with open(lables_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        lables = np.frombuffer(buffer, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(lables), 28, 28).astype(np.float64)
    
    return images, lables

# fig, ax = plt.subplots(nrows = 2, ncols = 5, sharex = True, sharey = True,)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0]
#     ax[i].imshow(img, cmap='Greys', interpolation = 'nearest')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()