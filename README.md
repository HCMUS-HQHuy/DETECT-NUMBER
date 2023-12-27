# Handwritten Digit Recognition
Handwritten Digits Recognition using KNN (k-nearest neighbors) classifier, Average model, ANN (artificial neural network)

# Describe

## Dataset (Training and Testing)
MNIST handwritten digits dataset is used for this. It contains 60000 images for training and 10000 images for testing, each images size 28x28 pixel square
Download dataset: http://yann.lecun.com/exdb/mnist/

## Future extraction

### 1) Flatten data
Flatten each 2D image size 28x28 square to 1D size 784.

### 2) Histogram

### 3) Downsample
Downsampling image 28x28 to nxn pixel (n = 4 or 7). 
## Classifier

### 1) KNN (K - nearest neighbors)
KNN algorithm is one of the more simple techniques used in machine learning, which:
*  Find K instances in the training set are nearest distance to a test data. Here we will use Euclidean distance.
*  Get the class with highest frequency from the K - most nearest distance above
### 2) Average models 
This algorithm;
* Calculate 10 models which represent for 10 classes 0 - 9, each models is the mean value of training dataset which has same label.
* Calculate distance between 10 models and test data.
* Get the class with minimum distance
### 3) ANN (Artificial Neural Network) 
I build an ANN using tensorflow.keras with 3 layers:
* Input-layer 
* 1 hidden-layer with 256 neurals
* Output-layer with 10 neurals represent for 10 class 0 - 9
```
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (img_train.shape[1:])))  # input-layer
model.add(tf.keras.layers.Dense(256, activation = 'relu'))  # hidden-layer with 256 neural
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))  # output-layer
```
training 
```
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(img_train, y_train, epochs = 10, verbose = 2)
```

# Accuracy compared

|              | Flatten | Histogram | Downsample 4x4 | Downsample 7x7 |
| ------------ | ------- | --------- | -------------  | -------------- |
| KNN (k = 100)| 94.44%  | 33.68%    |     80.78%     | 93.12%         |
| KNN (k = 200)| 92.91%  | 33.31%    | 79.48%         | 91.94%         |
| KNN (k = 300)| 91.80%  | 32.68%    | 78.36%         | 91.10%         |
| Average model| 82.03%  | 26.12%    | 64.02%         | 78.10%         |
| ANN          | 98.52%  | 34.65%    | 88.02%         | 97.38%         |

<p align="left">
  <img src="../master/image.png" width="300"/>
</p>

[Sorry for my bad English :'(]
