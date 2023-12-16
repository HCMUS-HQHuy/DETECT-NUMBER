import numpy
from  tqdm import tqdm


def KNN_predict(train_data, train_label, test_data, K):
    predict = []
    for sample in tqdm(test_data):
        distances = numpy.linalg.norm(train_data - sample, axis=1)
        indices = numpy.argsort(distances)[:K]
        k_nearest_labels = train_label[indices]
        most_common = numpy.bincount(k_nearest_labels).argmax()
        predict.append(most_common)
    return numpy.array(predict)
