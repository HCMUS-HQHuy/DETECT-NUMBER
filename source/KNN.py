import numpy

# def dis_euclid(vector1, vector2):
#     return numpy.sum((vector1 - vector2) ** 2)

def get_predict(arr):
    unique_elements, counts = numpy.unique(arr, return_counts=True)
    index_of_max_occurrence = numpy.argmax(counts)
    return int(unique_elements[index_of_max_occurrence])

def KNN_predict(train_data, train_label, sample, K):
    # val = numpy.full((K), numpy.inf)
    # candidates = numpy.zeros((K))
    # for i in range(train_data.shape[0]):
    #     dis = dis_euclid(train_data[i], sample)
    #     if val[K - 1] > dis:
    #         index_to_insert = numpy.searchsorted(val, dis)
    #         val = numpy.insert(val, index_to_insert, dis)
    #         val = val[:-1]
    #         candidates = numpy.insert(candidates, index_to_insert, train_lable[i])
    #         candidates = candidates[:-1]
    # return get_predict(candidates)
    distances = numpy.sum((train_data - sample) ** 2, axis=1)
    nearest_indices = numpy.argpartition(distances, K)[:K]
    candidates = train_label[nearest_indices]
    # print(candidates)
    return get_predict(candidates)