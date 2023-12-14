def accuracy(lables, imgVector): #lables lay tu ham loadData, imgVector lay tu cac ham getData

    correct_predictions = (lables == imgVector).sum()
    total_samples = len(lables)
    accuracy = correct_predictions / total_samples
    return accuracy
