import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_predict):
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.x_train]
        nearest = np.argsort(distances)[:self.k]
        top_k_y = [self.y_train[index] for index in nearest]
        d = {}
        for cls in top_k_y:
            d[cls] = d.get(cls, 0) + 1
        d_list = list(d.items())
        d_list.sort(key=lambda x: x[1], reverse=True)
        return np.array(d_list[0][0])

    def __repr__(self):
        return "KNN(k={})".format(self.k)