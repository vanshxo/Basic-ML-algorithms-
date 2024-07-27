from collections import Counter
import numpy as np
import pandas as pd

class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):

        votes = []
        for i in X_test:
            eucl = []
            for j in self.X_train:
                eucl.append(np.linalg.norm(i - j))
            eucl = list(enumerate(eucl))
            y_pred = sorted(eucl, key=lambda x: x[1])[0:self.n_neighbors]
            vote = []
            for k in y_pred:
                vote.append(self.y_train[k[0]])
            C = Counter(vote)

            votes.append(C.most_common()[0][0])

        return np.array(votes)