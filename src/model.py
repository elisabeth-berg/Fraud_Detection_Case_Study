from sklearn.ensemble import RandomForestClassifier
import numpy as np


class Model:

    def __init__(self):
        self.model = RandomForestClassifier(n_jobs=-1, max_features='sqrt',
                                            n_estimators=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        pre_predictions = self.model.predict_proba(X)

        predictions = np.delete(pre_predictions, 2, axis=1)

        weights = np.array([1, 2, 10])

        predictions = predictions / weights

        max_values = np.max(predictions, axis=1)


        predictions_with_max = np.concatenate([predictions, max_values.reshape(-1, 1)], axis=1)

        return np.around(predictions_with_max, decimals=4)
