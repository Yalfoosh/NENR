import numpy as np


class Loss:
    def __init__(self):
        pass

    def call(self, y_real, y_pred):
        pass

    def gradient(self, y_real, y_pred):
        pass


class SquareError(Loss):
    def call(self, y_real, y_pred):
        return np.square(y_real - y_pred) / 2

    def gradient(self, y_real, y_pred):
        return y_pred - y_real
