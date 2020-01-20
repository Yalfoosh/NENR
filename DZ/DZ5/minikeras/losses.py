import numpy as np


class Loss:
    def __init__(self):
        pass

    def call(self, y_real: np.ndarray, y_pred: np.ndarray):
        return np.abs(y_real - y_pred)

    def gradient(self, y_real: np.ndarray, y_pred: np.ndarray):
        return np.sign(y_real - y_pred)


class SquareError(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_real: np.ndarray, y_pred: np.ndarray):
        return np.square(y_real - y_pred) / 2

    def gradient(self, y_real: np.ndarray, y_pred: np.ndarray):
        return y_pred - y_real


class MeanSquareError(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_real: np.ndarray, y_pred: np.ndarray):
        return np.square(y_real - y_pred) / (2 * y_real.size)

    def gradient(self, y_real: np.ndarray, y_pred: np.ndarray):
        return (y_pred - y_real) / y_real.size


class AbsoluteError(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_real: np.ndarray, y_pred: np.ndarray):
        return np.abs(y_real - y_pred)

    def gradient(self, y_real: np.ndarray, y_pred: np.ndarray):
        return np.sign(y_pred - y_real)
