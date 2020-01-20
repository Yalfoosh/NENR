import numpy as np
from typing import List, Tuple


class Rule:
    def __init__(self, input_units: int):
        self.__weights = np.random.normal(loc=0.0,
                                          scale=2 / np.sqrt(input_units + 1),
                                          size=input_units)
        self.__bias = np.random.uniform(low=-1., high=1.)

        self.__a = np.random.normal(size=input_units)
        self.__b = np.random.normal(size=input_units)

    # region Properties
    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value: np.ndarray):
        self.__weights = value

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, value: float):
        self.__bias = value

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, value: np.ndarray):
        self.__a = value

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, value: np.ndarray):
        self.__b = value
    # endregion

    def activation(self, inputs: np.ndarray):
        return np.reciprocal(1 + np.exp(self.b * (inputs - self.a)))

    def flattened_activation(self, inputs: np.ndarray):
        return np.product(self.activation(inputs), axis=-1)

    def application(self, inputs: np.ndarray):
        return inputs @ self.weights + self.bias

    def inference(self, inputs: np.ndarray):
        return self.application(inputs) * self.flattened_activation(inputs)

    def update(self, gradients: List or Tuple, learning_rates: List[float] or Tuple[float] or np.ndarray):
        if not isinstance(learning_rates, np.ndarray):
            learning_rates = np.array(learning_rates)

        gradients = [np.mean(gradients[i], axis=0) for i in range(len(gradients))]
        gradients *= learning_rates

        self.weights -= gradients[0]
        self.bias -= gradients[1]
        self.a -= gradients[2]
        self.b -= gradients[3]
