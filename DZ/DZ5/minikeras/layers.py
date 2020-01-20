import numpy as np
import re
from typing import Dict

# TODO SERIALIZATION OF LAYER SHAPE AND WEIGHT

activation_regexi =\
    {
        "linear": re.compile(r"(?i)(l(in(ear)?)?)"),
        "sigmoid": re.compile(r"(?i)(sigm(oid)?)"),
        "relu": re.compile(r"(?i)(relu)")
    }


# region Base Classes
class Layer:
    def __init__(self, **kwargs):
        self.__id = (-1, True)
        self.__name = kwargs.get("name", "Layer")
        self._original_name = self.__name
        self._activation = None
        self.__trainable = False

    @property
    def id(self):
        return self.__id[0]

    @id.setter
    def id(self, value: int):
        if self.__id[1]:
            self.__id = (value, False)

        if self.__name == self._original_name:
            self.name = f"[{self.id}] {self._original_name}"

        else:
            print(f"Trying to set ID of layer {self.__id} (named {self.__name}), but it was already set and therefore "
                  f"cannot be changed!")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value: str = None):
        if value is None:
            value = f"[{self.id}] Layer"

        self.__name = value

    @property
    def activation(self):
        return self._activation

    @property
    def trainable(self):
        return self.__trainable

    @trainable.setter
    def trainable(self, value: bool):
        self.__trainable = value

    def build(self, input_units: int or None, **kwargs):
        return input_units

    def forward(self, inputs: np.ndarray, **kwargs):
        return inputs

    def backward(self, inputs: np.ndarray, output_gradient: np.ndarray, gradient_tape: Dict[int, np.ndarray]):
        return output_gradient

    def update(self, gradient_tape: Dict[int, np.ndarray], learning_rate: float, **kwargs):
        pass

    # Returns tuple string of "Layer name", class name, number of parameters, number of trainable parameters
    def summary(self):
        return f"{self.name}", self.__class__.__name__, 0, 0


class Activation(Layer):
    @staticmethod
    def parse(activation_name: str):
        for activation_string in activation_regexi:
            if activation_regexi[activation_string].match(activation_name):
                return string_to_class[activation_string]()

        raise KeyError(f"Wanted activation ({activation_name}) not found!")
# endregion


# region Layer Classes
class Input(Layer):
    def __init__(self, input_units: int, name: str = "Input", **kwargs):
        super().__init__(name="Input", **kwargs)

        if input_units is None:
            raise ValueError("Argument input_units cannot be None!")

        if not isinstance(input_units, int):
            raise TypeError(f"Argument input_units should be an integer (currently it is {type(input_units)})!")

        if input_units is not None:
            self.__input_units = input_units

        if name is not None:
            self.name = name

    def build(self, input_units: int or None, **kwargs):
        if self.__input_units is None:
            self.__input_units = input_units

        return self.__input_units

    def forward(self, inputs: np.ndarray, **kwargs):
        return inputs

    def backward(self, inputs: np.ndarray, output_gradient: np.ndarray, gradient_tape: Dict[int, np.ndarray]):
        return output_gradient

    def summary(self):
        return f"{self.name}", self.__class__.__name__ + f" (1 x {self.__input_units})", 0, 0


class Dense(Layer):
    def __init__(self, output_units: int, input_units: int or None = None, name: str = "Dense",
                 activation: Activation or str = "linear", **kwargs):
        super().__init__(name="Dense", **kwargs)
        self.__input_units = None
        self.__output_units = output_units
        self.__weights = None
        self.__biases = None

        if activation is None:
            activation = "linear"
        if isinstance(activation, str):
            activation = Activation.parse(activation)
        if isinstance(activation, Activation):
            self._activation = activation
        else:
            raise TypeError(f"Argument activation is of the wrong type: "
                            f"instead of {type(Activation)} or {type(None)} it is {type(activation)}!")

        if input_units is not None:
            self.__input_units = input_units

        if name is not None:
            self.name = name

    @property
    def weights(self):
        return self.__weights

    @property
    def biases(self):
        return self.__biases

    def build(self, input_units: int or None, **kwargs):
        if input_units is None:
            if self.__input_units is None:
                raise ValueError("Argument input_units cannot be None since the input_units parameter "
                                 "wasn't set on layer instantiation!")
            else:
                input_units = self.__input_units

        if not isinstance(input_units, int):
            raise TypeError(f"Argument input_units should be an integer (currently it is {type(input_units)})!")

        self.__weights = np.random.normal(loc=0.0,
                                          scale=2 / np.sqrt(input_units + self.__output_units),
                                          size=(input_units, self.__output_units))
        self.__biases = np.random.uniform(size=self.__output_units)

        self.trainable = True

        if self.__input_units != input_units:
            self.__input_units = input_units

        return self.__output_units

    def forward(self, inputs: np.ndarray, **kwargs):
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, inputs: np.ndarray, output_gradient: np.ndarray, gradient_tape: Dict):
        weight_gradient = np.outer(inputs, output_gradient.T)

        if output_gradient.size > self.__output_units:
            bias_gradient = np.mean(output_gradient, axis=1, keepdims=True)
        else:
            bias_gradient = output_gradient

        if gradient_tape[self.id] is None:
            gradient_tape[self.id] = np.array([weight_gradient]), np.array([bias_gradient])
        else:
            gradient_tape[self.id] = np.stack([gradient_tape[self.id][0][0], weight_gradient]),\
                                     np.stack([gradient_tape[self.id][1][0], bias_gradient])

        return np.dot(output_gradient, self.__weights.T)

    def update(self, gradient_tape: Dict[int, np.ndarray], learning_rate: float, **kwargs):
        if self.trainable:
            weight_gradient, bias_gradient = gradient_tape[self.id]

            if len(weight_gradient.shape) > len(self.__weights.shape):
                weight_gradient = np.mean(weight_gradient, axis=0)

            if len(bias_gradient.shape) > len(self.__biases.shape):
                bias_gradient = np.mean(bias_gradient, axis=0)

            self.__weights -= learning_rate * weight_gradient
            self.__biases -= learning_rate * bias_gradient

    def summary(self):
        parameters = 0

        if self.weights is not None:
            parameters += self.weights.size

        if self.biases is not None:
            parameters += self.biases.size

        return f"{self.name}", self.__class__.__name__ + f" ({self.__input_units} x {self.__output_units})",\
               parameters, parameters if self.trainable else 0


class Dropout(Layer):
    def __init__(self, dropout: float, input_units: int or None = None, name: str = "Dropout",
                 activation: str or Activation = "linear", **kwargs):
        super().__init__(name="Dropout", **kwargs)
        if dropout is None:
            raise ValueError("Argument dropout cannot be None!")

        if not isinstance(dropout, float):
            raise TypeError(f"Argument dropout should be a floating point value (it is currently {type(dropout)})!")

        if input_units is not None:
            if not isinstance(input_units, int):
                raise TypeError(f"Argument input units should be None or an integer value "
                                f"(it is currently {type(input_units)})!")

        self.__dropout = dropout

    def build(self, input_units: int, **kwargs):
        pass

# endregion


# region Activation Classes
class Linear(Activation):
    def __init__(self, **kwargs):
        super().__init__(name="Linear", **kwargs)

    def build(self, input_units: int or None, **kwargs):
        if input_units is None:
            raise ValueError("Argument input_units cannot be None!")

        if not isinstance(input_units, int):
            raise TypeError("Argument input_units should be an integer!")

        layer_index = kwargs.get("index", None)

        if layer_index is not None:
            self.name = f"[{layer_index}] Linear"

        return input_units

    def forward(self, inputs: np.ndarray, **kwargs):
        return inputs

    def backward(self, inputs: np.ndarray, output_gradient: np.ndarray, gradient_tape: Dict = None):
        return output_gradient

    def summary(self):
        return f"{self.name}", self.__class__.__name__, 0, 0


class Sigmoid(Activation):
    @staticmethod
    def __sigmoid(x):
        return np.reciprocal(1 + np.exp(-x))

    @staticmethod
    def __sigmoid_derivative(x, from_logits=False):
        if not from_logits:
            x = Sigmoid.__sigmoid(x)

        return x * (1. - x)

    def __init__(self, **kwargs):
        super().__init__(name="Sigmoid", **kwargs)

    def build(self, input_units: int or None, **kwargs):
        if input_units is None:
            raise ValueError("Argument input_units cannot be None!")

        if not isinstance(input_units, int):
            raise TypeError("Argument input_units should be an integer!")

        layer_index = kwargs.get("index", None)

        if layer_index is not None:
            self.name = f"[{layer_index}] Sigmoid"

        return input_units

    def forward(self, inputs: np.ndarray, **kwargs):
        return self.__sigmoid(inputs)

    def backwards(self, inputs: np.ndarray, output_gradient: np.ndarray, gradient_tape: Dict = None):
        return output_gradient * self.__sigmoid_derivative(inputs)

    def summary(self):
        return f"{self.name}", self.__class__.__name__, 0, 0


class ReLU(Activation):
    def __init__(self, **kwargs):
        super().__init__(name="ReLU", **kwargs)

    def build(self, input_units: int or None, **kwargs):
        if input_units is None:
            raise ValueError("Argument input_units cannot be None!")

        if not isinstance(input_units, int):
            raise TypeError("Argument input_units should be an integer!")

        layer_index = kwargs.get("index", None)

        if layer_index is not None:
            self.name = f"[{layer_index}] ReLU"

        return input_units

    def forward(self, inputs: np.ndarray, **kwargs):
        return np.maximum(0, inputs)

    def backward(self, inputs: np.ndarray, output_gradient: np.ndarray, gradient_tape: Dict = None):
        output = np.array(output_gradient, copy=True)
        output[inputs <= 0] = 0

        return output

    def summary(self):
        return f"{self.name}", self.__class__.__name__, 0, 0
# endregion


string_to_class =\
    {
        "linear": Linear,
        "sigmoid": Sigmoid,
        "relu": ReLU
    }
