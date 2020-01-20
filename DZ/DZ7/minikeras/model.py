from copy import deepcopy
import numpy as np
from sys import stdout
from tqdm import tqdm
from typing import Iterable, List, Tuple

from minikeras.callbacks import Callback
from minikeras.layers import Layer, Input
from minikeras.losses import Loss
from minikeras.util import pad_string


class Model:
    summary_columns = ["Layer Name", "Layer Type", "No. Params", "No. Trainable Params"]

    def __init__(self, layers: List[Layer]):
        for layer in layers:
            if layer is None or not isinstance(layer, Layer):
                raise TypeError(f"Given layer {layer} is not of type {type(Layer)} (instead it is {type(layer)})!")

        self.__layers = deepcopy(layers)

        self.__loss = None
        self.__compiled = False

        self.__input_cache = list()
        self.__gradient_cache = dict()

        self.__parameter_layer_indices = None
        self.__parameter_lengths = None

    @property
    def layers(self):
        return self.__layers

    @property
    def compiled(self):
        return self.__compiled

    @property
    def loss(self):
        return self.__loss

    @staticmethod
    def split_data_into_batches(dataset, batch_size):
        no_batches = len(dataset) // batch_size
        part_size = len(dataset) % batch_size
        full_size = batch_size * no_batches

        dataset_full = dataset[: full_size].reshape((no_batches, batch_size, *dataset[0].shape))
        dataset_part = np.array([])

        if part_size != 0:
            dataset_part = dataset[full_size:].reshape((1, part_size, *dataset[0].shape))

        dataset = list(dataset_full)

        if part_size != 0:
            dataset.extend(list(dataset_part))

        return dataset

    def add_layer(self, layer: Layer):
        if self.__compiled:
            raise RuntimeError("Can't add a layer to a model that is already compiled!")

        if layer is None:
            raise ValueError("Argument layer cannot be None!")

        if not isinstance(layer, Layer):
            raise TypeError(f"Argument layer must be of type {type(Layer)} (currently it is {type(layer)}")

        self.__layers.append(layer)

    def compile(self, loss: Loss, **kwargs):
        if loss is None:
            raise ValueError("Argument loss cannot be None!")

        if not isinstance(loss, Loss):
            raise TypeError(f"Argument loss should be of type {type(Loss)} (currently it is {type(loss)})!")

        self.__loss = loss

        unpacked_layers = list()
        current_input_units = None
        current_index = 0

        if isinstance(self.layers[0], Input):
            current_input_units = self.layers[0].build(None)
            self.layers[0].id = current_index
            current_index += 1

            unpacked_layers.append(self.layers[0])
            self.__layers = self.__layers[1:]
        else:
            current_input_units = kwargs.get("input_units", None)

        if current_input_units is None:
            raise RuntimeError("Didn't manage to see how much the first input units argument is: try passing it as "
                               "a kwargs with the name of 'input_units'.")

        for layer in self.layers:
            current_input_units = layer.build(current_input_units)
            layer.id = current_index
            current_index += 1

            unpacked_layers.append(layer)

            if layer.activation is not None:
                current_input_units = layer.activation.build(current_input_units)
                layer.activation.id = current_index
                current_index += 1

                unpacked_layers.append(layer.activation)

        self.__layers = unpacked_layers

        for layer in self.__layers:
            if layer.trainable:
                self.__gradient_cache[layer.id] = list()

        self.__compiled = True

    def fit(self, x: np.ndarray, y: np.ndarray,
            validation_split: float = 0.0,
            validation_data=None,
            batch_size: int = 1,
            epochs: int = 1,
            verbose: int = 1,
            callbacks: Callback or Iterable[Callback] or None = None,
            shuffle: bool = True,
            learning_rate: float = 0.1,
            **kwargs):
        if not self.__compiled:
            raise RuntimeError(f"Compile model {self} before fitting!")

        # region Parameter validation
        if validation_data is None:
            if validation_split is None or validation_split > 1.0:
                validation_split = 0.0
        else:
            validation_split = 0.0

        if batch_size is None or batch_size < 1:
            batch_size = len(x)

        if epochs < 1:
            raise ValueError(f"Epochs must be a positive integer (it is currently {epochs})!")

        if not (isinstance(callbacks, Callback) or isinstance(callbacks, list) or isinstance(callbacks, tuple)):
            callbacks = None

        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        # endregion

        history =\
            {
                "learning_rate": learning_rate,
                "loss": list()
            }

        if validation_data is not None:
            history["val_loss"] = list()

        if shuffle:
            p = np.random.permutation(len(x))
            x, y = x[p], y[p]

        if validation_data is None:
            if validation_split > 0.0:
                number_of_validation_entries = len(x) * validation_split

                validation_data = x[:number_of_validation_entries], y[:number_of_validation_entries]

                x = x[number_of_validation_entries:]
                y = y[number_of_validation_entries:]

        # region Data metamorphosis
        if verbose > 0:
            print("Preparing batches...")

        x, y = [self.split_data_into_batches(dataset, batch_size) for dataset in [x, y]]

        if validation_data is not None:
            validation_data = tuple([self.split_data_into_batches(dataset, batch_size) for dataset in validation_data])

        assert len(x) == len(y)

        if validation_data is not None:
            assert len(validation_data[0]) == len(validation_data[1])
        # endregion

        for callback in callbacks:
            callback.training_start(history, **kwargs)

        for epoch_index in range(epochs):
            for callback in callbacks:
                callback.epoch_start(history, **kwargs)

            if verbose > 0:
                print(f"\nEpoch {epoch_index + 1} / {epochs}")

            history["loss"].append(list())

            if validation_data is not None:
                history["val_loss"].append(list())

            if verbose == 0:
                batch_indices = range(len(x))
            else:
                batch_indices = tqdm(range(len(x)), file=stdout)

            for batch_index in batch_indices:
                for callback in callbacks:
                    callback.iteration_start(history, **kwargs)

                for layer_index in self.__gradient_cache:
                    if self.__layers[layer_index].trainable:
                        self.__gradient_cache[layer_index] = None

                for entry_index in range(len(x[batch_index])):
                    self.__input_cache.clear()
                    result = self.predict(x[batch_index][entry_index])

                    loss = self.loss.call(y[batch_index][entry_index], result)
                    loss_gradient = self.loss.gradient(y[batch_index][entry_index], result)

                    if verbose > 0:
                        batch_indices.set_description(f"Loss:\t{np.mean(loss):.06f}\t"
                                                      f"Learning rate: {history['learning_rate']}")
                    history["loss"][-1].append(np.mean(loss))

                    current_gradient = loss_gradient

                    for layer_index in range(len(self.__layers))[::-1]:
                        current_gradient = self.__layers[layer_index]\
                                               .backward(self.__input_cache[layer_index],
                                                         current_gradient,
                                                         self.__gradient_cache)

                for layer_index in self.__gradient_cache:
                    self.__layers[layer_index].update(self.__gradient_cache, history["learning_rate"])

                for callback in callbacks:
                    callback.iteration_end(history, **kwargs)

            if validation_data is not None:
                current_validation_loss = list()

                if verbose > 1:
                    print("Calculating validation loss...")

                for batch_index in range(len(validation_data[0])):
                    current_batch_validation_loss = list()

                    for entry_index in range(len(validation_data[0][batch_index])):
                        result = self.predict(validation_data[0][batch_index][entry_index])
                        loss = self.loss.call(validation_data[1][batch_index][entry_index], result)

                        current_batch_validation_loss.append(loss)

                    current_validation_loss.append(np.mean(current_batch_validation_loss))

                current_validation_loss = np.mean(current_validation_loss)
                history["val_loss"][-1].append(current_validation_loss)

                if verbose > 0:
                    print(f"Validation loss:\t{current_validation_loss:.06f}")

            for callback in callbacks:
                callback.epoch_end(history, **kwargs)

        for callback in callbacks:
            callback.training_end(history, **kwargs)

        return history

    def predict(self, inputs: np.ndarray):
        self.__input_cache.clear()

        for layer in self.layers:
            self.__input_cache.append(inputs)
            inputs = layer.forward(inputs)

        self.__input_cache.append(inputs)

        return inputs

    def summary(self, **kwargs):
        if len(self.layers) == 0:
            return "Empty Model\n"

        title_string = kwargs.get("title", "Model")
        extra_pad_width = kwargs.get("extra_pad_width", 2)

        raw_layer_returns = [layer.summary() for layer in self.layers]
        param_count = sum(raw_layer_return[2] for raw_layer_return in raw_layer_returns)
        tr_param_count = sum(raw_layer_return[3] for raw_layer_return in raw_layer_returns)

        layer_returns = [[str(column) for column in raw_layer_return] for raw_layer_return in raw_layer_returns]

        categories = list()

        for i in range(len(self.summary_columns)):
            categories.append(list())
            categories[i] = [self.summary_columns[i]]

            for j in range(len(layer_returns)):
                categories[i].append(layer_returns[j][i])

        category_maxima = [max([len(column) for column in category]) for category in categories]
        column_width = max(category_maxima) + extra_pad_width * 2

        to_return = pad_string(pad_string(title_string, 4 + len(title_string), pad_with=" "),
                               4 * column_width, pad_with="=") + "\n\n"

        to_return += "".join([pad_string(column_text, column_width, pad_with=" ")
                              for column_text in self.summary_columns])
        to_return += "\n" + "-" * (4 * column_width)

        for layer_return in layer_returns:
            to_return += "\n"
            to_return += "".join([pad_string(layer_column, column_width, pad_with=" ")
                                  for layer_column in layer_return])

        to_return += f"\n\nTotal number of parameters:\t\t\t\t{param_count}"
        to_return += f"\nTotal number of trainable parameters:\t{tr_param_count}"

        return to_return

    def extract_parameters(self):
        parameters = [layer.extract_parameters() for layer in self.layers]

        if self.__parameter_lengths is None or self.__parameter_layer_indices is None:
            self.__parameter_layer_indices = list()
            self.__parameter_lengths = list()

            for i, p in enumerate(parameters):
                if p is not None:
                    self.__parameter_layer_indices.append(i)
                    self.__parameter_lengths.append(len(p))

        parameters = np.concatenate([p for p in parameters if p is not None])

        return parameters

    def inject_parameters(self, parameters):
        _last_index = 0
        
        for i, length in zip(self.__parameter_layer_indices, self.__parameter_lengths):
            self.layers[i].inject_parameters(np.array(parameters[_last_index: _last_index + length]))

            _last_index += length

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary()
