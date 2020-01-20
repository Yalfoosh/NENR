import numpy as np
from sys import stdout
from tqdm import tqdm

from losses import Loss, SquareError
from rule import Rule
from util import split_data_into_batches, generate_anfis_dataset


class ANFIS:
    def __init__(self, input_size: int, rule_count: int):
        self.__rules = tuple(Rule(input_size) for _ in range(rule_count))

        self.__gradient_cache = {i: None for i in range(rule_count)}
        self.__loss = None

    @property
    def rules(self):
        return self.__rules

    def infer(self, inputs):
        flattened_activations = np.array([r.flattened_activation(inputs) for r in self.rules])
        fa_sum = np.sum(flattened_activations, axis=0)

        if np.abs(fa_sum) < 1e-16:
            fa_sum = 1e-16 if fa_sum > 0. else -1e-16

        normalized_activations = flattened_activations / fa_sum

        applications = np.array([r.application(inputs) for r in self.rules])

        return np.sum(normalized_activations @ applications)

    def compile(self, loss: Loss):
        if loss is None:
            raise ValueError("Argument loss cannot be None!")

        if not isinstance(loss, Loss):
            raise TypeError(f"Argument loss should be of type {type(Loss)} (it is currently {type(loss)}!")

        self.__loss = loss

    def fit(self, x: np.ndarray, y: np.ndarray,
            batch_size: int = 1,
            epochs: int = 1,
            verbose: int = 1,
            learning_rate_dense: float = 1e-4,
            learning_rate_sigmoid: float = 1e-3,
            shuffle: bool = True,
            **kwargs):
        if self.__loss is None:
            raise RuntimeError("Make sure you compile the model before fitting!")

        stop_after_no_improvement = kwargs.get("stop_after_no_improvement", True)

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if shuffle:
            p = np.random.permutation(len(x))
            x, y = x[p], y[p]

        learning_rates = np.array([learning_rate_dense] * 2 + [learning_rate_sigmoid] * 2)

        history = dict()
        history["loss"] = list()

        if verbose > 1:
            print(f"Splitting training data into batches of {batch_size}...")

        x, y = [split_data_into_batches(dataset, batch_size) for dataset in [x, y]]
        last_loss = float("inf")

        for epoch_index in range(epochs):
            if verbose > 1:
                print(f"Epoch {epoch_index + 1}/{epochs}")

            if verbose < 1:
                batch_indices = range(len(x))
            else:
                batch_indices = tqdm(range(len(x)), file=stdout)

            history["loss"].append(list())

            for batch_index in batch_indices:
                self.__gradient_cache = {i: None for i in self.__gradient_cache}

                current_loss = list()

                for entry_index in range(len(x[batch_index])):
                    inputs = x[batch_index][entry_index]
                    result = self.infer(inputs)

                    current_loss.append(self.__loss.call(y[batch_index][entry_index], result))
                    loss_grad = self.__loss.gradient(y[batch_index][entry_index], result)

                    # region Pre-calculation
                    f_results = np.array([r.application(inputs) for r in self.rules])
                    f_result_diff = np.array([f_results[i] - f_results for i in range(len(f_results))])

                    activations = np.array([r.activation(inputs) for r in self.rules])
                    flattened_activations = np.array([np.product(activations, axis=1)])[0]
                    flattened_activations_sum = np.sum(flattened_activations, axis=0)

                    if np.abs(flattened_activations_sum) < 1e-16:
                        flattened_activations_sum = 1e-16 if flattened_activations_sum < 0. else -1e-16

                    activation_vote = flattened_activations / flattened_activations_sum

                    activation_sum_sq = np.square(flattened_activations_sum)

                    squared_activation = np.square(activations)
                    inverse_activation = 1. - activations

                    mixed_sigmoid_grad = squared_activation * inverse_activation

                    for i, fa in enumerate(flattened_activations):
                        if np.abs(fa) < 1e-16:
                            fa = 1e-16 if fa > 0. else -1e-16

                        mixed_sigmoid_grad[i] /= fa
                    # endregion

                    # region Linear function gradient calculation

                    output_grad_wrt_weights = np.outer(activation_vote, inputs)
                    output_grad_wrt_bias = np.copy(activation_vote)

                    weight_grads = np.multiply(loss_grad, output_grad_wrt_weights)
                    bias_grads = np.multiply(loss_grad, output_grad_wrt_bias)
                    # endregion

                    # region Activation gradient calculation
                    output_grad_wrt_activations = np.divide(np.sum(flattened_activations * f_result_diff, axis=1),
                                                            activation_sum_sq)
                    loss_grad_wrt_activations = np.multiply(loss_grad, output_grad_wrt_activations)

                    activation_grad_wrt_a = np.array([r.b for r in self.rules]) * mixed_sigmoid_grad
                    activation_grad_wrt_b = (np.array([r.a for r in self.rules]) - inputs) * mixed_sigmoid_grad

                    a_grads = np.copy(activation_grad_wrt_a)
                    b_grads = np.copy(activation_grad_wrt_b)

                    for i, lgwrta in enumerate(loss_grad_wrt_activations):
                        a_grads[i] *= lgwrta
                        b_grads[i] *= lgwrta
                    # endregion

                    # region Gradient accumulation
                    for i in range(len(self.rules)):
                        if self.__gradient_cache[i] is None:
                            self.__gradient_cache[i] = np.array([weight_grads[i]]), np.array([bias_grads[i]]),\
                                                       np.array([a_grads[i]]), np.array([b_grads[i]])
                        else:
                            self.__gradient_cache[i] = np.stack([self.__gradient_cache[i][0][0], weight_grads[i]]), \
                                                       np.stack([self.__gradient_cache[i][1][0], bias_grads[i]]), \
                                                       np.stack([self.__gradient_cache[i][2][0], a_grads[i]]), \
                                                       np.stack([self.__gradient_cache[i][3][0], b_grads[i]])
                    # endregion

                [r.update(self.__gradient_cache[i], learning_rates) for i, r in enumerate(self.rules)]
                history["loss"][-1].append(np.mean(current_loss))

                if verbose > 0:
                    batch_indices.set_description(f"Loss:\t{np.mean(history['loss'][-1]):.06f}\t")
                    batch_indices.update()

            if last_loss < np.mean(history["loss"][-1]):
                learning_rates *= np.array([0.5, 0.5, 0.5, 0.5])

                if verbose > 1:
                    print(f"Learning rates set to {learning_rates}")

            if stop_after_no_improvement and np.any(learning_rates < 1e-15):
                if verbose > 0:
                    print(f"Fitting automatically stopped because one or more learning rates fell below the minimum "
                          f"value.")
                break

            last_loss = np.mean(history["loss"][-1])

        return history
