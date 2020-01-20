from enum import Enum, auto
import numpy as np
import re
from typing import Dict

from minikeras.metrics import Metric

reduce_lr_on_plateau_mode_regexi =\
    {
        "auto": re.compile(r"(?i)(a(uto)?)"),
        "min": re.compile(r"(?i)(min)"),
        "max": re.compile(r"(?i)(max)")
    }


class Callback:
    def training_start(self, model_status: Dict = None, **kwargs):
        pass

    def epoch_start(self, model_status: Dict = None, **kwargs):
        pass

    def iteration_start(self, model_status: Dict = None, **kwargs):
        pass

    def training_end(self, model_status: Dict = None, **kwargs):
        pass

    def epoch_end(self, model_status: Dict = None, **kwargs):
        pass

    def iteration_end(self, model_status: Dict = None, **kwargs):
        pass


class ReduceLROnPlateau(Callback):
    class Mode(Enum):
        Auto = auto()
        Minimum = auto()
        Maximum = auto()

        @staticmethod
        def parse(reduce_lr_on_plateau_mode_name: str):
            for name in reduce_lr_on_plateau_mode_regexi:
                if reduce_lr_on_plateau_mode_regexi[name].match(reduce_lr_on_plateau_mode_name):
                    return reduce_lr_on_plateau_name_to_mode[name]

    def __init__(self, monitor: str or Metric = "val_loss", factor: float = 0.1, patience: int = 5,
                 verbose: int = 1, mode: str or Mode = "auto",
                 min_delta: float = 1e-4, cooldown: int = 0, min_lr: float = 0., **kwargs):
        # region Argument validation
        if verbose is None or not isinstance(verbose, int):
            verbose = 1

        if monitor is None:
            if verbose > 1:
                print(f"Argument monitor was set to None, defaulting to 'loss'...")

            monitor = "loss"

        if isinstance(monitor, Metric):
            monitor = Metric.get_key(monitor)
        elif isinstance(monitor, str):
            monitor = Metric.enum_to_name(monitor)
        else:
            raise TypeError(f"Argument monitor should be a string or of type {type(Metric)} "
                            f"(it is currently {type(monitor)})!")

        if factor is None:
            raise ValueError("Argument factor cannot be None!")

        if not (isinstance(factor, float) or isinstance(factor, int)):
            raise TypeError(f"Argument factor should be a floating point value or an integer "
                            f"(it is currently {type(factor)})!")

        if patience is None:
            raise ValueError("Argument patience cannot be None!")

        if isinstance(patience, float):
            patience = int(patience)
            
        if not isinstance(patience, int):
            raise TypeError(f"Argument patience should be an integer value (it is currently {type(patience)})!")

        if isinstance(mode, str):
            mode = ReduceLROnPlateau.Mode.parse(mode)

        if mode is None:
            mode = ReduceLROnPlateau.Mode.Auto

        if not isinstance(mode, self.Mode):
            raise TypeError(f"Argument mode should be of type {type(self.Mode)} (it is currently {type(mode)})!")

        if min_delta is None:
            raise ValueError("Argument min_delta cannot be None!")

        if isinstance(min_delta, int):
            min_delta = float(min_delta)

        if not isinstance(min_delta, float):
            raise TypeError(f"Argument min_delta should be a floating point value (it is currently {type(min_delta)})!")

        if cooldown is None:
            cooldown = 0

        if not isinstance(cooldown, int):
            raise TypeError(f"Argument cooldown should be a integer value (it is currently {type(cooldown)})!")

        if min_lr is None:
            min_lr = 0.

        if isinstance(min_lr, int):
            min_lr = float(min_lr)

        if not isinstance(min_lr, float):
            raise TypeError(f"Argument min_lr should be a floating point value (it is currently {type(min_lr)})!")
        # endregion

        self.__monitor = monitor
        self.__factor = factor
        self.__patience = patience
        self.__verbose = verbose
        self.__original_mode = mode
        self.__mode = self.__original_mode
        self.__min_delta = min_delta
        self.__cooldown = cooldown
        self.__min_lr = min_lr

        self.__monitor_values = list()
        self.__patience_counter = 0
        self.__cooldown_counter = 0

        self.__best_monitor = None

    def training_start(self, model_status: Dict = None, **kwargs):
        if self.__monitor not in model_status:
            raise KeyError(f"The value you want to monitor ({self.__monitor}) wasn't found in the model status "
                           f"(which contains {sorted(model_status.keys())})!")

        self.__mode = self.__original_mode
        self.__patience_counter = 0
        self.__cooldown_counter = self.__cooldown

    def epoch_start(self, model_status: Dict = None, **kwargs):
        self.__monitor_values.clear()

    def training_end(self, model_status: Dict = None, **kwargs):
        self.__monitor_values.clear()

    def epoch_end(self, model_status: Dict = None, **kwargs):
        self.__monitor_values.extend(model_status[self.__monitor][-1])

        if self.__best_monitor is None:
            self.__best_monitor = self.__monitor_values[-1]

        if self.__mode == self.Mode.Auto:
            i = len(self.__monitor_values)

            if i == 0:
                raise RuntimeError("The first monitored value is None, despite the impossibility of this happening. "
                                   "Contact the author and/or create a pull request.")

            first_monitored_value = np.mean(self.__monitor_values[0])

            while i != 0 and self.__mode == self.Mode.Auto:
                mean_monitor_value = np.mean(self.__monitor_values[:i])

                if mean_monitor_value > first_monitored_value:
                    self.__mode = self.Mode.Maximum

                    if self.__verbose > 1:
                        print("Mode set from Auto to Maximum.")

                elif mean_monitor_value < first_monitored_value:
                    self.__mode = self.Mode.Minimum

                    if self.__verbose > 1:
                        print("Mode set from Auto to Minimum.")
                else:
                    i //= 2

            if i == 0:
                if self.__verbose > 0:
                    print("While trying to determine the real mode of ReduceLROnPlateau there was no change in "
                          "monitored values, so the mode will remain Auto and operations will be skipped for this "
                          "epoch.")
                return

        last_value = self.__monitor_values[-1]
        difference = self.__best_monitor - last_value

        if self.__mode == self.Mode.Maximum:
            difference = -difference

            self.__best_monitor = max(self.__best_monitor, last_value)
        else:
            self.__best_monitor = min(self.__best_monitor, last_value)

        self.__cooldown_counter += 1

        if difference < self.__min_delta:
            if model_status["learning_rate"] > self.__min_lr:
                if self.__cooldown_counter > self.__cooldown:
                    self.__patience_counter += 1

                    if self.__patience_counter > self.__patience:
                        model_status["learning_rate"] = max(self.__min_lr,
                                                            model_status["learning_rate"] * self.__factor)

                        if self.__verbose > 0:
                            print(f"Learning rate adjusted to {model_status['learning_rate']}\n")

                        self.__cooldown_counter = 0
                        self.__patience_counter = 0
        else:
            self.__patience_counter = 0


class ModelCheckpoint(Callback):
    pass


reduce_lr_on_plateau_name_to_mode = \
    {
        "auto": ReduceLROnPlateau.Mode.Auto,
        "min": ReduceLROnPlateau.Mode.Minimum,
        "max": ReduceLROnPlateau.Mode.Maximum
    }