from typing import Callable


from civilization.signal import Antenna, CallbackID, Listener


class Callback:
    def register_listeners(self, antenna: Antenna):
        raise NotImplementedError


class BestSpecimenCheckpoint(Callback):
    def __init__(self):
        self.__records = list()

    def record_best(self, **kwargs):
        self.__records.append(kwargs["population"][0])

    def register_listeners(self, antenna: Antenna):
        after_iteration_listener = Listener(function=self.record_best)

        antenna.register_listener(identifier=CallbackID.IterationEnd, listener=after_iteration_listener)


class EarlyStopping(Callback):
    def __init__(self, condition: Callable, exception_to_raise: Exception = RuntimeError, verbose: int = 0):
        self.__condition = condition
        self.__exception_to_raise = exception_to_raise
        self.__verbose = verbose

    @property
    def condition(self):
        return self.__condition

    @property
    def exception_to_raise(self):
        return self.__exception_to_raise

    @property
    def verbose(self):
        return self.__verbose

    def check_condition(self, **kwargs):
        if self.condition(**kwargs):
            if self.verbose > 0:
                print(f"[Early Stopping]\tCondition met, stopping...")

            raise self.exception_to_raise

    def register_listeners(self, antenna: Antenna):
        after_iteration_listener = Listener(function=self.check_condition)

        antenna.register_listener(identifier=CallbackID.IterationEnd, listener=after_iteration_listener)
