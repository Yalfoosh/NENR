from typing import Callable


class Variable:
    def __init__(self, reference_value):
        self._reference_value = reference_value

    @property
    def reference_value(self):
        return self._reference_value

    def get(self, **kwargs):
        raise NotImplemented


class Constant(Variable):
    def __init__(self, reference_value):
        super().__init__(reference_value=reference_value)

    def get(self, **kwargs):
        return self.reference_value


class FunctionVariable(Variable):
    def __init__(self, reference_value, function: Callable):
        super().__init__(reference_value=reference_value)

        self.__function = function

    @property
    def function(self):
        return self.__function

    def get(self, **kwargs):
        return self.function(self.reference_value, **kwargs)


class TrackingFunctionVariable(FunctionVariable):
    def __init__(self, reference_value, function: Callable, key_to_track: str):
        super().__init__(reference_value=reference_value, function=function)

        self.__key_to_track = key_to_track

    @property
    def key_to_track(self):
        return self.__key_to_track

    def get(self, **kwargs):
        self._reference_value = kwargs.get(self.key_to_track, self._reference_value)

        return super().get(**kwargs)
