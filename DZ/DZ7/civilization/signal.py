from enum import Enum, auto
from typing import Any, Callable, Dict, List


class CallbackID(Enum):
    TrainingStart = auto()
    TrainingEnd = auto()
    IterationStart = auto()
    IterationEnd = auto()

    BeforeSelection = auto()
    AfterSelection = auto()
    BeforeCross = auto()
    AfterCross = auto()
    BeforeMutation = auto()
    AfterMutation = auto()


class Listener:
    def __init__(self, function: Callable):
        self.__function = function

    def notify(self, **kwargs):
        self.__function(**kwargs)


class Antenna:
    def __init__(self, kwargs_in_a_list):
        self.__kwargs_in_a_list = kwargs_in_a_list
        self.__identifier_to_receiver: Dict[Any, List[Listener]] = dict()

    def register_broadcaster(self, identifier):
        if identifier not in self.__identifier_to_receiver:
            self.__identifier_to_receiver[identifier] = list()

    def register_listener(self, identifier, listener: Listener):
        self.register_broadcaster(identifier)
        self.__identifier_to_receiver[identifier].append(listener)

    def publish(self, identifier):
        if identifier in self.__identifier_to_receiver:
            for receiver in self.__identifier_to_receiver[identifier]:
                receiver.notify(**(self.__kwargs_in_a_list[0]))
