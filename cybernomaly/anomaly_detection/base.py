from abc import ABC, abstractmethod

__all__ = ["Monitor"]

class Monitor(ABC):
    @abstractmethod
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def update_predict(self, *args, **kwargs):
        raise NotImplementedError("abstract method")
