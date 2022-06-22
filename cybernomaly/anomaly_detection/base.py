from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

__all__ = ["Monitor"]


class Monitor(ABC, BaseEstimator):
    @abstractmethod
    def partial_fit(self, *args, **kwargs):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def detect(self, *args, **kwargs):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def update_detect(self, *args, **kwargs):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def update_detect_score(self, *args, **kwargs):
        raise NotImplementedError("abstract method")

    @staticmethod
    def _snap_time(t, interval):
        return round(t / interval) * interval
