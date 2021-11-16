import importlib
import inspect
import pkgutil
from time import sleep

from cybernomaly.packet_inspection import protos
from cybernomaly.packet_inspection.protos.base import Proto


def _find_subclasses(package, base_class):
    if isinstance(package, str):
        package = importlib.import_module(package)

    results = {
        name: obj
        for name, obj in inspect.getmembers(
            package,
            lambda m: inspect.isclass(m)
            and m is not base_class
            and issubclass(m, base_class),
        )
    }

    if hasattr(package, "__path__"):
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_name = package.__name__ + "." + name
            results.update(_find_subclasses(full_name, base_class))
    return results


class DeepPacketInspector:
    _STATES = _find_subclasses(protos, Proto)

    def __init__(self, start="IP"):
        self.states = {}
        for name, proto in self._STATES.items():
            self.states[name] = self.prime(proto(self))()
        self._start = self.states[start]

        self._current = None
        self._metadata = None
        self._reset()
        print(self._current)

    def _reset(self):
        self._current = self._start
        self._stopped = False
        self._metadata = {}

    def prime(self, proto):
        def wrapper(*args, **kwargs):
            v = proto.run(self, *args, **kwargs)
            v.send(None)
            return v

        return wrapper

    def set_state(self, state):
        self._current = self.states[state]

    def read(self, packet):
        self._payload = packet.getlayer(self._current)
        return self.process()

    def process(self):
        try:
            print("stage: ", self._current.__qualname__)
            self._current.send(self._payload)
        except StopIteration:
            self._stopped = True

        meta = self._metadata
        self._reset()
        return meta

    def add_meta(self, key, meta):
        self._metadata[key] = meta
