import importlib
import inspect
import pkgutil
from time import sleep

from cybernomaly.packet_inspection import protos
from cybernomaly.packet_inspection.protos.base import Protocol


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


class PacketReport:
    def __init__(self, meta):
        self._meta = meta

    def summary(self, delim=" | "):
        out = []
        for layer, meta in self._meta.items():
            layer = layer.replace(" ", "")
            detail = []
            for key, val in sorted(meta.items(), key=lambda x: x[0]):
                detail.append(f"{key}={val}")
            if detail:
                out.append(f"{layer}[{','.join(detail)}]")
            else:
                out.append(layer)
        return delim.join(out)

    @property
    def meta(self):
        return self._meta


class DeepPacketInspector:
    _STATES = _find_subclasses(protos, Protocol)

    def __init__(self, start=None, default="skip"):
        self.states = {}
        for name, proto in self._STATES.items():
            self.states[name] = proto(self)
        self._start = start
        if default == "stop":
            self._default = None
        elif default == "skip":
            self._default = self.states["_SKIP_STATE"]
        else:
            raise ValueError(f"Unsupported default action '{default}'")

        self._metadata = None
        self._reset()

    def _reset(self):
        self._current = self._start
        self._stopped = False
        self._metadata = {}

    def set_state(self, state):
        self._current = state

    def process(self, packet):
        payload = packet
        if self._current is None and payload is not None:
            self._current = payload.getlayer(0).name

        while self._current is not None and payload is not None:
            payload = payload.getlayer(self._current)
            next_state = self.states.get(self._current, self._default)
            if next_state is not None:
                next_state.process(payload)

        meta = self._metadata
        self._reset()
        return PacketReport(meta)

    def add_meta(self, key, meta):
        self._metadata[key] = meta
