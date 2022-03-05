class Protocol:
    def __init__(self, inspector):
        self.inspector = inspector

    def process(self, payload):
        if payload is None:
            return
        meta = self.get_meta(payload)
        next_proto = payload.getlayer(1)
        next_state = next_proto.name if next_proto is not None else None
        self.inspector.set_state(next_state)
        if meta is not None:
            self.inspector.add_meta(payload.name, meta)

    def get_meta(self, payload):
        raise NotImplementedError("Abstract class")

    @property
    def name(self):
        return self.__class__.__name__


class _SKIP_STATE(Protocol):
    def get_meta(self, payload):
        proto = payload.getlayer(0).name
        if proto in ("Raw", "Padding"):
            return None
        return {}
