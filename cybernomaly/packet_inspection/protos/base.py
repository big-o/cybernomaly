class Proto:
    def __init__(self, inspector):
        self.inspector = inspector

    def run(self, *args, **kwargs):
        while True:
            payload = yield
            print(self.__class__.__name__)
            if payload is None:
                break
            next_state, meta = self.process(payload)
            print(next_state, meta)
            if next_state is None:
                break
            inspector.set_state(next_state)
            inspector.add_meta(payload.name, meta)

    def process(self, payload):
        raise NotImplementedError("Abstract class")
