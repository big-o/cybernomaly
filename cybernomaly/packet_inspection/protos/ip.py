from cybernomaly.packet_inspection.protos.base import Protocol


class IP(Protocol):
    def get_meta(self, payload):
        meta = {
            "src": payload.src,
            "dst": payload.dst,
        }
        return meta
