from cybernomaly.packet_inspection.protos.base import Protocol


class UDP(Protocol):
    def get_meta(self, payload):
        meta = {
            "sport": payload.sport,
            "dport": payload.dport,
        }
        return meta
