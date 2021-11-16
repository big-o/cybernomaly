from kamene.layers.l2 import IP_PROTOS

from cybernomaly.packet_inspection.protos.base import Proto


class IP(Proto):
    PROTOS_IP = {v: k.upper() for k, v in vars(IP_PROTOS).items()}

    def process(self, payload):
        print("IP processing")
        next_state = IP.PROTOS_IP.get(payload.proto)
        meta = {
            "src": payload.src,
            "dst": payload.dst,
        }
        return next_state, meta
