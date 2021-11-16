from time import sleep

from kamene.all import PcapReader

from cybernomaly.packet_inspection import protos


class PcapPlayer:
    def __init__(self, filename):
        self.filename = filename
        self.seen = 0

    def replay(self, n_packets=None, speed=1, callback=None, **kwargs):
        t = None
        n_packets = n_packets if n_packets is not None else -1

        with PcapReader(self.filename) as pcap:
            for n, pkt in enumerate(pcap):
                if n == n_packets:
                    break

                if speed is not None and t is not None:
                    sleep((pkt.time - t) / speed)

                self.seen += 1
                t = pkt.time

                if callback is not None:
                    callback(pkt, **kwargs)

                yield pkt
