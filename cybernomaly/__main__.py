import click

from cybernomaly.packet_inspection import DeepPacketInspector, PcapPlayer

# from cybernomaly.anomaly_detection import MIDAS_R

_DEFAULT_FMT = "%.time% %-6s,IP.proto% %-15s,IP.src% -> %-15s,IP.dst%"


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--num",
    "-n",
    type=int,
    help="Number of packets to read.",
)
@click.option(
    "--speed",
    "-s",
    default=1,
    type=float,
    help="Speed at which to replay PCAP file. "
    "Set to 'inf' for instantaneous playback.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    default=_DEFAULT_FMT,
    help="Format string for summarising packets. "
    "See (kamene.packet.sprintf) for more details.",
)
def main(filename, num, speed, fmt):
    """Analyse a PCAP file for anomalous packets."""
    # midas = MIDAS_R()
    dpi = DeepPacketInspector()
    player = PcapPlayer(filename)
    for pkt in player.replay(n_packets=num, speed=speed):
        meta = dpi.read(pkt)
        # midas.update(src, dst, player.t)
        print(f"{player.seen}: {pkt.sprintf(fmt)}")
        print(f"{meta}")


if __name__ == "__main__":
    main()