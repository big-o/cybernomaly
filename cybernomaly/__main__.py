import csv
import click
from datetime import datetime

from cybernomaly.packet_inspection import DeepPacketInspector, PcapPlayer
from cybernomaly.anomaly_detection import MIDAS_R

_DEFAULT_FMT = "%.time% %-6s,IP.proto% %-15s,IP.src% -> %-15s,IP.dst%"


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--num", "-n", type=int, help="Number of packets to read.",
)
@click.option(
    "--offset", "-o", type=int, help="Number of packets at the beginning to skip.",
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
def main(filename, num, offset, speed, fmt):
    """Analyse a PCAP file for anomalous packets."""
    dpi = DeepPacketInspector()

    if filename.endswith(".csv"):
        with open(filename) as fh:
            reader = csv.reader(fh)
            midasr = MIDAS_R(ticksize=60)
            maxscore = 0
            for seen, row in enumerate(reader, 1):
                if seen > num:
                    break
                src, dst, ts, desc = row
                label = 0 if desc == "-" else 1
                ts = datetime.strptime(ts, "%m/%d/%Y-%H:%M").timestamp()
                score = midasr.update_predict_score(src, dst, t=ts)
                # print(f"{seen}: [[{score}]] {report.summary()}")
                # if label != 0 or seen % 50000 == 0:
                if score > midasr.thresh or label != 0:
                    print(
                        f"{seen}: {ts}: {desc} {score:.3g} (t = {midasr.thresh:.3g}) {src} -> {dst}"
                    )

    else:
        player = PcapPlayer(filename)
        midasr = MIDAS_R()
        for pkt in player.replay(n_packets=num, offset=offset, speed=speed):
            report = dpi.process(pkt)
            rmeta = report.meta
            ipinfo = rmeta.get("IP", {})
            srcip, dstip = (
                ipinfo.get("src"),
                ipinfo.get("dst"),
            )
            portinfo = rmeta.get("UDP", rmeta.get("TCP", {}))
            srcport, dstport = (
                portinfo.get("sport"),
                portinfo.get("dport"),
            )
            src, dst = f"{srcip}:{srcport}", f"{dstip}:{dstport}"

            score = midasr.update_predict_score(src, dst, t=player.t)
            # print(f"{player.seen}: [[{score}]] {report.summary()}")
            print(f"{player.seen}: {player.t}: [[{score}]] {src} -> {dst}")


if __name__ == "__main__":
    main()
