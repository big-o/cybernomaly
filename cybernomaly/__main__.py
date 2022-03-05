import csv
import click
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import Progress

from cybernomaly.packet_inspection import DeepPacketInspector, PcapPlayer
from cybernomaly.anomaly_detection import MIDAS_R

_DEFAULT_FMT = "%.time% %-6s,IP.proto% %-15s,IP.src% -> %-15s,IP.dst%"


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def rowcount(filename):
    f = open(filename, "rb")
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b"\n") for buf in f_gen)


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--num",
    "-n",
    type=int,
    help="Number of packets to read.",
)
@click.option(
    "--offset",
    "-o",
    type=int,
    help="Number of packets at the beginning to skip.",
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
@click.option(
    "--out",
    "-O",
    type=str,
    default=None,
    help="File to save a plot and table of scores to.",
)
def main(filename, num, offset, speed, fmt, out):
    """Analyse a PCAP file for anomalous packets."""
    dpi = DeepPacketInspector()

    if filename.endswith(".csv"):
        total = rowcount(filename) - 1
        with open(filename) as fh:
            reader = csv.reader(fh)
            next(reader)
            midasr = MIDAS_R(
                error_rate=2 / 768,
                false_pos_prob=0.6,
                decay=0.6,
                ticksize=1,
                mode="log",
            )
            results = []
            xlim = [None, None]
            with Progress(expand=True) as progress:
                task = progress.add_task("Processing...", total=total)
                for seen, row in enumerate(reader, 1):
                    try:
                        if num is not None and seen > num:
                            break

                        progress.update(
                            task, description=f"Processing #{seen}/{total}..."
                        )
                        ts, src, dst = row
                        ts = int(ts)
                        if xlim[0] is None or ts < xlim[0]:
                            xlim[0] = ts
                        if xlim[1] is None or ts > xlim[1]:
                            xlim[1] = ts
                        score = midasr.update_predict_score(src, dst, t=ts)
                        results.append([ts, src, dst, score])
                        progress.update(task, advance=1)
                    except KeyboardInterrupt:
                        break

        results = pd.DataFrame(results, columns=["t", "src", "dst", "score"])
        print(results.sort_values("score", ascending=False).head(20))
        print(results.groupby("t")["score"].max().sort_values(ascending=False).head(20))
        results.to_csv(f"{out}.csv")
        if out:
            data = results.groupby("t")["score"].max().reset_index()
            fig, ax = plt.subplots(figsize=(8, 6))
            data.plot(x="t", y="score", ax=ax)
            ax.hlines(
                midasr.thresh_,
                *xlim,
                linestyle="dashed",
                label=r"$\alpha = " f"{midasr.alpha}$",
            )
            ax.set_xlim(xlim)
            ax.legend()
            fig.savefig(f"{out}.png")

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
