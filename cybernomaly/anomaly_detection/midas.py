from probables import CountMinSketch
from time import time
from cybernomaly.exceptions import NotInitializedError
from scipy.stats import chi2


class MIDAS_R:
    "https://arxiv.org/abs/1911.04464"

    def __init__(
        self, error_rate=0.1, false_pos_prob=0.05, decay=0.5, agg=max, ticksize=1
    ):
        self.error_rate = error_rate
        self.false_pos_prob = false_pos_prob
        if not (0 <= decay <= 1):
            raise ValueError(f"Decay factor must be in the range (0, 1)")
        self.decay = decay
        self.agg = agg
        self.ticksize = ticksize

        self._edge_tot = self._create_cms()
        self._node_tot = self._create_cms()
        self._edge_cur = self._create_cms()
        self._node_cur = self._create_cms()

        self.thresh = chi2(df=1).ppf(1 - (self.false_pos_prob / 2))
        self._start = None
        self._last_update = None

    def update_predict_score(self, src, dst, count=1, t=None):
        now = (int(t) if t is not None else int(time())) / self.ticksize
        if self._start is None:
            self._start = now

        now -= self._start

        if self._last_update is None:
            self._last_update = now

        if now > self._last_update:
            if self.decay:
                for cms in (self._edge_cur, self._node_cur):
                    for i in range(len(cms._bins)):
                        cms._bins[i] *= self.decay
            else:
                self._edge_cur.clear()
                self._node_cur.clear()
            self._last_update = now

        edge = f"{src}->{dst}"
        src, dst = f"{src}", f"{dst}"
        self._update_edge(edge, count)
        self._update_node(src, count)
        self._update_node(dst, count)

        edge_score = self._score(
            self._edge_cur.check(edge), self._edge_tot.check(edge), t=now
        )
        src_score = self._score(
            self._node_cur.check(src), self._node_tot.check(src), t=now
        )
        dst_score = self._score(
            self._node_cur.check(dst), self._node_tot.check(dst), t=now
        )

        score = self.agg(edge_score, src_score, dst_score)

        return score

    def update_predict(self, src, dst, count=1, t=None):
        score = self.update_predict_score(src, dst, count, t)
        return score > self._thresh

    def _create_cms(self):
        cms = CountMinSketch(
            confidence=self.false_pos_prob / 2, error_rate=self.error_rate
        )
        return cms

    def _update(self, item, count, cur, tot):
        tot.add(item, count)
        cur.add(item, count)

    def _score(self, cur, tot, t):
        if tot == 0 or t == 0:
            score = 0
        else:
            score = ((cur - tot / (t + 1)) ** 2) / (tot * t)
        return score

    def _update_edge(self, edge, count):
        self._update(edge, count, self._edge_cur, self._edge_tot)

    def _update_node(self, node, count):
        self._update(node, count, self._node_cur, self._node_tot)
