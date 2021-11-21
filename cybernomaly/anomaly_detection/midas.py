from probables import CountMinSketch
from time import time
from cybernomaly.exceptions import NotInitializedError
from scipy.stats import chi2


class MIDAS_R:
    "https://arxiv.org/abs/1911.04464"

    def __init__(self, error_rate=0.1, false_pos_prob=0.05, decay=0.5, agg=max):
        self.error_rate = error_rate
        self.false_pos_prob = false_pos_prob
        if not (0 <= self.decay <= 1):
            raise ValueError(f"Decay factor must be in the range (0, 1)")
        self.decay = decay
        self.agg = agg

    def initialize(self, t=None):
        self._edge_tot = self._create_cms()
        self._node_tot = self._create_cms()
        self._edge_cur = self._create_cms()
        self._node_cur = self._create_cms()
        self._last_update = t if t is not None else int(time())

    def update_predict_score(self, src, dst, count=1, t=None):
        self._check_is_initialized()

        now = int(t) if t is not None else int(time())
        if now > self._last_update:
            if self.decay:
                for i, _ in enumerate(self._bins):
                    self._edge_cur._bins[i] *= self.decay
                    self._node_cur._bins[i] *= self.decay
            else:
                self._edge_cur.clear()
                self._node_cur.clear()
            self._last_update = now

        self._update_edge(src, dst, count)
        self._update_node(src, count)
        self._update_node(dst, count)

        edge_score = self._score(self._edge_cur, self._edge_tot)
        src_score = self._score(self._node_cur.check(src), self._node_tot.check(src))
        dst_score = self._score(self._node_cur.check(dst), self._node_tot.check(dst))

        score = self.agg(edge_score, src_score, dst_score)

        return score

    def update_predict(self, src, dst, count=1, t=None):
        score = self.update_predict_score(src, dst, count, t)
        return score > self._thresh

    def _check_is_initialized(self):
        for at in ("_edge_tot", "_node_tot", "_edge_cur", "_node_cur", "_last_update"):
            if not hasattr(self, at):
                raise NotInitializedError(
                    f"{self} must be initialized before receiving any data!"
                )

    def _create_cms(self):
        cms = CountMinSketch(
            confidence=self.false_pos_prob / 2, error_rate=self.error_rate
        )
        return cms

    def _update(self, item, count, cur, tot):
        tot.add(item, count)

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
