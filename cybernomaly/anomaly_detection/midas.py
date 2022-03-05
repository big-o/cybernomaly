from functools import lru_cache

import numpy as np
from probables import CountMinSketch
from time import time
from cybernomaly.exceptions import NotInitializedError
from scipy.stats import chi2


class MIDAS_R:
    "https://arxiv.org/abs/1911.04464"

    def __init__(
        self,
        error_rate=0.1,
        false_pos_prob=0.02,
        decay=0.5,
        agg=max,
        ticksize=1,
        alpha=0.05,
        mode="raw",
        precision=5,
    ):
        self.error_rate = error_rate
        self.false_pos_prob = false_pos_prob

        if not (0 <= decay <= 1):
            raise ValueError(f"Decay factor must be in the range [0, 1]")
        self.decay = decay

        self.agg = agg
        self.ticksize = ticksize

        if not (0 < alpha < 1):
            raise ValueError("alpha must be in the range (0, 1)")
        self.alpha = alpha

        try:
            self._transform_fn = getattr(self, f"_score_transform_{mode}")
        except AttributeError:
            modes = sorted(
                [
                    fn.split("_")[-1]
                    for fn in dir(self)
                    if fn.startswith("_score_transform_")
                ]
            )
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {modes}")
        self.mode = mode
        self.thresh_ = self._transform_fn(chi2.ppf(1 - self.alpha, df=1))

        if precision is not None and (
            not isinstance(precision, (int, np.integer)) or precision < 0
        ):
            raise ValueError("precision must be a non-negative integer.")
        self.precision = precision

        self._edge_tot = self._create_cms()
        self._src_tot = self._create_cms()
        self._dst_tot = self._create_cms()
        self._edge_cur = self._create_cms()
        self._src_cur = self._create_cms()
        self._dst_cur = self._create_cms()

        self._start = None
        self._last_update = None

    def update_predict_score(self, src, dst, count=1, t=None):
        now = (int(t) if t is not None else int(time())) / self.ticksize
        if self._start is None:
            self._start = now - 1

        now -= self._start

        if self._last_update is None:
            self._last_update = now

        if now > self._last_update:
            if self.decay:
                for cms in (self._edge_cur, self._src_cur, self._dst_cur):
                    for i in range(len(cms._bins)):
                        cms._bins[i] *= self.decay
            else:
                self._edge_cur.clear()
                self._src_cur.clear()
                self._dst_cur.clear()
            self._last_update = now

        edge = repr((src, dst))
        src, dst = repr(src), repr(dst)
        self._update_edge(edge, count)
        self._update_src(src, count)
        self._update_dst(dst, count)

        edge_score = self._score(
            self._edge_cur.check(edge), self._edge_tot.check(edge), t=now
        )
        src_score = self._score(
            self._src_cur.check(src), self._src_tot.check(src), t=now
        )
        dst_score = self._score(
            self._dst_cur.check(dst), self._dst_tot.check(dst), t=now
        )

        score = self.agg(edge_score, src_score, dst_score)
        if self.precision is not None:
            score = round(score, self.precision)

        score = self._transform_fn(score)

        return score

    def update_predict(self, src, dst, count=1, t=None):
        score = self.update_predict_score(src, dst, count, t)
        return score > self.thresh_

    def _create_cms(self):
        cms = CountMinSketch(
            confidence=1 - self.false_pos_prob / 2, error_rate=self.error_rate
        )
        return cms

    def _update(self, item, count, cur, tot):
        tot.add(item, count)
        cur.add(item, count)

    @lru_cache(maxsize=40960)
    def _score(self, cur, tot, t):
        if tot == 0 or t <= 1:
            score = 0
        else:
            score = (((cur - tot / t) * t) ** 2) / (tot * (t - 1))
        return score

    def _score_transform_raw(self, score):
        return score

    @lru_cache(maxsize=40960)
    def _score_transform_log(self, score):
        return np.log1p(score)

    @lru_cache(maxsize=40960)
    def _score_transform_pvalue(self, score):
        return chi2.sf(score, df=1)

    def _update_edge(self, edge, count):
        self._update(edge, count, self._edge_cur, self._edge_tot)

    def _update_src(self, node, count):
        self._update(node, count, self._src_cur, self._src_tot)

    def _update_dst(self, node, count):
        self._update(node, count, self._dst_cur, self._dst_tot)
