from functools import lru_cache
from time import time

import numpy as np
from probables import CountMinSketch
from scipy.stats import chi2

from cybernomaly.anomaly_detection.base import Monitor


class MIDAS_R(Monitor):
    """
    Anomaly detector for a simple stream of graph edgesi using the MIDAS-R [1]_
    algorithm. Works on simple edge counts and timings, without any more advanced
    features.

    To also account for node/edge properties, MStream is the multi-dimensional
    equivalent.

    References
    ----------
    .. [1] MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams
           https://arxiv.org/abs/1911.04464
    """

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

    def partial_fit(self, src, dst):
        return self

    def update(self, src, dst, count=1, t=None):
        edge, src, dst = self._format_keys(src, dst)
        t = self._snap_time(t if t is not None else time(), self.ticksize)
        return self._update(edge, src, dst, count, t)

    def _update(self, edge, src, dst, count, t):
        if self._start is None:
            self._start = t - 1

        t -= self._start
        self.now_ = t

        if self._last_update is None:
            self._last_update = t

        if t > self._last_update:
            if self.decay:
                for cms in (self._edge_cur, self._src_cur, self._dst_cur):
                    for i in range(len(cms._bins)):
                        cms._bins[i] *= self.decay
            else:
                self._edge_cur.clear()
                self._src_cur.clear()
                self._dst_cur.clear()
            self._last_update = t

        self._update_edge(edge, count)
        self._update_src(src, count)
        self._update_dst(dst, count)

    def detect_score(self, src, dst):
        edge, src, dst = self._format_keys(src, dst)
        t = self._snap_time(t if t is not None else time(), self.ticksize)
        return self._detect_score(edge, src, dst)

    def detect(self, src, dst):
        return self.detect_score(src, dst) > self.thresh_

    def _detect_score(self, edge, src, dst):
        edge, src, dst = self._format_keys(src, dst)
        edge_score = self._score(
            self._edge_cur.check(edge), self._edge_tot.check(edge)
        )
        src_score = self._score(
            self._src_cur.check(src), self._src_tot.check(src)
        )
        dst_score = self._score(
            self._dst_cur.check(dst), self._dst_tot.check(dst)
        )

        score = self.agg(edge_score, src_score, dst_score)
        if self.precision is not None:
            score = round(score, self.precision)

        score = self._transform_fn(score)

        return score

    def update_detect_score(self, src, dst, count=1, t=None):
        edge, src, dst = self._format_keys(src, dst)
        t = self._snap_time(t if t is not None else time(), self.ticksize)
        self._update(edge, src, dst, count, t)
        return self._detect_score(edge, src, dst)

    def update_detect(self, src, dst, count=1, t=None):
        return self.update_detect_score(src, dst, count, t) > self.thresh_

    def _format_keys(self, src, dst):
        edge = repr((src, dst))
        src = repr(src)
        dst = repr(dst)
        return edge, src, dst

    def _create_cms(self):
        cms = CountMinSketch(
            confidence=1 - self.false_pos_prob / 2, error_rate=self.error_rate
        )
        return cms

    def _update_cms(self, item, count, cur, tot):
        tot.add(item, count)
        cur.add(item, count)

    @lru_cache(maxsize=40960)
    def _score(self, cur, tot):
        t = self.now_
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
        self._update_cms(edge, count, self._edge_cur, self._edge_tot)

    def _update_src(self, node, count):
        self._update_cms(node, count, self._src_cur, self._src_tot)

    def _update_dst(self, node, count):
        self._update_cms(node, count, self._dst_cur, self._dst_tot)
