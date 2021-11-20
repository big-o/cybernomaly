from probables import CountMinSketch


class MIDAS_R:
    "https://arxiv.org/abs/1911.04464"

    def __init__(self, error_rate=0.1, false_pos_prob=0.05, window_size=10):
        self.error_rate = error_rate
        self.false_pos_prob = false_pos_prob

        edge_tot = self._create_cms()
        node_tot = self._create_cms()
        edge_cur = self._create_cms()
        node_cur = self._create_cms()

    def _create_cms(self):
        cms = CountMinSketch(
            confidence=self.false_pos_prob / 2, error_rate=self.error_rate
        )
        return cms


MIDAS_R()
