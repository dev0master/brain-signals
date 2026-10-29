import numpy as np

from core.features.bandpower import BAND_NAMES, relative_band_powers
from core.sources import ReplaySource, SourceInfo, SyntheticEEG


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_replay_returns_samples_matching_elapsed_time_and_loops():
    clock = FakeClock()
    data = np.arange(2 * 100, dtype=float).reshape(2, 100)
    src = ReplaySource(data, SourceInfo(fs=100, ch_names=["a", "b"], kind="replay"), clock=clock)
    src.start()
    clock.t = 0.25
    first = src.read_chunk()
    assert first.shape == (2, 25)
    np.testing.assert_array_equal(first[0], np.arange(25))
    clock.t = 1.05  # 80 عيّنة أخرى ← تتجاوز النهاية وتعود للبداية
    second = src.read_chunk()
    assert second.shape == (2, 80)
    assert second[0, -1] == 4  # (25 + 80 - 1) % 100


def test_replay_without_loop_stops_at_end():
    clock = FakeClock()
    data = np.ones((1, 50))
    src = ReplaySource(data, SourceInfo(fs=100, ch_names=["a"], kind="replay"),
                       clock=clock, loop=False)
    src.start()
    clock.t = 2.0
    assert src.read_chunk().shape == (1, 50)
    clock.t = 3.0
    assert src.read_chunk().shape == (1, 0)


def test_synthetic_alpha_appears_in_occipital_channel_when_active():
    src = SyntheticEEG(fs=256, alpha_period_sec=8.0, seed=0)
    data = src.generate(int(8 * 256))  # دورة كاملة: 4 ثوان ألفا ثم 4 بدونها
    o1 = src.info.ch_names.index("O1")
    alpha = BAND_NAMES.index("alpha")
    on = relative_band_powers(data[o1, : 4 * 256][None], 256)[0, alpha]
    off = relative_band_powers(data[o1, 4 * 256 :][None], 256)[0, alpha]
    assert on > 2 * off
