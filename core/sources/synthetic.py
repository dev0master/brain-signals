"""مولّد EEG اصطناعي بسيط — للاختبارات والتجربة بلا أي بيانات أو جهاز.

ليس محاكاة دقيقة للدماغ، لكنه يحمل الخصائص التي نحتاج رؤيتها:
- ضوضاء خلفية "وردية" (قدرتها أعلى في الترددات المنخفضة كما في EEG الحقيقي)
- موجات ألفا 10Hz تظهر وتختفي، أقوى في القنوات الخلفية (O1/O2)
  كما يحدث عند إغلاق العينين والاسترخاء
- تشويش الكهرباء 50Hz بسيط، ليظهر أثر فلتر notch
"""
import time
from collections.abc import Callable

import numpy as np
from scipy.signal import lfilter

from .base import SignalSource, SourceInfo

DEFAULT_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "P4", "O1", "O2"]


class SyntheticEEG(SignalSource):
    _PINK_POLES = (0.99, 0.9, 0.5)

    def __init__(
        self,
        fs: float = 256.0,
        ch_names: list[str] | None = None,
        seed: int = 0,
        mains_hz: float = 50.0,
        alpha_period_sec: float = 8.0,   # كل كم ثانية تتبدّل فترة ألفا (استرخاء/انتباه)
        clock: Callable[[], float] = time.monotonic,
    ):
        names = ch_names or DEFAULT_CHANNELS
        self.info = SourceInfo(fs=fs, ch_names=names, kind="synthetic",
                               description="إشارة اصطناعية للتجربة")
        self._rng = np.random.default_rng(seed)
        self._mains = mains_hz
        self._alpha_period = alpha_period_sec
        self._clock = clock
        self._t_last: float | None = None
        self._carry = 0.0
        self._n = 0  # عدد العيّنات المولَّدة حتى الآن (الزمن = n / fs)
        # قوة ألفا لكل قناة: أعلى في الخلف (O)، أقل في الأمام (Fp)
        self._alpha_gain = np.array([2.0 if n.startswith("O") else 1.2 if n.startswith("P")
                                     else 0.5 for n in names])
        self._pink_state = np.zeros((len(names), len(self._PINK_POLES)))

    def start(self) -> None:
        self._t_last = self._clock()

    def read_chunk(self) -> np.ndarray:
        if self._t_last is None:
            self.start()
        now = self._clock()
        exact = (now - self._t_last) * self.info.fs + self._carry
        n = int(exact)
        self._carry = exact - n
        self._t_last = now
        return self.generate(n)

    def generate(self, n: int) -> np.ndarray:
        """يولّد n عيّنة تالية (بالميكروفولت)."""
        fs = self.info.fs
        t = (self._n + np.arange(n)) / fs
        self._n += n
        n_ch = len(self.info.ch_names)

        # ضوضاء وردية تقريبية: مجموع ضوضاء بيضاء مُنعَّمة بثوابت زمنية مختلفة
        # (كل مُنعِّم: y[i] = a*y[i-1] + (1-a)*x[i]، ونحفظ حالته بين القطع)
        white = self._rng.standard_normal((n_ch, n))
        pink = np.zeros_like(white)
        for k, a in enumerate(self._PINK_POLES):
            smoothed, zf = lfilter([1 - a], [1, -a], white, axis=1, zi=self._pink_state[:, k : k + 1])
            self._pink_state[:, k] = zf[:, 0]
            pink += smoothed
        background = 25.0 * pink + 3.0 * white

        # ألفا تظهر في النصف الأول من كل دورة وتختفي في الثاني
        alpha_on = (t % self._alpha_period) < (self._alpha_period / 2)
        alpha = 15.0 * np.sin(2 * np.pi * 10.0 * t) * alpha_on
        mains = 4.0 * np.sin(2 * np.pi * self._mains * t)

        return background + self._alpha_gain[:, None] * alpha[None, :] + mains[None, :]

    def alpha_active(self, t_sec: float) -> bool:
        """هل كانت موجات ألفا ظاهرة عند هذا الزمن؟ (للاختبارات والعرض)"""
        return (t_sec % self._alpha_period) < (self._alpha_period / 2)

    @property
    def position_sec(self) -> float:
        return self._n / self.info.fs
