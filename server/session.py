"""جلسة البث المباشر: المصدر ← الفلترة ← الذاكرة الدائرية ← التحليل ← رسالة للواجهة.

كل 100ms تُستدعى step() فتقرأ العيّنات الجديدة وتعيد رسالة JSON واحدة فيها كل ما
تحتاجه الواجهة: قطعة الإشارة للرسم، قدرة النطاقات، جودة القنوات، ونتائج المحلّلات
(كاشف الصرع، الحالة الذهنية...) التي تُضاف لاحقاً عبر add_analyzer.
"""
from collections.abc import Callable

import numpy as np

from core.config import MAINS_HZ
from core.features.bandpower import BAND_NAMES, relative_band_powers
from core.signal.filters import StreamFilter, make_chain
from core.signal.quality import channel_quality
from core.signal.windows import RingBuffer
from core.sources.base import SignalSource

DISPLAY_FS = 128.0     # يكفي للرسم بالعين ويخفف حجم الرسائل
BANDS_WINDOW_SEC = 2.0
BUFFER_SEC = 30.0      # تحتفظ الجلسة بآخر 30 ثانية للمحلّلات

# المحلّل: دالة تأخذ (الجلسة) وتعيد قاموساً يُضاف للرسالة، أو None إن لم يحن وقته
Analyzer = Callable[["LiveSession"], dict | None]


class LiveSession:
    def __init__(self, source: SignalSource, mains_hz: tuple[float, ...] = MAINS_HZ):
        self.source = source
        info = source.info
        self.fs = info.fs
        self.n_ch = len(info.ch_names)
        self.filter = StreamFilter(make_chain(self.fs, 0.5, 45.0, mains_hz), self.n_ch)
        self.buffer = RingBuffer(self.n_ch, int(BUFFER_SEC * self.fs))
        self.decim = max(1, int(round(self.fs / DISPLAY_FS)))
        self._decim_offset = 0  # لتبقى عيّنات العرض منتظمة عبر القطع
        self.analyzers: dict[str, Analyzer] = {}
        self.started = False

    def add_analyzer(self, name: str, analyzer: Analyzer) -> None:
        self.analyzers[name] = analyzer

    def hello(self) -> dict:
        """أول رسالة: وصف المصدر لترسم الواجهة القنوات بأسمائها."""
        info = self.source.info
        return {
            "type": "hello",
            "fs": self.fs,
            "display_fs": self.fs / self.decim,
            "ch_names": info.ch_names,
            "kind": info.kind,
            "description": info.description,
            "events": info.events,
            "bands": BAND_NAMES,
            "duration_sec": getattr(self.source, "duration_sec", None),
        }

    def step(self) -> dict:
        if not self.started:
            self.source.start()
            self.started = True
        raw = self.source.read_chunk()
        filtered = self.filter.process(raw)
        self.buffer.push(filtered)

        # تخفيض معدل العيّنات للعرض (الإشارة مفلترة أصلاً تحت 45Hz)
        display = filtered[:, self._decim_offset :: self.decim]
        self._decim_offset = (self._decim_offset - filtered.shape[1]) % self.decim

        msg: dict = {
            "type": "frame",
            "t": round(self.source.position_sec, 3),
            "samples": np.round(display, 2).tolist(),
        }
        n_bands = int(BANDS_WINDOW_SEC * self.fs)
        if self.buffer.filled >= n_bands:
            recent = self.buffer.latest(n_bands)
            msg["bands"] = np.round(relative_band_powers(recent, self.fs), 4).tolist()
            msg["quality"] = channel_quality(recent)
        for name, analyzer in self.analyzers.items():
            result = analyzer(self)
            if result is not None:
                msg[name] = result
        return msg

    def close(self) -> None:
        self.source.stop()
