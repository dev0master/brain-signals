"""جلسة البث المباشر: المصدر ← الفلترة ← الذاكرة الدائرية ← التحليل ← رسالة للواجهة.

كل 100ms تُستدعى step() فتقرأ العيّنات الجديدة وتعيد رسالة JSON واحدة فيها كل ما
تحتاجه الواجهة: قطعة الإشارة للرسم، قدرة النطاقات، جودة القنوات، ونتائج المحلّلات
(كاشف الصرع، الحالة الذهنية، معايرة الكتابة بالدماغ...) التي تُضاف عبر add_analyzer.
"""
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from core.config import MAINS_HZ
from core.features.bandpower import BAND_NAMES, relative_band_powers
from core.signal.filters import StreamFilter, make_chain
from core.signal.quality import channel_quality
from core.signal.windows import RingBuffer
from core.sources.base import SignalSource
from core.sources.replay import save_edf

DISPLAY_FS = 128.0     # يكفي للرسم بالعين ويخفف حجم الرسائل
BANDS_WINDOW_SEC = 2.0
BUFFER_SEC = 30.0      # تحتفظ الجلسة بآخر 30 ثانية للمحلّلات
MAX_RECORD_SEC = 3 * 3600  # حدّ التسجيل: 3 ساعات (16 قناة بـ250Hz ≈ 170MB في الذاكرة)

# المحلّل: دالة تأخذ (الجلسة) وتعيد قاموساً يُضاف للرسالة، أو None إن لم يحن وقته
Analyzer = Callable[["LiveSession"], dict | None]


class LiveSession:
    def __init__(self, source: SignalSource, mains_hz: tuple[float, ...] = MAINS_HZ,
                 record_to: Path | None = None):
        self.source = source
        info = source.info
        self.fs = info.fs
        self.ch_names = list(info.ch_names)
        self.n_ch = len(info.ch_names)
        self.filter = StreamFilter(make_chain(self.fs, 0.5, 45.0, mains_hz), self.n_ch)
        self.buffer = RingBuffer(self.n_ch, int(BUFFER_SEC * self.fs))
        self.decim = max(1, int(round(self.fs / DISPLAY_FS)))
        self._decim_offset = 0  # لتبقى عيّنات العرض منتظمة عبر القطع
        self.analyzers: dict[str, Analyzer] = {}
        self.started = False
        # ساعة الجلسة: كم عيّنة وصلت منذ البداية — المحلّلات تحسب بها الزمن (لا بموضع التسجيل)
        self.n_samples = 0
        self.last_raw = np.empty((self.n_ch, 0))       # ما وصل في هذه الخطوة قبل الفلترة
        self.last_filtered = np.empty((self.n_ch, 0))  # وبعدها
        # التسجيل: الإشارة الخام كما وصلت + علامات (مثل لحظة ظهور كل أمر في المعايرة)
        self.record_to = record_to
        self._recorded: list[np.ndarray] = []
        self.markers: list[dict] = []
        self.saved_path: Path | None = None
        # step وclose قد يُستدعيان من خيطين (الخادم يعالج في خيط منفصل) — لا يتداخلان
        self._lock = threading.Lock()
        self._closed = False

    @property
    def elapsed_sec(self) -> float:
        return self.n_samples / self.fs

    def add_analyzer(self, name: str, analyzer: Analyzer) -> None:
        self.analyzers[name] = analyzer

    def mark(self, description: str, onset_sec: float | None = None, duration: float = 0.0) -> None:
        """علامة في التسجيل المحفوظ (تظهر كـ annotation في ملف EDF)."""
        onset = self.elapsed_sec if onset_sec is None else onset_sec
        self.markers.append({"onset": round(onset, 4), "duration": duration, "description": description})

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
            "recording": self.record_to.name if self.record_to else None,
        }

    def step(self) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("الجلسة مغلقة")
            return self._step()

    def _step(self) -> dict:
        if not self.started:
            self.source.start()
            self.started = True
        raw = self.source.read_chunk()
        filtered = self.filter.process(raw)
        self.buffer.push(filtered)
        self.last_raw, self.last_filtered = raw, filtered
        self.n_samples += raw.shape[1]
        if self.record_to is not None and raw.shape[1] and self.elapsed_sec <= MAX_RECORD_SEC:
            self._recorded.append(raw.astype(np.float32))

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
        if self.record_to is not None:
            msg["recording_sec"] = round(min(self.elapsed_sec, MAX_RECORD_SEC), 1)
        for name, analyzer in self.analyzers.items():
            result = analyzer(self)
            if result is not None:
                msg[name] = result
        return msg

    def recorded(self) -> np.ndarray:
        """كل الإشارة الخام المسجّلة حتى الآن (قنوات، عيّنات)."""
        if not self._recorded:
            return np.empty((self.n_ch, 0), dtype=np.float32)
        return np.concatenate(self._recorded, axis=1)

    def save(self) -> Path | None:
        """يحفظ التسجيل كملف EDF (إن طُلب التسجيل ووصلت ثانية على الأقل)."""
        data = self.recorded()
        if self.record_to is None or data.shape[1] < self.fs:
            return None
        end = data.shape[1] / self.fs
        # علامة بدأت قبل نهاية التسجيل وتمتد بعدها (أُغلقت الجلسة أثناء أمر) تُقصّ عند النهاية
        markers = [{**m, "duration": min(m["duration"], end - m["onset"])} for m in self.markers if m["onset"] < end]
        self.saved_path = save_edf(self.record_to, data, self.fs, self.ch_names, markers)
        return self.saved_path

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.source.stop()
            finally:
                self.save()
