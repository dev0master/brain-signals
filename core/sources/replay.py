"""إعادة تشغيل تسجيل محفوظ كأنه بث مباشر من جهاز.

هذا بديل الجهاز الآن: نأخذ تسجيلاً حقيقياً (مثلاً ملف EDF لمريض صرع) ونعطي
بقية النظام عيّناته بنفس سرعة الزمن الحقيقي، فيعمل كل شيء كما سيعمل مع جهاز.
"""
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .base import SignalSource, SourceInfo


class ReplaySource(SignalSource):
    def __init__(
        self,
        data: np.ndarray,
        info: SourceInfo,
        speed: float = 1.0,
        loop: bool = True,
        clock: Callable[[], float] = time.monotonic,
        start_sec: float = 0.0,
    ):
        if data.ndim != 2 or data.shape[0] != len(info.ch_names):
            raise ValueError("data يجب أن يكون بالشكل (عدد القنوات، عدد العيّنات)")
        self.data = data
        self.info = info
        self.speed = speed        # 2.0 = ضعف السرعة الحقيقية
        self.loop = loop          # عند النهاية: يعود للبداية أو يتوقف
        self._clock = clock       # قابل للاستبدال في الاختبارات
        self._pos = int(start_sec * info.fs)  # موضع العيّنة التالية
        self._t_last: float | None = None
        self._carry = 0.0         # كسور العيّنات المتبقية بين القراءات

    def start(self) -> None:
        self._t_last = self._clock()

    def read_chunk(self) -> np.ndarray:
        if self._t_last is None:
            self.start()
        now = self._clock()
        # كم عيّنة كان الجهاز سيرسل خلال الوقت المنقضي؟
        exact = (now - self._t_last) * self.info.fs * self.speed + self._carry
        n = int(exact)
        self._carry = exact - n
        self._t_last = now
        return self._take(n)

    def _take(self, n: int) -> np.ndarray:
        total = self.data.shape[1]
        if n <= 0 or (not self.loop and self._pos >= total):
            return np.empty((self.data.shape[0], 0))
        if self.loop:
            idx = (self._pos + np.arange(n)) % total
            self._pos = (self._pos + n) % total
            return self.data[:, idx]
        chunk = self.data[:, self._pos : self._pos + n]
        self._pos += chunk.shape[1]
        return chunk

    def seek(self, sec: float) -> None:
        self._pos = int(np.clip(sec * self.info.fs, 0, self.data.shape[1] - 1))

    @property
    def position_sec(self) -> float:
        return self._pos / self.info.fs

    @property
    def duration_sec(self) -> float:
        return self.data.shape[1] / self.info.fs

    @classmethod
    def from_edf(cls, path: str | Path, events: list[dict] | None = None, **kwargs) -> "ReplaySource":
        """يقرأ ملف EDF (الصيغة القياسية لتسجيلات EEG الطبية)."""
        import mne

        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="error")
        raw.pick("eeg")
        # بعض الملفات تحشو الأسماء بنقاط ("Fc5.") أو تسبقها بـ "EEG " — نزيلها لتطابق الأسماء القياسية
        cleaned = {n: n.removeprefix("EEG ").strip(". ") for n in raw.ch_names}
        if len(set(cleaned.values())) == len(cleaned):
            raw.rename_channels(cleaned)
        data_uv = raw.get_data() * 1e6  # MNE يعطي فولت، نحوّل إلى ميكروفولت
        info = SourceInfo(
            fs=float(raw.info["sfreq"]),
            ch_names=list(raw.ch_names),
            kind="replay",
            description=Path(path).name,
            events=events or [],
        )
        return cls(data_uv, info, **kwargs)
