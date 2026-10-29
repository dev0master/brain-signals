"""تقطيع الإشارة المتصلة إلى نوافذ زمنية قصيرة يحلّلها النموذج واحدة واحدة."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def sliding_windows(data: np.ndarray, fs: float, win_sec: float,
                    step_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """يقطّع (قنوات، عيّنات) إلى نوافذ متداخلة.

    يعيد:
      windows: (عدد النوافذ، قنوات، عيّنات النافذة)
      starts:  زمن بداية كل نافذة بالثواني
    """
    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if data.shape[1] < win:
        return np.empty((0, data.shape[0], win)), np.empty(0)
    view = sliding_window_view(data, win, axis=1)[:, ::step, :]  # (قنوات، نوافذ، عيّنات)
    windows = np.ascontiguousarray(view.transpose(1, 0, 2))
    starts = np.arange(windows.shape[0]) * step / fs
    return windows, starts


def window_batches(data: np.ndarray, fs: float, win_sec: float, step_sec: float,
                   batch: int = 256):
    """مثل sliding_windows لكن على دفعات — تسجيل ساعة كاملة بنوافذ متداخلة
    يحتاج مئات الميغابايت لو نُسخ دفعة واحدة."""
    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if data.shape[1] < win:
        return
    view = sliding_window_view(data, win, axis=1)[:, ::step, :]
    for i in range(0, view.shape[1], batch):
        part = np.ascontiguousarray(view[:, i : i + batch, :].transpose(1, 0, 2))
        yield part, (i + np.arange(part.shape[0])) * step / fs


class RingBuffer:
    """ذاكرة دائرية تحفظ آخر N ثانية من البث المباشر لتحليلها."""

    def __init__(self, n_channels: int, n_samples: int):
        self.buf = np.zeros((n_channels, n_samples))
        self.filled = 0  # كم عيّنة حقيقية دخلت (حتى يمتلئ)

    def push(self, chunk: np.ndarray) -> None:
        n = chunk.shape[1]
        if n == 0:
            return
        size = self.buf.shape[1]
        if n >= size:
            self.buf[:] = chunk[:, -size:]
        else:
            self.buf = np.roll(self.buf, -n, axis=1)
            self.buf[:, -n:] = chunk
        self.filled = min(size, self.filled + n)

    @property
    def full(self) -> bool:
        return self.filled >= self.buf.shape[1]

    def latest(self, n: int) -> np.ndarray:
        return self.buf[:, -n:]
