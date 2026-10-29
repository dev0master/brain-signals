"""فلاتر الإشارة.

قاعدة مهمة: النموذج يجب أن يرى في التشغيل المباشر إشارة مُعالَجة بنفس الطريقة
التي تدرّب عليها. الفلتر "السببي" (causal) يستخدم الماضي فقط فيصلح للبث المباشر،
لذلك نستخدمه في التدريب أيضاً على التسجيل المتصل قبل تقطيعه.

الفلتر "صفري الطور" (filtfilt) يمرّ على الإشارة ذهاباً وإياباً فلا يزيح توقيتها،
لكنه يحتاج المستقبل — يصلح للعرض والتحليل بعد انتهاء التسجيل فقط.
(مشروع DSP4.m استخدم filter السببي في تحليل بعدي، فأزاح توقيت الموجات.)
"""
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos


def bandpass_sos(low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """فلتر يمرّر الترددات بين low وhigh فقط (Butterworth)."""
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def notch_sos(freq: float, fs: float, quality: float = 30.0) -> np.ndarray:
    """فلتر يحذف تردداً واحداً ضيقاً (تشويش الكهرباء 50Hz)."""
    b, a = iirnotch(freq, quality, fs=fs)
    return tf2sos(b, a)


def make_chain(fs: float, low: float = 0.5, high: float = 45.0,
               mains_hz: float | tuple[float, ...] | None = (50.0, 60.0)) -> np.ndarray:
    """سلسلة الفلاتر القياسية: تمرير نطاق + حذف تردد (أو ترددات) الكهرباء.

    تُعاد كمصفوفة sos واحدة (أقسام الفلاتر متتالية) لتطبيقها مرة واحدة.
    """
    high = min(high, fs / 2 * 0.95)  # لا يمكن تجاوز نصف معدل العيّنات (Nyquist)
    sections = [bandpass_sos(low, high, fs)]
    mains = (mains_hz,) if isinstance(mains_hz, (int, float)) else (mains_hz or ())
    for f in mains:
        if f < fs / 2:
            sections.append(notch_sos(f, fs))
    return np.vstack(sections)


def apply_causal(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    """تطبيق سببي على تسجيل كامل (n_channels, n_samples) — نفس نتيجة البث قطعةً قطعة."""
    return StreamFilter(sos, data.shape[0]).process(data)


def apply_zero_phase(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
    """تطبيق صفري الطور — للعرض والتحليل البعدي فقط، لا للبث."""
    return sosfiltfilt(sos, data, axis=-1)


class StreamFilter:
    """فلتر سببي يحفظ حالته بين القطع، فتمرّ الإشارة كأنها قطعة واحدة متصلة."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = sos
        self.n_channels = n_channels
        self._zi: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.shape[1] == 0:
            return chunk
        if self._zi is None:
            # نبدأ الحالة من أول قيمة لتفادي قفزة كبيرة في البداية
            base = sosfilt_zi(self.sos)  # (n_sections, 2)
            self._zi = base[:, None, :] * chunk[:, :1][None, :, :]
        out, self._zi = sosfilt(self.sos, chunk, axis=-1, zi=self._zi)
        return out

    def reset(self) -> None:
        self._zi = None
