"""قدرة نطاقات الترددات — أبسط وأهم "ميزة" في EEG.

الدماغ ينتج موجات بترددات مختلفة، ولكل نطاق دلالة تقريبية:
  دلتا 0.5–4Hz  : النوم العميق
  ثيتا 4–8Hz    : النعاس، التأمل، الذاكرة
  ألفا 8–13Hz   : الاسترخاء بعينين مغمضتين (تضعف مع الانتباه البصري)
  بيتا 13–30Hz  : التركيز والتفكير النشط والتوتر
  جاما 30–45Hz  : معالجة معقّدة (وتختلط بسهولة بنشاط العضلات)

نوقف جاما عند 45Hz لا 50Hz (كما في DSP4.m) حتى لا يدخل تشويش الكهرباء 50Hz.
"""
import numpy as np
from scipy.signal import welch

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)


def psd(data: np.ndarray, fs: float, seg_sec: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """طيف القدرة بطريقة Welch: كم من طاقة الإشارة في كل تردد."""
    nperseg = min(data.shape[-1], int(seg_sec * fs))
    return welch(data, fs=fs, nperseg=nperseg, axis=-1)


def band_powers(data: np.ndarray, fs: float, bands: dict = BANDS) -> np.ndarray:
    """القدرة المطلقة لكل نطاق (µV²).

    data: (..., عيّنات) ← النتيجة: (..., عدد النطاقات)
    """
    freqs, pxx = psd(data, fs)
    out = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        out.append(np.trapezoid(pxx[..., mask], freqs[mask], axis=-1))
    return np.stack(out, axis=-1)


def relative_band_powers(data: np.ndarray, fs: float, bands: dict = BANDS) -> np.ndarray:
    """نسبة كل نطاق من مجموع النطاقات — لا تتأثر بقوة التلامس أو المضخّم."""
    bp = band_powers(data, fs, bands)
    return bp / np.maximum(bp.sum(axis=-1, keepdims=True), 1e-12)


def frontal_alpha_asymmetry(alpha_power: np.ndarray, ch_names: list[str],
                            left: str = "F3", right: str = "F4") -> float | None:
    """عدم تناظر ألفا الأمامي: ln(ألفا يمين) − ln(ألفا يسار).

    فرضية بحثية تربط القيم الموجبة بالمزاج الإيجابي والإقبال، والسالبة بالانسحاب.
    الدليل عليها متضارب — نعرضها كمؤشر للتعلّم لا كحكم على الشعور.
    """
    names = [n.lower() for n in ch_names]
    if left.lower() not in names or right.lower() not in names:
        return None
    a_l = alpha_power[names.index(left.lower())]
    a_r = alpha_power[names.index(right.lower())]
    return float(np.log(max(a_r, 1e-12)) - np.log(max(a_l, 1e-12)))
