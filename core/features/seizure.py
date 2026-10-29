"""ميزات كشف نوبات الصرع.

أثناء النوبة تُطلق مجموعات كبيرة من الخلايا العصبية معاً وبإيقاع، فتصبح الإشارة:
- أكبر سعةً وأكثر "تعرّجاً" ← line length والطاقة ترتفع
- أكثر انتظاماً وإيقاعية ← تتغير معاملات Hjorth
- تتركز قدرتها في ترددات معيّنة (غالباً ثيتا/دلتا وأحياناً بيتا)

كل ميزة تُحسب لكل قناة، ثم تُرصّ القنوات بجانب بعضها في متجه واحد للنافذة.
"""
import numpy as np
from scipy.stats import kurtosis, skew

from .bandpower import BAND_NAMES, band_powers

PER_CHANNEL = [
    "line_length", "log_variance", "hjorth_mobility", "hjorth_complexity",
    "kurtosis", "skewness",
] + [f"rel_{b}" for b in BAND_NAMES] + ["log_total_power"]


def window_features(windows: np.ndarray, fs: float) -> np.ndarray:
    """windows: (نوافذ، قنوات، عيّنات) ← (نوافذ، قنوات × عدد الميزات)."""
    x = windows - windows.mean(axis=-1, keepdims=True)
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)

    var_x = x.var(axis=-1) + 1e-12
    var_dx = dx.var(axis=-1) + 1e-12
    var_ddx = ddx.var(axis=-1) + 1e-12
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / mobility

    # line length: مجموع طول "الخط" المرسوم — يرتفع بقوة مع النوبات
    line_length = np.abs(dx).mean(axis=-1)

    bp = band_powers(x, fs)  # (نوافذ، قنوات، نطاقات)
    total = bp.sum(axis=-1, keepdims=True) + 1e-12
    rel = bp / total

    feats = np.concatenate([
        np.stack([line_length, np.log(var_x), mobility, complexity,
                  kurtosis(x, axis=-1), skew(x, axis=-1)], axis=-1),
        rel,
        np.log(total),
    ], axis=-1)  # (نوافذ، قنوات، ميزات)
    return feats.reshape(feats.shape[0], -1)


def feature_names(ch_names: list[str]) -> list[str]:
    return [f"{ch}:{f}" for ch in ch_names for f in PER_CHANNEL]
