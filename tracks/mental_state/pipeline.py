"""سلسلة الحالة الذهنية: من الإشارة إلى "كم ابتعدتَ عن حالة راحتك".

لكل نافذة 4 ثوانٍ نحسب لكل قطب نصيب كل نطاق من الطاقة (دلتا، ثيتا، ألفا، بيتا)،
ونسباً معروفة في الأبحاث:
  ثيتا/بيتا أمامياً   — تنخفض عادة مع الانتباه
  ألفا/بيتا خلفياً    — ألفا الخلفية تضعف حين ينشغل الدماغ
  عدم تناظر ألفا الأمامي (F4 مقابل F3) — مؤشر بحثي ضعيف الدليل، نعرضه للتعلّم

المعايرة: سعة موجات الدماغ تختلف كثيراً بين الناس (سُمك الجمجمة، تلامس الأقطاب…).
فبدلاً من مقارنة شخص بأشخاص آخرين، نطلب منه دقيقة راحة في البداية ونقيس كل شيء
بعدها كانحراف عن خط الأساس هذا. هكذا تعمل أجهزة كثيرة فعلاً.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.config import MAINS_HZ
from core.features.bandpower import BANDS, relative_band_powers
from core.signal.filters import apply_causal, make_chain
from core.signal.windows import window_batches

from .data import CHANNELS

WIN_SEC = 4.0
STEP_SEC = 2.0
SETTLE_SEC = 4.0      # أول ثوانٍ من كل تسجيل يستقر فيها الفلتر — لا نستخدمها
CALIB_SEC = 60.0      # دقيقة الراحة الأولى = خط الأساس الشخصي
BANDS4 = {k: BANDS[k] for k in ("delta", "theta", "alpha", "beta")}  # جاما مفلترة في EEGMAT
FRONTAL = ["Fz", "F3", "F4"]
POSTERIOR = ["P3", "P4", "Pz", "O1", "O2"]

BAND_AR = {"delta": "دلتا", "theta": "ثيتا", "alpha": "ألفا", "beta": "بيتا"}


def feature_names(channels: list[str] = CHANNELS) -> list[str]:
    per_channel = [f"{BAND_AR[b]} عند {ch}" for ch in channels for b in BANDS4]
    return per_channel + ["ثيتا/بيتا أمامياً", "ألفا/بيتا خلفياً", "عدم تناظر ألفا الأمامي"]


def window_features(windows: np.ndarray, fs: float, channels: list[str] = CHANNELS) -> np.ndarray:
    """(نوافذ، قنوات، عيّنات) ← (نوافذ، 4 × القنوات + 3 ميزات)."""
    rel = relative_band_powers(windows, fs, BANDS4)  # (نوافذ، قنوات، 4)
    log_rel = np.log(np.maximum(rel, 1e-6))
    idx = {c: i for i, c in enumerate(channels)}
    b = {name: i for i, name in enumerate(BANDS4)}

    def mean_band(chs, band):
        return rel[:, [idx[c] for c in chs], b[band]].mean(axis=1)

    theta_beta = np.log(mean_band(FRONTAL, "theta") / mean_band(FRONTAL, "beta"))
    alpha_beta = np.log(mean_band(POSTERIOR, "alpha") / mean_band(POSTERIOR, "beta"))
    faa = log_rel[:, idx["F4"], b["alpha"]] - log_rel[:, idx["F3"], b["alpha"]]
    return np.concatenate([log_rel.reshape(len(rel), -1),
                           np.stack([theta_beta, alpha_beta, faa], axis=1)], axis=1).astype(np.float32)


def recording_features(data_uv: np.ndarray, fs: float,
                       channels: list[str] = CHANNELS) -> tuple[np.ndarray, np.ndarray]:
    """ميزات كل نوافذ التسجيل وزمن بداية كل نافذة، بعد إسقاط فترة استقرار الفلتر."""
    filtered = apply_causal(make_chain(fs, 0.5, 45.0, MAINS_HZ), data_uv)
    feats, starts = [], []
    for windows, s in window_batches(filtered, fs, WIN_SEC, STEP_SEC):
        feats.append(window_features(windows, fs, channels))
        starts.append(s)
    X, starts = np.vstack(feats), np.concatenate(starts)
    keep = starts >= SETTLE_SEC
    return X[keep], starts[keep]


def calibrate(X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """كل ميزة كانحراف عن خط الأساس الشخصي (بوحدات تشتته)."""
    mu = baseline.mean(axis=0)
    sd = baseline.std(axis=0) + 1e-6
    return (X - mu) / sd


def make_model():
    # انحدار لوجستي: بسيط ويمكن قراءة أوزانه — نعرف أي ميزة دفعت القرار.
    # class_weight="balanced": عدد نوافذ الحالتين غير متساوٍ (في SAM40 تسع محاولات مهام مقابل
    # محاولتي استرخاء)، وبدونه ينحاز منتصف المقياس نحو الحالة الأكثر عدداً.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))


SETTINGS = {"window_sec": WIN_SEC, "step_sec": STEP_SEC, "calibration_sec": CALIB_SEC,
            "bands": list(BANDS4), "model": "انحدار لوجستي على قدرة النطاقات لكل قطب + 3 نسب"}
