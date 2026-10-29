"""سلسلة كشف الصرع — نفس الدوال تُستخدم في التدريب وفي تحليل أي ملف جديد.

الخطوات لكل تسجيل:
  1. فلترة سببية 0.5–45Hz + حذف تشويش الكهرباء (نفس فلتر البث المباشر)
  2. تقطيع إلى نوافذ 4 ثوانٍ، كل نافذة تبدأ بعد السابقة بثانيتين
  3. حساب ميزات كل نافذة (line length، Hjorth، قدرة النطاقات... لكل قناة)
  4. النموذج يعطي لكل نافذة احتمال "نوبة"
  5. تنعيم: لا إنذار إلا إذا كانت 3 من آخر 4 نوافذ فوق العتبة
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from core.config import DATA_DIR, MAINS_HZ
from core.evaluation.metrics import Event, windows_to_events
from core.features.seizure import window_features
from core.signal.filters import apply_causal, make_chain
from core.signal.windows import window_batches

from .data import Recording, load_recording

WIN_SEC = 4.0
STEP_SEC = 2.0
THRESHOLD = 0.5   # احتمال النافذة الذي نعتبرها عنده "مشبوهة"
VOTE_K, VOTE_N = 3, 4
FEATURE_VERSION = 1  # غيّره إن تغيّرت الميزات ليُعاد حسابها بدل استخدام القديم


def recording_features(data_uv: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """ميزات كل نوافذ التسجيل: (نوافذ، ميزات) وزمن بداية كل نافذة."""
    filtered = apply_causal(make_chain(fs, 0.5, 45.0, MAINS_HZ), data_uv)
    feats, starts = [], []
    for windows, s in window_batches(filtered, fs, WIN_SEC, STEP_SEC):
        feats.append(window_features(windows, fs).astype(np.float32))
        starts.append(s)
    return np.vstack(feats), np.concatenate(starts)


def window_labels(starts: np.ndarray, seizures: list[tuple[float, float]]) -> np.ndarray:
    """النافذة "نوبة" إن وقع منتصفها داخل نوبة حددها الأطباء."""
    centers = starts + WIN_SEC / 2
    y = np.zeros(len(starts), dtype=np.int8)
    for s, e in seizures:
        y[(centers >= s) & (centers <= e)] = 1
    return y


def cached_features(rec: Recording) -> dict:
    """ميزات تسجيل من CHB-MIT — تُحسب مرة واحدة وتُحفظ (ساعة EEG تأخذ ثوانٍ)."""
    path = DATA_DIR / "chbmit" / "features" / f"{rec.file}.v{FEATURE_VERSION}.npz"
    if path.exists():
        z = np.load(path)
        return {"X": z["X"], "starts": z["starts"], "y": z["y"], "duration": float(z["duration"])}
    data, fs = load_recording(rec.path)
    X, starts = recording_features(data, fs)
    y = window_labels(starts, rec.seizures)
    duration = data.shape[1] / fs
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, starts=starts, y=y, duration=duration)
    return {"X": X, "starts": starts, "y": y, "duration": duration}


def make_model() -> HistGradientBoostingClassifier:
    # أشجار قرار متتالية تصحح كل منها أخطاء ما قبلها.
    # early_stopping معطّل: يقتطع 10% عشوائية للتحقق، ومع تكرار نوافذ النوبات
    # (انظر fit_balanced) ستقع نسخ متطابقة في الطرفين فيصبح التحقق بلا معنى.
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=0,
    )


def fit_balanced(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    """يدرّب النموذج بعد موازنة الفئتين.

    النوبات أقل من 1% من النوافذ، فبلا موازنة يتعلم النموذج أن "لا" دائماً تكاد تكون صحيحة.
    نكرّر نوافذ النوبات حتى يتساوى عددها مع الطبيعية. هذا مكافئ لإعطائها وزناً أكبر
    (class_weight)، لكنه في scikit-learn 1.9 أسرع بنحو 30 مرة لأن الأوزان تبطئ تقسيم القيم.
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    reps = max(1, len(neg) // max(len(pos), 1))
    idx = np.concatenate([neg, np.repeat(pos, reps)])
    return make_model().fit(X[idx], y[idx])


def decide(starts: np.ndarray, probs: np.ndarray) -> list[Event]:
    return windows_to_events(starts, probs >= THRESHOLD, WIN_SEC, VOTE_K, VOTE_N)


def timeline(starts: np.ndarray, probs: np.ndarray, detected: list[Event]) -> dict:
    """ما تحتاجه الواجهة لرسم الخط الزمني. زمن كل نقطة = نهاية النافذة
    (اللحظة التي يصبح فيها القرار متاحاً في البث المباشر)."""
    return {
        "times": np.round(starts + WIN_SEC, 1).tolist(),
        "probs": np.round(probs, 3).tolist(),
        "detected": [e.as_dict() for e in detected],
        "threshold": THRESHOLD,
    }


SETTINGS = {
    "window_sec": WIN_SEC, "step_sec": STEP_SEC, "threshold": THRESHOLD,
    "vote": f"{VOTE_K} من {VOTE_N} نوافذ", "feature_version": FEATURE_VERSION,
}
