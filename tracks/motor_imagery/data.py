"""بيانات الكتابة بالدماغ: BNCI 2014-001 (BCI Competition IV, 2a).

9 أشخاص، كل منهم سجّل جلستين في يومين مختلفين. في كل محاولة يظهر سهم على
الشاشة فيتخيّل الشخص حركة — دون أن يتحرك فعلاً — لمدة 4 ثوانٍ:
  يد يسرى · يد يمنى · القدمان · اللسان
22 قطباً فوق مناطق الحركة، 250 عيّنة/ثانية، 288 محاولة في كل جلسة.

تخيّل حركة اليد يُضعف موجات ميو/بيتا (8–30Hz) فوق منطقة الحركة في الجهة المعاكسة
من الدماغ (اليد اليمنى ← فوق C3 يساراً). هذا "الأثر" هو ما يتعلّم النموذج قراءته.

في الملفات كما تقرؤها MOABB، كل محاولة "علامة" (annotation) تبدأ لحظة ظهور السهم
ومدتها 4 ثوانٍ (فترة التخيّل). تحققنا من ذلك بالقياس لا بالافتراض: نموذج يميّز
اليد اليسرى من اليمنى أعطى 53% (صدفة) قبل العلامة، و88% من 0.5 إلى 2.5 ث بعدها.
"""
import numpy as np

from core.config import DATA_DIR, MAINS_HZ
from core.signal.filters import apply_causal, make_chain

CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
CLASS_AR = {"left_hand": "اليد اليسرى", "right_hand": "اليد اليمنى", "feet": "القدمان", "tongue": "اللسان"}
SUBJECTS = list(range(1, 10))
SESSIONS = ("0train", "1test")  # يومان مختلفان

# النافذة التي يراها النموذج: من نصف ثانية إلى ثانيتين ونصف بعد ظهور السهم
# (النصف ثانية الأولى رد فعل بصري ولم يبدأ التخيّل بعد)
EPOCH_SEC = (0.5, 2.5)
BAND = (8.0, 30.0)
CACHE_VERSION = 1


def _dataset():
    from moabb.datasets import BNCI2014_001
    return BNCI2014_001()


def download(subjects: list[int] = SUBJECTS) -> None:
    """ينزّل ملفات الأشخاص (نحو 85MB لكل شخص) — الموجود لا يُنزَّل مرة أخرى."""
    ds = _dataset()
    for s in subjects:
        print(f"الشخص {s}…", flush=True)
        ds.data_path(s)


def epochs_from_raw(raw) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """فلترة سببية للتسجيل المتصل ثم قصّ نافذة كل محاولة.

    نفلتر التسجيل كاملاً قبل القص — كما سيحدث في البث المباشر حيث يمر الفلتر
    على الإشارة باستمرار — لا كل محاولة وحدها.
    """
    eeg = raw.copy().pick("eeg")
    fs = float(eeg.info["sfreq"])
    data = apply_causal(make_chain(fs, *BAND, MAINS_HZ), eeg.get_data() * 1e6)
    a, b = (int(round(t * fs)) for t in EPOCH_SEC)
    X, y = [], []
    for onset_sec, label in zip(raw.annotations.onset, raw.annotations.description):
        cue = int(round((onset_sec - raw.first_time) * fs))
        if label in CLASSES and cue + b <= data.shape[1]:
            X.append(data[:, cue + a : cue + b])
            y.append(CLASSES.index(label))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), fs, list(eeg.ch_names)


def load_subject(subject: int) -> dict:
    """محاولات الشخص من الجلستين: X (محاولات، قنوات، عيّنات)، y، session لكل محاولة."""
    path = DATA_DIR / "bnci2014_001_epochs" / f"subject_{subject}.v{CACHE_VERSION}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"X": z["X"], "y": z["y"], "session": z["session"], "fs": float(z["fs"]),
                "ch_names": list(z["ch_names"])}
    sessions = _dataset().get_data([subject])[subject]
    Xs, ys, ss = [], [], []
    fs, ch_names = 0.0, []
    for session in SESSIONS:
        for raw in sessions[session].values():
            X, y, fs, ch_names = epochs_from_raw(raw)
            if len(y):  # أول ثلاث تسجيلات في كل جلسة لقياس حركة العين، بلا محاولات
                Xs.append(X)
                ys.append(y)
                ss.append(np.full(len(y), session))
    out = {"X": np.concatenate(Xs), "y": np.concatenate(ys), "session": np.concatenate(ss),
           "fs": fs, "ch_names": ch_names}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{**out, "ch_names": np.array(ch_names)})
    return out
