"""أسماء الأقطاب وتركيباتها.

لنظام 10-20 جيلان من الأسماء لنفس المواضع: القديم T3/T4/T5/T6 والحديث T7/T8/P7/P8.
EEGMAT يستخدم القديم، وBrainFlow وCHB-MIT الحديث — نوحّدها هنا ليعرف النظام القطب أياً كان اسمه.

التركيب "الثنائي" (bipolar) في CHB-MIT: كل قناة فرق قطبين متجاورين (مثل FP1-F7). الجهاز المنزلي
يسجّل كل قطب مقابل مرجع واحد (شحمة الأذن مثلاً)، فنحسب الفروق بالطرح: (Fp1−مرجع) − (F7−مرجع) = Fp1−F7.
"""
import numpy as np

OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# مواضع نظام 10-20 و10-10 الشائعة — لاقتراحها في الواجهة عند تسمية أقطاب الجهاز
STANDARD_10_20 = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO9", "PO7", "PO3", "POz", "PO4", "PO8", "PO10", "O1", "Oz", "O2",
]


def key(name: str) -> str:
    """مفتاح مقارنة موحّد: 'EEG Fp1-REF' و'FP1' ← 'FP1'، و'T3' ← 'T7'."""
    n = name.upper().replace("EEG", "").strip(" .")
    for suffix in ("-REF", "-A1", "-A2", "-LE"):
        n = n.removesuffix(suffix)
    n = n.strip()
    return OLD_TO_NEW.get(n, n)


def find(ch_names: list[str], wanted: list[str]) -> list[int | None]:
    """موضع كل قطب مطلوب في قائمة القنوات، أو None إن لم يوجد."""
    keys = [key(n) for n in ch_names]
    return [keys.index(key(w)) if key(w) in keys else None for w in wanted]


def present(ch_names: list[str], wanted: list[str]) -> list[str]:
    """الأقطاب المطلوبة الموجودة فعلاً (بأسمائها في wanted وبترتيبها)."""
    return [w for w, i in zip(wanted, find(ch_names, wanted)) if i is not None]


def missing(ch_names: list[str], wanted: list[str]) -> list[str]:
    return [w for w, i in zip(wanted, find(ch_names, wanted)) if i is None]


def derivable_pairs(ch_names: list[str], pairs: list[str]) -> list[str]:
    """القنوات الثنائية (مثل 'FP1-F7') التي يمكن حسابها من أقطاب الجهاز."""
    out = []
    for pair in pairs:
        a, b = pair.split("-")
        if None not in find(ch_names, [a, b]):
            out.append(pair)
    return out


def bipolar(data: np.ndarray, ch_names: list[str], pairs: list[str]) -> np.ndarray:
    """(أقطاب، عيّنات) مقابل مرجع واحد ← (أزواج، عيّنات) بالطرح."""
    out = np.empty((len(pairs), data.shape[-1]), dtype=data.dtype)
    for k, pair in enumerate(pairs):
        ia, ib = find(ch_names, pair.split("-"))
        if ia is None or ib is None:
            raise ValueError(f"لا يمكن حساب {pair}: أحد القطبين غير موجود")
        out[k] = data[ia] - data[ib]
    return out
