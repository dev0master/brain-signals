"""بيانات الحالة الذهنية.

EEGMAT (PhysioNet) — "EEG during mental arithmetic task", Zyma et al. 2019:
  36 شخصاً، 19 قطباً بنظام 10-20، 500 عيّنة/ثانية.
  SubjectXX_1.edf : الشخص مرتاح قبل المهمة
  SubjectXX_2.edf : أثناء طرح أعداد ذهنياً (مثل 3141 − 42، ثم − 42 …)
  المقاطع منقّاة من الشوائب (حركة العين والعضلات) قبل النشر.

ملاحظتان صادقتان:
- البيانات مفلترة عند 30Hz قبل النشر، فنطاق جاما (30–45Hz) لا معنى له هنا ولا نستخدمه.
- الراحة والمهمة تسجيلان منفصلان بترتيب ثابت (الراحة أولاً). جزء من الفرق قد يأتي
  من التسجيل نفسه (تغيّر تلامس الأقطاب مع الوقت) لا من الحالة الذهنية فقط.
"""
import numpy as np

from core.config import DATA_DIR
from core.download import download_file

BASE_URL = "https://physionet.org/files/eegmat/1.0.0"
SUBJECTS = list(range(36))
CONDITIONS = {1: "rest", 2: "task"}
CONDITION_AR = {"rest": "راحة", "task": "حساب ذهني"}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3", "C4",
            "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]


def eegmat_dir():
    return DATA_DIR / "eegmat"


def file_name(subject: int, condition: int) -> str:
    return f"Subject{subject:02d}_{condition}.edf"


def path_of(subject: int, condition: int):
    return eegmat_dir() / file_name(subject, condition)


def download(subjects: list[int] = SUBJECTS) -> None:
    """36 شخصاً × ملفان ≈ 185MB. الموجود لا يُنزَّل مرة أخرى."""
    files = [(s, c) for s in subjects for c in CONDITIONS]
    for i, (s, c) in enumerate(files, 1):
        dest = path_of(s, c)
        if dest.exists():
            continue
        print(f"[{i}/{len(files)}] {dest.name}", flush=True)
        download_file(f"{BASE_URL}/{dest.name}", dest, quiet=True)
    info = eegmat_dir() / "subject-info.csv"
    if not info.exists():
        download_file(f"{BASE_URL}/subject-info.csv", info, quiet=True)


def normalize_channel(name: str) -> str:
    """'EEG Fp1' ← 'Fp1'، وأسماء النظام القديم T3/T4/T5/T6 كما هي في هذه البيانات."""
    return name.replace("EEG", "").replace("-A1", "").replace("-A2", "").strip()


def load_recording(subject: int, condition: int) -> tuple[np.ndarray, float]:
    """القنوات الـ19 بالترتيب القياسي، بالميكروفولت (float32)."""
    import mne

    raw = mne.io.read_raw_edf(str(path_of(subject, condition)), preload=False, verbose="error")
    names = [normalize_channel(n).lower() for n in raw.ch_names]
    missing = [c for c in CHANNELS if c.lower() not in names]
    if missing:
        raise ValueError(f"{file_name(subject, condition)}: قنوات مفقودة {missing}")
    raw.pick([names.index(c.lower()) for c in CHANNELS])
    data = raw.get_data()
    data *= 1e6
    return data.astype(np.float32), float(raw.info["sfreq"])


def available_subjects() -> list[int]:
    return [s for s in SUBJECTS if all(path_of(s, c).exists() for c in CONDITIONS)]
