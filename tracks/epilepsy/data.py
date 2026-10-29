"""بيانات الصرع: CHB-MIT من PhysioNet.

تسجيلات EEG لأطفال مصابين بالصرع في مستشفى بوسطن للأطفال، 256 عيّنة/ثانية،
كل ملف ساعة تقريباً. الأطباء حدّدوا بداية ونهاية كل نوبة بالثانية في ملف
chbXX-summary.txt — هذه هي "الإجابات الصحيحة" التي نقيس عليها النموذج.

الأقطاب مركّبة "ثنائية القطب": كل قناة فرق بين قطبين متجاورين (مثل FP1-F7).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.config import DATA_DIR
from core.download import download_file

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
PATIENTS = ["chb01", "chb03", "chb05"]  # 7 + 7 + 5 = 19 نوبة
FS = 256.0

# القنوات الـ18 القياسية (ترتيب "الموزة المزدوجة") — موجودة في كل ملفات هؤلاء المرضى
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


@dataclass
class Recording:
    patient: str
    file: str                                   # مثل chb01_03.edf
    seizures: list[tuple[float, float]] = field(default_factory=list)  # (بداية، نهاية) بالثواني

    @property
    def path(self) -> Path:
        return chb_dir(self.patient) / self.file

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.patient}/{self.file}"


def chb_dir(patient: str) -> Path:
    return DATA_DIR / "chbmit" / patient


def parse_summary(text: str, patient: str) -> list[Recording]:
    """يقرأ ملف الملخص: اسم كل ملف وأوقات نوباته.

    يدعم الصيغتين الموجودتين في CHB-MIT:
      "Seizure Start Time: 2996 seconds" و "Seizure 1 Start Time: 2996 seconds"
    """
    recordings: list[Recording] = []
    start = None
    for line in text.splitlines():
        line = line.strip()
        if m := re.match(r"File Name:\s*(\S+\.edf)", line):
            recordings.append(Recording(patient, m.group(1)))
        elif m := re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line):
            start = float(m.group(1))
        elif (m := re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)) and start is not None:
            recordings[-1].seizures.append((start, float(m.group(1))))
            start = None
    return recordings


def load_summary(patient: str) -> list[Recording]:
    path = chb_dir(patient) / f"{patient}-summary.txt"
    if not path.exists():
        return []
    return parse_summary(path.read_text(encoding="utf-8", errors="replace"), patient)


def downloaded_recordings(patients: list[str] = PATIENTS) -> list[Recording]:
    """التسجيلات المنزَّلة فعلاً (مع أوقات نوباتها)."""
    return [r for p in patients for r in load_summary(p) if r.path.exists()]


def seizures_for_path(path: Path) -> list[dict]:
    """أوقات النوبات المعروفة لملف — تُعرض في البث المباشر للمقارنة."""
    if not re.match(r"chb\d+_\d+.*\.edf$", path.name):
        return []
    patient = path.name.split("_")[0]
    for rec in load_summary(patient):
        if rec.file == path.name:
            return [{"start": s, "end": e, "label": "نوبة مسجّلة طبياً"} for s, e in rec.seizures]
    return []


# ---------- التنزيل ----------

def choose_files(recordings: list[Recording], n_free: int) -> list[Recording]:
    """كل الملفات التي فيها نوبات + n_free ملفات بلا نوبات موزّعة على طول التسجيل.

    الملفات الخالية ضرورية: بها نعرف كم مرة ينذر النموذج بلا سبب.
    """
    with_seizures = [r for r in recordings if r.seizures]
    free = [r for r in recordings if not r.seizures]
    if n_free and free:
        idx = np.linspace(0, len(free) - 1, min(n_free, len(free))).round().astype(int)
        free = [free[i] for i in sorted(set(idx))]
    else:
        free = []
    return sorted(with_seizures + free, key=lambda r: r.file)


def fetch_summary(patient: str) -> list[Recording]:
    path = chb_dir(patient) / f"{patient}-summary.txt"
    if not path.exists():
        download_file(f"{BASE_URL}/{patient}/{patient}-summary.txt", path)
    return load_summary(patient)


def plan_download(patients: list[str], n_free: int) -> list[Recording]:
    return [r for p in patients for r in choose_files(fetch_summary(p), n_free)]


def download(recordings: list[Recording]) -> None:
    for i, rec in enumerate(recordings, 1):
        if rec.path.exists():
            print(f"[{i}/{len(recordings)}] موجود: {rec.file}")
            continue
        print(f"[{i}/{len(recordings)}] تنزيل {rec.file} ({len(rec.seizures)} نوبة)")
        download_file(rec.url, rec.path)


# ---------- القراءة ----------

def select_channels(ch_names: list[str]) -> list[int] | None:
    """مواضع القنوات الـ18 القياسية بالترتيب، أو None إن نقصت إحداها.

    MNE يعيد تسمية القنوات المكررة ("T8-P8" مرتين ← "T8-P8-0" و"T8-P8-1")
    فنقبل الاسم الأصلي أو الاسم متبوعاً برقم التكرار.
    """
    upper = [n.upper().strip() for n in ch_names]
    out = []
    for std in CHANNELS:
        idx = next((i for i, n in enumerate(upper)
                    if n == std or (n.startswith(std + "-") and n[len(std) + 1:].isdigit())), None)
        if idx is None:
            return None
        out.append(idx)
    return out


def load_recording(path: Path) -> tuple[np.ndarray, float]:
    """يقرأ ملف EDF ويعيد القنوات الـ18 القياسية بالميكروفولت (float32).

    نختار القنوات قبل القراءة فلا يُحمَّل من القرص إلا ما نحتاجه — ساعة كاملة
    بكل القنوات بدقة float64 تأخذ نحو 170MB.
    """
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="error")
    idx = select_channels(raw.ch_names)
    if idx is None:
        raise ValueError(f"{path.name}: لا يحتوي القنوات القياسية الـ18")
    raw.pick(idx)
    data = raw.get_data()
    data *= 1e6
    return data.astype(np.float32), float(raw.info["sfreq"])
