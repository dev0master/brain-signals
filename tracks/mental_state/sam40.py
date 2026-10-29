"""SAM40 — مقياس «مسترخٍ ↔ تحت مهمة مجهِدة» (أردناه «هادئ ↔ متوتر»؛ انظر التنبيه أدناه).

Ghosh et al. 2022, Data in Brief 40:107772 — figshare، رخصة CC BY 4.0.
40 شخصاً، 32 قطباً (Emotiv)، 128 عيّنة/ثانية، كل محاولة 25 ثانية، 3 محاولات لكل حالة:
  Relax         : استرخاء
  Stroop        : قراءة لون الكلمة لا الكلمة نفسها (مثل كلمة «أحمر» مكتوبة بالأزرق) تحت ضغط الوقت
  Arithmetic    : حساب ذهني تحت ضغط الوقت
  Mirror_image  : تمييز الصورة المتناظرة
بعد كل محاولة مهمة قيّم الشخص توتره من 1 إلى 10 (scales.xls).

تنبيه صادق: المهام صُمّمت لإحداث التوتر، لكنها أيضاً «انشغال ذهني». فالمقياس قد يقيس
الانشغال لا التوتر بالذات. لذلك نختبر: هل ترتفع قيمته في المحاولات التي قال أصحابها
إنها أكثر توتراً؟ (ارتباط مع التقييم الذاتي).

التنزيل: ملف Data.rar واحد (760MB) من figshare، ونستخرج منه raw_data و scales.xls فقط.
"""
import numpy as np

from core.config import DATA_DIR

FIGSHARE_URL = "https://ndownloader.figshare.com/files/27956376"
FS = 128.0
SUBJECTS = list(range(1, 41))
TRIALS = (1, 2, 3)
TASKS = {"Stroop": "ستروب", "Arithmetic": "حساب ذهني", "Mirror_image": "صورة المرآة"}
RATING_COLUMN = {"Arithmetic": "Maths", "Mirror_image": "Symmetry", "Stroop": "Stroop"}
CHANNELS = ["Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9", "T7", "CP5", "CP1", "P3", "P7",
            "PO9", "O1", "Pz", "Oz", "O2", "PO10", "P8", "P4", "CP2", "CP6", "T8", "FT10", "FC6", "C4",
            "FC2", "F4", "F8", "Fp2"]  # من Coordinates.locs بنفس ترتيب صفوف الملفات


def root():
    return DATA_DIR / "sam40" / "extracted" / "Data"


def trial_path(kind: str, subject: int, trial: int):
    return root() / "raw_data" / f"{kind}_sub_{subject}_trial{trial}.mat"


def available_subjects() -> list[int]:
    kinds = ["Relax", *TASKS]
    return [s for s in SUBJECTS if all(trial_path(k, s, t).exists() for k in kinds for t in TRIALS)]


def load_trial(kind: str, subject: int, trial: int) -> np.ndarray:
    """(32 قناة، 3200 عيّنة) بالميكروفولت."""
    import scipy.io

    m = scipy.io.loadmat(str(trial_path(kind, subject, trial)))
    return np.asarray(m["Data"], dtype=np.float32)


def ratings() -> dict[tuple[int, str, int], float]:
    """تقييم التوتر الذاتي (1–10) لكل (شخص، مهمة، محاولة)."""
    import pandas as pd

    xl = pd.read_excel(root() / "scales.xls", header=None, engine="xlrd")
    trial_of_col, task_of_col = {}, {}
    current = None
    for col in range(1, xl.shape[1]):
        head = xl.iat[0, col]
        if isinstance(head, str) and head.startswith("Trial_"):
            current = int(head.split("_")[1])
        trial_of_col[col] = current
        task_of_col[col] = str(xl.iat[1, col]).strip()
    column_task = {v: k for k, v in RATING_COLUMN.items()}
    out = {}
    for row in range(2, xl.shape[0]):
        subject = xl.iat[row, 0]
        if not isinstance(subject, (int, float)) or np.isnan(subject):
            continue
        for col in range(1, xl.shape[1]):
            task = column_task.get(task_of_col[col])
            value = xl.iat[row, col]
            if task and trial_of_col[col] and isinstance(value, (int, float)) and not np.isnan(value):
                out[(int(subject), task, trial_of_col[col])] = float(value)
    return out


def download_and_extract() -> None:
    """ينزّل Data.rar (760MB) ويستخرج raw_data و scales.xls بـ 7-Zip."""
    import shutil
    import subprocess

    from core.download import download_file

    folder = DATA_DIR / "sam40"
    rar = folder / "Data.rar"
    if not rar.exists():
        download_file(FIGSHARE_URL, rar)
    if available_subjects():
        return
    seven = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"
    subprocess.run([seven, "x", "-y", f"-o{folder / 'extracted'}", str(rar),
                    "Data/raw_data/*", "Data/scales.xls", "Data/Coordinates.locs"], check=True)
