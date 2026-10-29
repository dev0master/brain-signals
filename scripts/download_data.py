"""تنزيل مجموعات البيانات.

    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.download_data chbmit            صرع: يعرض الحجم ويطلب التأكيد
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.download_data chbmit --yes      بلا سؤال
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.download_data chbmit --free 10  ملفات خالية أكثر
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.download_data bnci2014_001      الكتابة بالدماغ (~770MB)
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.download_data eegmat            مرتاح ↔ مركّز (~185MB)
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.download_data sam40             مسترخٍ ↔ مهمة مجهِدة (~760MB)

الملفات الموجودة لا تُنزَّل مرة أخرى، فيمكن إيقاف التنزيل واستئنافه.
"""
import argparse
import shutil
import sys

from core.config import DATA_DIR, ensure_dirs
from tracks.epilepsy import data as chb
from tracks.mental_state import data as eegmat
from tracks.mental_state import sam40
from tracks.motor_imagery import data as mi

CHB_FILE_MB = 42  # حجم ملف CHB-MIT لساعة واحدة تقريباً
BNCI_SUBJECT_MB = 86  # ملفّا الجلستين لكل شخص


def confirm(need_gb: float, yes: bool) -> bool:
    free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
    print(f"المطلوب تنزيله ≈ {need_gb:.1f} GB · المساحة الحرة: {free_gb:.1f} GB")
    if need_gb > free_gb - 2:
        print("المساحة غير كافية ✘")
        return False
    return yes or input("متابعة؟ [y/N] ").strip().lower() in {"y", "yes", "نعم", "ن"}


def chbmit(args) -> int:
    plan = chb.plan_download(args.patients, args.free)
    missing = [r for r in plan if not r.path.exists()]
    n_seiz = sum(len(r.seizures) for r in plan)
    print(f"المرضى: {', '.join(args.patients)} · {len(plan)} ملفاً ({n_seiz} نوبة)")
    print(f"المجلد: {DATA_DIR / 'chbmit'}")
    if not missing:
        print("كل الملفات موجودة ✔")
        return 0
    if not confirm(len(missing) * CHB_FILE_MB / 1000, args.yes):
        return 1
    chb.download(plan)
    return 0


def bnci(args) -> int:
    print(f"الكتابة بالدماغ: {len(mi.SUBJECTS)} أشخاص × جلستان (BNCI 2014-001)")
    if not confirm(len(mi.SUBJECTS) * BNCI_SUBJECT_MB / 1000, args.yes):
        return 1
    mi.download()
    return 0


def eegmat_dl(args) -> int:
    print(f"الحالة الذهنية: {len(eegmat.SUBJECTS)} شخصاً × ملفان (EEGMAT)")
    if not confirm(0.185, args.yes):
        return 1
    eegmat.download()
    return 0


def sam40_dl(args) -> int:
    print("التوتر: SAM40 — ملف Data.rar واحد (760MB)، نستخرج منه البيانات الخام فقط (~360MB) بـ 7-Zip")
    if not confirm(1.1, args.yes):
        return 1
    sam40.download_and_extract()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["chbmit", "bnci2014_001", "eegmat", "sam40"])
    parser.add_argument("--patients", nargs="+", default=chb.PATIENTS)
    parser.add_argument("--free", type=int, default=6, help="عدد الساعات الخالية من النوبات لكل مريض")
    parser.add_argument("--yes", action="store_true", help="لا تسأل قبل التنزيل")
    args = parser.parse_args()

    ensure_dirs()
    code = {"chbmit": chbmit, "bnci2014_001": bnci, "eegmat": eegmat_dl, "sam40": sam40_dl}[args.dataset](args)
    if code == 0:
        print("تم ✔")
    return code


if __name__ == "__main__":
    sys.exit(main())
