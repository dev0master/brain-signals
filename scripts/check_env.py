"""فحص البيئة — شغّله بعد التثبيت:

    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.check_env            فحص المكتبات والمجلدات
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.check_env --download  + تنزيل تسجيل صغير للتجربة (~2.5MB)
"""
import argparse
import importlib
import shutil
import sys

from core.config import DATA_DIR, MODELS_DIR, ensure_dirs

LIBS = ["numpy", "scipy", "mne", "sklearn", "joblib", "moabb", "pyriemann", "brainflow",
        "fastapi", "uvicorn", "dotenv", "torch", "braindecode"]
OPTIONAL = {"torch", "braindecode"}  # التعلّم العميق — يُثبَّت لاحقاً


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="نزّل تسجيلاً واحداً من PhysioNet للتأكد أن التنزيل يعمل")
    args = parser.parse_args()

    ok = True
    print(f"Python {sys.version.split()[0]}")
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            print(f"  ✔ {name:12s} {getattr(mod, '__version__', '')}")
        except ImportError:
            mark = "–" if name in OPTIONAL else "✘"
            note = "(اختياري، لمرحلة الكتابة بالدماغ)" if name in OPTIONAL else "(مفقود!)"
            print(f"  {mark} {name:12s} {note}")
            ok = ok and name in OPTIONAL

    ensure_dirs()
    free_gb = shutil.disk_usage(DATA_DIR).free / 1e9
    print(f"\nمجلد البيانات: {DATA_DIR}  (المساحة الحرة: {free_gb:.1f} GB)")
    print(f"مجلد النماذج:  {MODELS_DIR}")
    if free_gb < 8:
        print("  تنبيه: بيانات الصرع تحتاج نحو 5GB")

    if args.download:
        from mne.datasets import eegbci

        # الشخص 1، التسجيل 4: تخيّل حركة اليد اليسرى/اليمنى (64 قناة، 160Hz)
        files = eegbci.load_data(1, [4], path=str(DATA_DIR), update_path=False)
        print(f"\nتم التنزيل: {files[0]}")

    print("\nالنتيجة:", "جاهز ✔" if ok else "توجد مكتبات ناقصة ✘")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
