"""تجهيز النماذج لأقطاب جهازك — نفس زر «جهّز النماذج» في صفحة «جهازك».

    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m scripts.prepare_device

يقرأ الإعداد المحفوظ (DATA_DIR/device.json)، ثم يعيد تدريب مقياسَي الحالة الذهنية ونموذج الصرع العام
على الأقطاب المشتركة بين جهازك والبيانات العامة، ويقيّمها على أشخاص لم يرهم النموذج.
يأخذ بضع دقائق. الكتابة بالدماغ لا تُجهَّز هنا: تحتاج جلسة معايرة لك أنت.
"""
import sys

from server.device import load_config, run_prepare


def main() -> int:
    cfg = load_config()
    if cfg is None:
        print("لم يُعدّ جهاز بعد. افتح صفحة «جهازك» في الواجهة واحفظ الإعداد أولاً.")
        return 1
    run_prepare(cfg["ch_names"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
