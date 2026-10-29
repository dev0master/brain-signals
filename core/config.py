"""إعدادات المشروع: أين تُحفظ البيانات والنماذج، وتردد الكهرباء المحلي."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

# مجموعات البيانات المنزَّلة (مسار بأحرف إنجليزية فقط)
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / "neuro_data")))

# النماذج المدرَّبة وتقارير تقييمها (JSON) — الواجهة تقرأ التقارير من هنا
MODELS_DIR = PROJECT_DIR / "models"

# تردد شبكة الكهرباء: 50Hz في معظم الدول العربية وأوروبا، 60Hz في أمريكا.
# الكهرباء تُحدث تشويشاً بهذا التردد في الإشارة فنزيله بفلتر notch.
# نحذف الاثنين افتراضياً: تسجيلات الصرع (CHB-MIT) من بوستن (60Hz) وجهازك لاحقاً هنا (50Hz).
MAINS_HZ = tuple(float(x) for x in os.environ.get("MAINS_HZ", "50,60").split(","))

# مكتبات MNE وMOABB تنزّل البيانات إلى هذا المجلد بدل مجلد المستخدم الافتراضي
os.environ.setdefault("MNE_DATA", str(DATA_DIR))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
