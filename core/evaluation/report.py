"""تقارير التقييم.

كل نموذج يُحفظ معه تقرير JSON فيه نتائجه الفعلية وطريقة تقييمه. صفحة "النماذج"
تعرض هذه الملفات كما هي — لا تُكتب أي أرقام في الواجهة يدوياً.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import MODELS_DIR


def save_report(model_id: str, report: dict, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    report = {"model_id": model_id,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **report}
    path = models_dir / f"{model_id}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_reports(models_dir: Path = MODELS_DIR) -> list[dict]:
    if not models_dir.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(models_dir.glob("*.json"))]


def load_report(model_id: str, models_dir: Path = MODELS_DIR) -> dict | None:
    path = models_dir / f"{model_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
