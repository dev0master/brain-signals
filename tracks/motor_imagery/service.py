"""ما يحتاجه الخادم من مسار الكتابة بالدماغ."""
import json
from functools import lru_cache

from core.config import MODELS_DIR
from core.evaluation.report import load_report

from . import speller
from .data import CLASS_AR, CLASSES

HELDOUT_DIR = MODELS_DIR / "mi_heldout"
MAX_TARGET_LEN = 40


class SpellerError(ValueError):
    """رسالة مفهومة للمستخدم."""


def info() -> dict:
    return {
        "groups": speller.GROUPS,
        # التوزيع الافتراضي (لوضع «جرّب بنفسك»). لكل شخص توزيعه في subjects()
        "commands": [{"command": speller.COMMAND_OF_CLASS[c], "class": c, "imagine": CLASS_AR[c],
                      "action": speller.COMMAND_AR[speller.COMMAND_OF_CLASS[c]]} for c in CLASSES],
        "class_ar": CLASS_AR,
        "seconds_per_command": speller.SECONDS_PER_COMMAND,
        "initial": speller.view(speller.State()),
    }


def subjects() -> list[dict]:
    report = load_report("motor_imagery") or {}
    rows = {r["subject"]: r for r in report.get("per_subject", [])}
    out = []
    for path in sorted(HELDOUT_DIR.glob("subject_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        s = int(path.stem.split("_")[1])
        r = rows.get(s, {})
        out.append({"subject": s, "accuracy": r.get("test_accuracy"), "pipeline_ar": r.get("pipeline_ar"),
                    "mapping": r.get("mapping"), "spelling": r.get("spelling"),
                    "spelling_default_mapping": r.get("spelling_default_mapping")})
    return out


@lru_cache(maxsize=16)
def _heldout(subject: int, mtime: float) -> dict:  # mtime: يُعاد التحميل إن أُعيد التدريب
    return json.loads((HELDOUT_DIR / f"subject_{subject}.json").read_text(encoding="utf-8"))


def simulate(subject: int, target: str, seed: int = 0, min_confidence: float = 0.0) -> dict:
    path = HELDOUT_DIR / f"subject_{subject}.json"
    if not path.exists():
        raise SpellerError("لا توجد نتائج لهذا الشخص — شغّل التدريب: python -m tracks.motor_imagery.train")
    target = target.strip()
    if not target:
        raise SpellerError("اكتب الجملة التي تريد أن يكتبها المستخدم")
    if len(target) > MAX_TARGET_LEN:
        raise SpellerError(f"الجملة طويلة — الحد {MAX_TARGET_LEN} حرفاً (كل حرف يأخذ نحو دقيقة)")
    try:
        result = speller.simulate(target, _heldout(subject, path.stat().st_mtime), seed=seed,
                                  min_confidence=min_confidence)
    except ValueError as exc:
        raise SpellerError(str(exc)) from exc
    return {"subject": subject, **result}


def step(state: dict, command: str) -> dict:
    if command not in speller.COMMANDS:
        raise SpellerError(f"أمر غير معروف: {command}")
    s = speller.State(text=state.get("text", ""), group=state.get("group"), cursor=int(state.get("cursor", 0)))
    return speller.view(speller.apply(s, command))
