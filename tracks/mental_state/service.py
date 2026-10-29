"""ما يحتاجه الخادم من مسار الحالة الذهنية."""
import bisect
import json
import re
from pathlib import Path

from core.config import MODELS_DIR
from core.evaluation.report import load_report

from .data import CONDITIONS, path_of
from .pipeline import CALIB_SEC

OOF_DIR = MODELS_DIR / "mental_oof"

# كل مقياس: طرفاه، وما تمثّله الحالتان في بياناته
SCALES = {
    "mental_focus": {"title": "مرتاح ↔ مركّز", "low": "مرتاح", "high": "مركّز"},
    # اسمه الصادق: يميّز الاسترخاء من الانشغال بمهمة مجهِدة، لكنه لا يتبع شدة التوتر
    # التي يشعر بها الشخص (انظر rating_agreement في التقرير)
    "mental_stress": {"title": "مسترخٍ ↔ تحت مهمة مجهِدة", "low": "مسترخٍ", "high": "تحت مهمة مجهِدة"},
}


def scales() -> list[dict]:
    out = []
    for scale_id, meta in SCALES.items():
        report = load_report(scale_id)
        if not report:
            continue
        subjects = sorted(p.stem[len(scale_id) + 1:] for p in OOF_DIR.glob(f"{scale_id}_*.json"))
        out.append({
            "id": scale_id, **meta,
            "auc": report.get("auc"),
            "balanced_accuracy": report["summary"]["balanced_accuracy"],
            "subjects_rising": report.get("subjects_rising"),
            "without_calibration": report.get("without_calibration"),
            "rating_agreement": report.get("rating_agreement"),
            "validity_notes": report.get("validity_notes", []),
            "top_features": report.get("top_features", []),
            "dataset": report.get("dataset"),
            "subjects": subjects,
            "per_subject": {r["subject"]: r for r in report.get("per_subject", [])},
        })
    return out


def timeline(scale_id: str, subject: str) -> dict | None:
    if scale_id not in SCALES or not re.fullmatch(r"[\w-]+", subject):
        return None
    path = OOF_DIR / f"{scale_id}_{subject}.json"
    if not path.exists():
        return None
    tl = json.loads(path.read_text(encoding="utf-8"))
    m = re.fullmatch(r"Subject(\d+)", subject)
    if scale_id == "mental_focus" and m:
        # ملفا الراحة والمهمة لفتحهما في البث المباشر
        s = int(m.group(1))
        tl["sources"] = [{"label": label, "source_id": f"edf:eegmat/{path_of(s, c).name}"}
                         for c, label in ((1, "تسجيل الراحة"), (2, "تسجيل الحساب الذهني"))]
    return tl


def live_analyzer(path: Path):
    """للبث المباشر لملفات EEGMAT: قيمة المقياس عند اللحظة الحالية من توقعات
    النموذج الذي لم يرَ هذا الشخص (مع المعايرة)."""
    m = re.fullmatch(r"Subject(\d+)_(\d)\.edf", path.name)
    if not m or int(m.group(2)) not in CONDITIONS:
        return None
    tl = timeline("mental_focus", f"Subject{int(m.group(1)):02d}")
    if not tl or "calibrated" not in tl:
        return None
    # الخط الزمني الموحّد: الراحة من 0، والمهمة بعد نهاية الراحة
    offset = 0.0 if m.group(2) == "1" else next(s["start"] for s in tl["segments"] if s["kind"] == "task")
    times, probs = tl["times"], tl["calibrated"]

    def analyze(session) -> dict | None:
        t = session.source.position_sec + offset
        if t <= CALIB_SEC:
            return {"calibrating": True, "remaining_sec": round(CALIB_SEC - t)}
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return None
        truth = next((s["label"] for s in tl["segments"] if s["start"] <= t < s["end"]), None)
        return {"value": probs[i], "truth": truth, **SCALES["mental_focus"]}

    return analyze
