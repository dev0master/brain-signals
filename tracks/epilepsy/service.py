"""ما يحتاجه الخادم من مسار الصرع: قائمة التسجيلات، خطوطها الزمنية، تحليل ملف مرفوع،
ومحلّل للبث المباشر."""
import bisect
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from core.config import DATA_DIR, MODELS_DIR
from core.evaluation.report import load_report

from .data import FS, downloaded_recordings, select_channels
from .pipeline import decide, recording_features, timeline

OOF_DIR = MODELS_DIR / "epilepsy_oof"
GENERAL_MODEL = MODELS_DIR / "epilepsy_general.joblib"


class AnalysisError(ValueError):
    """رسالة خطأ مفهومة للمستخدم (ملف غير مناسب مثلاً)."""


def _source_id(path: Path) -> str:
    """معرّف المصدر الذي يفتح هذا الملف في البث المباشر."""
    return f"edf:{path.relative_to(DATA_DIR).as_posix()}"


def list_recordings() -> list[dict]:
    out = []
    for r in downloaded_recordings():
        out.append({"file": r.file, "patient": r.patient, "n_seizures": len(r.seizures),
                    "analyzed": (OOF_DIR / f"{r.file}.json").exists(),
                    "source_id": _source_id(r.path)})
    return out


def recording_timeline(file: str) -> dict | None:
    """توقعات ملف من النماذج التي لم تره أثناء التدريب (من train.py)."""
    path = OOF_DIR / f"{Path(file).name}.json"
    if not path.exists():
        return None
    tl = json.loads(path.read_text(encoding="utf-8"))
    rec = next((r for r in downloaded_recordings() if r.file == tl["file"]), None)
    tl["source_id"] = _source_id(rec.path) if rec else None
    return tl


@lru_cache(maxsize=1)
def _general_bundle(mtime: float) -> dict:  # mtime في المفتاح: يُعاد التحميل إن أُعيد التدريب
    return joblib.load(GENERAL_MODEL)


def analyze_edf(path: Path, original_name: str) -> dict:
    """يحلّل ملف EDF مرفوعاً بالنموذج العام."""
    import mne

    if not GENERAL_MODEL.exists():
        raise AnalysisError("النموذج غير مدرَّب بعد. شغّل: python -m tracks.epilepsy.train")
    bundle = _general_bundle(GENERAL_MODEL.stat().st_mtime)
    try:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="error")
    except Exception as exc:
        raise AnalysisError(f"تعذّرت قراءة الملف كـ EDF: {exc}") from exc
    idx = select_channels(raw.ch_names)
    if idx is None:
        raise AnalysisError(
            "الملف لا يحتوي الأقطاب الثنائية الـ18 التي تدرّب عليها النموذج "
            "(FP1-F7، F7-T7، ... CZ-PZ). النموذج لا يصلح لتركيب أقطاب مختلف.")
    raw.pick(idx).load_data(verbose="error")  # نقرأ القنوات الـ18 فقط من القرص
    if float(raw.info["sfreq"]) != FS:
        raw.resample(FS, verbose="error")  # النموذج تدرّب على 256 عيّنة/ثانية
    data = (raw.get_data() * 1e6).astype(np.float32)
    X, starts = recording_features(data, FS)
    probs = bundle["model"].predict_proba(X)[:, 1]
    detected = decide(starts, probs)
    warnings = []
    # موثوقية النموذج العام كما قيست فعلاً على أشخاص لم يرهم (من تقرير التدريب)
    report = load_report("epilepsy_general")
    if report and (ev := report.get("seizure_events")):
        warnings.append(
            f"عند اختباره على أشخاص لم يرهم، اكتشف هذا النموذج {ev['detected']} من {ev['n_seizures']} نوبة "
            f"مع {ev['false_alarms_per_hour']:.1f} إنذار كاذب في الساعة — نتيجته على ملفك ضعيفة الموثوقية. "
            "السبب: تدرّب على 3 أطفال فقط، ونوبات كل شخص تختلف عن غيره.")
    if Path(original_name).name in bundle["train_files"]:
        warnings.append("هذا الملف كان ضمن بيانات تدريب النموذج العام، فالنتيجة متفائلة أكثر من الواقع. "
                        "لنتيجة صادقة اختره من قائمة التسجيلات المحفوظة.")
    return {
        "file": original_name,
        "duration_sec": data.shape[1] / FS,
        "seizures": [],  # لا نعرف الإجابة الصحيحة لملف مرفوع
        "general": timeline(starts, probs, detected),
        "warnings": warnings,
    }


def live_analyzer(path: Path):
    """محلّل للبث المباشر: يقرأ احتمال النوبة عند اللحظة الحالية من توقعات النموذج
    الذي لم يرَ هذا الملف (نفس ما تعرضه صفحة فحص الصرع)."""
    tl = recording_timeline(path.name)
    if not tl or "patient_specific" not in tl:
        return None
    model = tl["patient_specific"]
    times, probs = model["times"], model["probs"]
    events = model["detected"]

    def analyze(session) -> dict | None:
        t = session.source.position_sec
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return None
        alarm = any(e.get("alarm", e["start"]) <= t <= e["end"] for e in events)
        return {"prob": probs[i], "alarm": alarm, "threshold": model["threshold"]}

    return analyze
