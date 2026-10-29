"""تدريب كاشف الصرع وتقييمه بصدق.

    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m tracks.epilepsy.train

نقيّم بطريقتين، ولا يُختبر أي نموذج على تسجيل رآه أثناء التدريب:

1) نموذج خاص بكل مريض — "اترك تسجيلاً واحداً":
   لكل ملف من ملفات المريض: ندرّب على بقية ملفاته ونختبر على هذا الملف.
   هكذا يعمل الكاشف حين يتعلم نوبات شخص معيّن ثم يراقبه لاحقاً.

2) نموذج عام — "اترك مريضاً واحداً":
   ندرّب على مريضين ونختبر على كل ملفات الثالث، كأنه شخص جديد تماماً.
   أصعب بكثير، لأن نوبات كل مريض لها شكل مختلف.

المخرجات في models/:
  epilepsy_patient_specific.json   تقرير (1)
  epilepsy_general.json/.joblib    تقرير (2) + النموذج العام مدرَّباً على كل البيانات
  epilepsy_oof/<ملف>.json          توقعات كل ملف من النموذج الذي لم يره (للواجهة)
"""
import json
import sys
import time
from collections import defaultdict

import joblib
import numpy as np

from core.config import MODELS_DIR
from core.evaluation.metrics import Event, classification_summary, combine_event_metrics, event_metrics
from core.evaluation.report import save_report
from core.evaluation.splits import assert_no_leak

from .data import CHANNELS, FS, PATIENTS, Recording, downloaded_recordings
from .pipeline import SETTINGS, cached_features, decide, fit_balanced, timeline

OOF_DIR = MODELS_DIR / "epilepsy_oof"
LABELS = ["طبيعي", "نوبة"]


def fit_predict(train: list[Recording], test: Recording, feats: dict) -> np.ndarray:
    groups = np.array([r.file for r in train] + [test.file])
    assert_no_leak(groups, np.arange(len(train)), np.array([len(train)]))
    X = np.vstack([feats[r.file]["X"] for r in train])
    y = np.concatenate([feats[r.file]["y"] for r in train])
    model = fit_balanced(X, y)
    return model.predict_proba(feats[test.file]["X"])[:, 1]


def evaluate(test: Recording, probs: np.ndarray, feats: dict) -> tuple[dict, dict]:
    f = feats[test.file]
    detected = decide(f["starts"], probs)
    truth = [Event(s, e) for s, e in test.seizures]
    metrics = event_metrics(truth, detected, f["duration"] / 3600)
    return metrics, {**timeline(f["starts"], probs, detected), "metrics": metrics}


def window_summary(parts: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    y = np.concatenate([p[0] for p in parts])
    pred = np.concatenate([p[1] >= SETTINGS["threshold"] for p in parts]).astype(int)
    return classification_summary([LABELS[v] for v in y], [LABELS[v] for v in pred], labels=LABELS)


def pretty(m: dict) -> str:
    lat = f"{m['mean_latency_sec']:.0f}s" if m["mean_latency_sec"] is not None else "—"
    return (f"اكتُشف {m['detected']}/{m['n_seizures']} · إنذارات كاذبة {m['false_alarms']} "
            f"في {m['hours']:.1f} ساعة ({m['false_alarms_per_hour']:.2f}/ساعة) · زمن الاكتشاف {lat}")


def main() -> int:
    recs = downloaded_recordings()
    if not recs:
        print("لا توجد تسجيلات. نزّلها أولاً:  python -m scripts.download_data chbmit")
        return 1
    by_patient: dict[str, list[Recording]] = defaultdict(list)
    for r in recs:
        by_patient[r.patient].append(r)

    print(f"حساب الميزات لـ {len(recs)} تسجيلاً (مرة واحدة ثم تُحفظ)…")
    t0 = time.time()
    feats = {}
    for i, r in enumerate(recs, 1):
        feats[r.file] = cached_features(r)
        sys.stdout.write(f"\r  {i}/{len(recs)} {r.file}")
        sys.stdout.flush()
    print(f"\n  تم في {time.time() - t0:.0f} ثانية")

    oof: dict[str, dict] = {r.file: {"file": r.file, "patient": r.patient,
                                      "duration_sec": feats[r.file]["duration"],
                                      "seizures": [{"start": s, "end": e} for s, e in r.seizures]}
                            for r in recs}

    # ---------- (1) نموذج خاص بكل مريض ----------
    print("\n(1) نموذج خاص بكل مريض — اترك تسجيلاً واحداً")
    ps_metrics, ps_windows, per_patient = [], [], []
    for patient, prs in by_patient.items():
        parts = []
        for test in prs:
            probs = fit_predict([r for r in prs if r is not test], test, feats)
            m, tl = evaluate(test, probs, feats)
            oof[test.file]["patient_specific"] = tl
            parts.append(m)
            ps_windows.append((feats[test.file]["y"], probs))
        agg = combine_event_metrics(parts)
        per_patient.append({"patient": patient, "files": len(prs), **agg})
        ps_metrics.extend(parts)
        print(f"  {patient}: {pretty(agg)}")
    ps_total = combine_event_metrics(ps_metrics)
    print(f"  المجموع: {pretty(ps_total)}")

    # ---------- (2) نموذج عام ----------
    print("\n(2) نموذج عام — اترك مريضاً واحداً (كأنه شخص جديد)")
    gen_metrics, gen_windows, gen_per_patient = [], [], []
    for patient, prs in by_patient.items():
        train = [r for p, rs in by_patient.items() if p != patient for r in rs]
        X = np.vstack([feats[r.file]["X"] for r in train])
        y = np.concatenate([feats[r.file]["y"] for r in train])
        model = fit_balanced(X, y)
        parts = []
        for test in prs:
            probs = model.predict_proba(feats[test.file]["X"])[:, 1]
            m, tl = evaluate(test, probs, feats)
            oof[test.file]["general"] = tl
            parts.append(m)
            gen_windows.append((feats[test.file]["y"], probs))
        agg = combine_event_metrics(parts)
        gen_per_patient.append({"patient": patient, "files": len(prs), **agg})
        gen_metrics.extend(parts)
        print(f"  {patient} (تدريب على {len(by_patient) - 1} مرضى آخرين): {pretty(agg)}")
    gen_total = combine_event_metrics(gen_metrics)
    print(f"  المجموع: {pretty(gen_total)}")

    # ---------- الحفظ ----------
    dataset = (f"CHB-MIT (PhysioNet) — {len(by_patient)} مرضى، {len(recs)} تسجيلاً، "
               f"{ps_total['hours']:.0f} ساعة، {ps_total['n_seizures']} نوبة")
    notes = [
        "الحساسية: نسبة النوبات الحقيقية التي اكتُشفت.",
        "الإنذارات الكاذبة/ساعة: كم مرة أنذر النظام بلا نوبة. المستشفيات تقبل عادة أقل من 1 في الساعة.",
        "زمن الاكتشاف: من بداية النوبة كما حددها الأطباء حتى لحظة الإنذار الفعلي.",
        "نموذج يقول «لا نوبة» دائماً دقته فوق 99% وحساسيته 0% — لذلك لا نعتمد على الدقة.",
    ]
    save_report("epilepsy_patient_specific", {
        "title": "كشف الصرع — نموذج خاص بكل مريض",
        "track": "epilepsy",
        "dataset": dataset,
        "evaluation": "اترك تسجيلاً واحداً: لكل ملف، ندرّب على بقية ملفات المريض نفسه ونختبر عليه.",
        "seizure_events": ps_total,
        "per_patient": per_patient,
        "summary": window_summary(ps_windows),
        "settings": SETTINGS,
        "notes": notes,
    })
    save_report("epilepsy_general", {
        "title": "كشف الصرع — نموذج عام لشخص جديد",
        "track": "epilepsy",
        "dataset": dataset,
        "evaluation": "اترك مريضاً واحداً: ندرّب على المرضى الآخرين ونختبر على كل ملفات هذا المريض.",
        "seizure_events": gen_total,
        "per_patient": gen_per_patient,
        "summary": window_summary(gen_windows),
        "settings": SETTINGS,
        "notes": notes + ["هذا النموذج هو المستخدم لتحليل الملفات التي ترفعها في صفحة فحص الصرع."],
    })

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for file, item in oof.items():
        (OOF_DIR / f"{file}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")

    print("\nتدريب النموذج العام النهائي على كل البيانات…")
    X = np.vstack([feats[r.file]["X"] for r in recs])
    y = np.concatenate([feats[r.file]["y"] for r in recs])
    joblib.dump({"model": fit_balanced(X, y), "train_files": [r.file for r in recs],
                 "channels": CHANNELS, "fs": FS, "settings": SETTINGS},
                MODELS_DIR / "epilepsy_general.joblib")
    print(f"تم ✔  التقارير في {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
