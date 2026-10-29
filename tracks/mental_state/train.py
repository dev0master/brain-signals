"""تدريب مقاييس الحالة الذهنية وتقييمها على أشخاص لم يرهم النموذج.

    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m tracks.mental_state.train

المقياس الأول «مرتاح ↔ مركّز» من EEGMAT (راحة مقابل حساب ذهني).

التقييم: نقسم الأشخاص إلى 6 مجموعات؛ في كل مرة ندرّب على 5 ونختبر على السادسة.
فكل رقم في التقرير لشخص لم يرَ النموذج أي ثانية من إشارته.

نقارن طريقتين:
  بلا معايرة  : النموذج يحكم على إشارة الشخص مباشرة
  مع معايرة   : أول دقيقة راحة = خط أساس الشخص، والحكم على الانحراف عنه
               (تلك الدقيقة لا تدخل في الاختبار)
قررنا قبل رؤية النتائج أن الواجهة تستخدم «مع معايرة» لأنها طريقة الاستخدام الواقعية؛
ونعرض الطريقتين في التقرير.
"""
import json
import sys
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from core.config import MODELS_DIR
from core.evaluation.metrics import classification_summary
from core.evaluation.report import save_report
from core.evaluation.splits import assert_no_leak

from . import data as eegmat
from . import sam40
from .pipeline import CALIB_SEC, SETTINGS, WIN_SEC, calibrate, feature_names, make_model, recording_features

OOF_DIR = MODELS_DIR / "mental_oof"
N_FOLDS = 6


@dataclass
class SubjectData:
    """نوافذ شخص واحد على خط زمني موحّد (الحالة الهادئة أولاً ثم المهمة)."""
    subject: str
    X: np.ndarray          # نوافذ الاختبار (بدون دقيقة المعايرة)
    y: np.ndarray          # 0 = الحالة الهادئة، 1 = المهمة
    t: np.ndarray          # زمن نهاية كل نافذة على الخط الموحّد (ثانية)
    baseline: np.ndarray   # نوافذ دقيقة المعايرة
    segments: list[dict] = field(default_factory=list)
    # للتحقق من مقياس التوتر: لكل نافذة مهمة، التوتر الذي قيّمه صاحبها لتلك المحاولة (وإلا nan)
    rating: np.ndarray | None = None
    trial: np.ndarray | None = None   # معرّف المحاولة لكل نافذة (لتجميع النوافذ)


def eegmat_subject(s: int) -> SubjectData:
    rest, fs = eegmat.load_recording(s, 1)
    task, _ = eegmat.load_recording(s, 2)
    Xr, tr = recording_features(rest, fs)
    Xt, tt = recording_features(task, fs)
    rest_dur, task_dur = rest.shape[1] / fs, task.shape[1] / fs
    end_r = tr + WIN_SEC               # زمن نهاية كل نافذة
    calib = end_r <= CALIB_SEC
    return SubjectData(
        subject=f"Subject{s:02d}",
        X=np.vstack([Xr[~calib], Xt]),
        y=np.concatenate([np.zeros((~calib).sum(), int), np.ones(len(Xt), int)]),
        t=np.concatenate([end_r[~calib], rest_dur + tt + WIN_SEC]),
        baseline=Xr[calib],
        segments=[
            {"kind": "calibration", "label": "معايرة (راحة)", "start": 0.0, "end": CALIB_SEC},
            {"kind": "calm", "label": "راحة", "start": CALIB_SEC, "end": rest_dur},
            {"kind": "task", "label": "حساب ذهني", "start": rest_dur, "end": rest_dur + task_dur},
        ],
    )


def sam40_subject(s: int, ratings: dict) -> SubjectData:
    """محاولات SAM40 منفصلة (25ث لكل منها)، نرصّها على خط زمني للعرض:
    استرخاء 1 = معايرة، استرخاء 2–3 = هادئ، ثم محاولات المهام الثلاث.
    (هذا ترتيب للعرض فقط، لا ترتيب التسجيل الفعلي.)"""
    trial_sec = sam40.load_trial("Relax", s, 1).shape[1] / sam40.FS
    X, y, t, rating, trial = [], [], [], [], []
    segments = []
    clock = 0.0
    baseline = None
    plan = [("Relax", 1, "calibration", "معايرة (استرخاء)")]
    plan += [("Relax", k, "calm", "استرخاء") for k in (2, 3)]
    plan += [(task, k, "task", f"{label} {k}") for task, label in sam40.TASKS.items() for k in sam40.TRIALS]
    for kind, k, seg_kind, label in plan:
        feats, starts = recording_features(sam40.load_trial(kind, s, k), sam40.FS, sam40.CHANNELS)
        segments.append({"kind": seg_kind, "label": label, "start": clock, "end": clock + trial_sec})
        if seg_kind == "calibration":
            baseline = feats
        else:
            X.append(feats)
            y.append(np.full(len(feats), int(seg_kind == "task")))
            t.append(clock + starts + WIN_SEC)
            rating.append(np.full(len(feats), ratings.get((s, kind, k), np.nan)))
            trial.append(np.full(len(feats), f"{kind}/{k}"))
        clock += trial_sec
    return SubjectData(subject=f"S{s:02d}", X=np.vstack(X), y=np.concatenate(y), t=np.concatenate(t),
                       baseline=baseline, segments=segments,
                       rating=np.concatenate(rating), trial=np.concatenate(trial))


def rating_agreement(subjects: list[SubjectData], probs: list[np.ndarray]) -> dict | None:
    """هل يرتفع المقياس في محاولات المهام التي قال أصحابها إنها أكثر توتراً؟

    نأخذ لكل محاولة مهمة متوسط قيمة المقياس، ونقارنه بتقييم التوتر الذاتي:
      - عبر كل المحاولات (يخلط الفروق بين الناس)
      - داخل كل شخص ثم المتوسط (هل تتبع قراءة الشخص تقييماته هو؟)
    """
    from scipy.stats import spearmanr

    pooled_p, pooled_r, within = [], [], []
    for s, p in zip(subjects, probs):
        if s.rating is None:
            continue
        mp, mr = [], []
        for tr in np.unique(s.trial[s.y == 1]):
            mask = s.trial == tr
            if not np.isnan(s.rating[mask][0]):
                mp.append(p[mask].mean())
                mr.append(s.rating[mask][0])
        pooled_p += mp
        pooled_r += mr
        if len(set(mr)) > 1:
            within.append(spearmanr(mp, mr).statistic)
    if not pooled_p:
        return None
    rho, pval = spearmanr(pooled_p, pooled_r)
    return {"spearman_pooled": float(rho), "p_value": float(pval), "n_trials": len(pooled_p),
            "mean_within_subject": float(np.nanmean(within)), "n_subjects": len(within)}


def features_of(sd: SubjectData, calibrated: bool) -> np.ndarray:
    return calibrate(sd.X, sd.baseline) if calibrated else sd.X


def cross_subject_predictions(subjects: list[SubjectData], calibrated: bool) -> list[np.ndarray]:
    """احتمال «مهمة» لكل نافذة، من نموذج لم يتدرّب على صاحبها."""
    groups = np.arange(len(subjects))
    probs: list[np.ndarray | None] = [None] * len(subjects)
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(groups, groups=groups):
        assert_no_leak(groups, train_idx, test_idx)
        X = np.vstack([features_of(subjects[i], calibrated) for i in train_idx])
        y = np.concatenate([subjects[i].y for i in train_idx])
        model = make_model().fit(X, y)
        for i in test_idx:
            probs[i] = model.predict_proba(features_of(subjects[i], calibrated))[:, 1]
    return probs  # type: ignore[return-value]


def summarize(subjects: list[SubjectData], probs: list[np.ndarray], labels: list[str]) -> dict:
    y = np.concatenate([s.y for s in subjects])
    p = np.concatenate(probs)
    per_subject = []
    for s, ps in zip(subjects, probs):
        calm, task = ps[s.y == 0].mean(), ps[s.y == 1].mean()
        per_subject.append({"subject": s.subject, "calm_mean": float(calm), "task_mean": float(task),
                            "auc": float(roc_auc_score(s.y, ps))})
    return {
        "summary": classification_summary([labels[v] for v in y], [labels[int(v >= 0.5)] for v in p], labels=labels),
        "auc": float(roc_auc_score(y, p)),
        # هل يرتفع المقياس فعلاً عند هذا الشخص حين يبدأ المهمة؟
        "subjects_rising": float(np.mean([r["task_mean"] > r["calm_mean"] for r in per_subject])),
        "per_subject": per_subject,
    }


def top_features(model, channels: list[str], n: int = 8) -> list[dict]:
    """أكثر الميزات تأثيراً في قرار النموذج النهائي، مع اتجاهها."""
    coef = model[-1].coef_[0]
    names = feature_names(channels)
    order = np.argsort(-np.abs(coef))[:n]
    return [{"feature": names[i], "weight": round(float(coef[i]), 3),
             "direction": "ترتفع مع المهمة" if coef[i] > 0 else "تنخفض مع المهمة"} for i in order]


EYE_CHANNELS = {"Fp1", "Fp2"}


def validity_notes(result: dict, features: list[dict]) -> list[str]:
    """ملاحظات صدق تُكتب من الأرقام نفسها لا باليد."""
    notes = []
    top = features[:3]
    eye = [f["feature"] for f in top if f["feature"].split(" عند ")[-1] in EYE_CHANNELS]
    if eye:
        notes.append(f"من أقوى 3 ميزات: {'، '.join(eye)} — أقطاب الجبهة هذه الأكثر تأثراً بحركة العين والرمش، "
                     "فقد يقرأ النموذج جزئياً سلوك العينين أثناء المهمة لا نشاط الدماغ وحده.")
    ag = result.get("rating_agreement")
    if ag:
        if abs(ag["mean_within_subject"]) < 0.2:
            verdict = ("المقياس لا يتبع شدة التوتر التي شعر بها الأشخاص؛ هو يميّز الاسترخاء من الانشغال بمهمة، "
                       "لا «كم أنت متوتر».")
        elif ag["mean_within_subject"] > 0:
            verdict = "المقياس يميل إلى الارتفاع في المحاولات التي شعر فيها أصحابها بتوتر أكبر."
        else:
            verdict = "المقياس يتحرك عكس التوتر الذي شعر به الأشخاص — لا يصلح مقياساً للتوتر."
        notes.append(f"توافقه مع التوتر الذي قيّمه كل شخص (1–10) لكل محاولة: سبيرمان {ag['spearman_pooled']:+.2f} "
                     f"عبر {ag['n_trials']} محاولة، و{ag['mean_within_subject']:+.2f} داخل الشخص الواحد. {verdict}")
    return notes


def run_scale(scale_id: str, title: str, dataset: str, subjects: list[SubjectData],
              labels: list[str], extra_notes: list[str], channels: list[str] = eegmat.CHANNELS,
              calibration_text: str = "أول دقيقة راحة خط أساس شخصي ولا تدخل في الاختبار.") -> dict:
    results = {}
    oof = {}
    for mode, calibrated in (("calibrated", True), ("raw", False)):
        probs = cross_subject_predictions(subjects, calibrated)
        results[mode] = summarize(subjects, probs, labels)
        for s, p in zip(subjects, probs):
            oof.setdefault(s.subject, {"subject": s.subject, "segments": s.segments, "times": np.round(s.t, 1).tolist()})
            oof[s.subject][mode] = np.round(p, 3).tolist()
        r = results[mode]
        r["rating_agreement"] = rating_agreement(subjects, probs)
        print(f"  {'مع معايرة' if calibrated else 'بلا معايرة'}: دقة متوازنة {r['summary']['balanced_accuracy']:.0%} · "
              f"AUC {r['auc']:.2f} · المقياس يرتفع عند {r['subjects_rising']:.0%} من الأشخاص", flush=True)
        if (ag := r["rating_agreement"]) is not None:
            print(f"    توافق مع التوتر الذي قيّمه الأشخاص: سبيرمان {ag['spearman_pooled']:+.2f} "
                  f"(p={ag['p_value']:.3f}) عبر {ag['n_trials']} محاولة · داخل الشخص {ag['mean_within_subject']:+.2f}",
                  flush=True)

    final = make_model().fit(np.vstack([features_of(s, True) for s in subjects]),
                             np.concatenate([s.y for s in subjects]))
    joblib.dump({"model": final, "feature_names": feature_names(channels), "channels": channels, "settings": SETTINGS},
                MODELS_DIR / f"{scale_id}.joblib")
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for name, item in oof.items():
        (OOF_DIR / f"{scale_id}_{name}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")

    cal, raw = results["calibrated"], results["raw"]
    save_report(scale_id, {
        "title": title,
        "track": "mental_state",
        "dataset": dataset,
        "evaluation": f"اترك أشخاصاً: {len(subjects)} شخصاً في {N_FOLDS} مجموعات — كل شخص يُختبر بنموذج لم يرَ إشارته. "
                      f"مع معايرة: {calibration_text}",
        "summary": cal["summary"],
        "auc": cal["auc"],
        "subjects_rising": cal["subjects_rising"],
        "without_calibration": {k: raw[k] for k in ("auc", "subjects_rising")} | {
            "balanced_accuracy": raw["summary"]["balanced_accuracy"]},
        "per_subject": cal["per_subject"],
        "rating_agreement": cal["rating_agreement"],
        "top_features": top_features(final, channels),
        "validity_notes": validity_notes(cal, top_features(final, channels)),
        "settings": SETTINGS,
        "notes": [
            "مستوى الصدفة 50% (حالتان). AUC: احتمال أن تحصل نافذة من المهمة على قيمة أعلى من نافذة راحة "
            "— 0.5 صدفة و1.0 فصل تام.",
            "المقياس احتمال وليس قراءة للأفكار: يقول «إشارتك تشبه إشارة من ينشغل بمهمة ذهنية أكثر مما تشبه الراحة».",
        ] + extra_notes,
    })
    return results


def main() -> int:
    subjects_ids = eegmat.available_subjects()
    if not subjects_ids:
        print("لا توجد بيانات. نزّلها: python -m scripts.download_data eegmat")
        return 1
    print(f"«مرتاح ↔ مركّز» — EEGMAT، {len(subjects_ids)} شخصاً", flush=True)
    subjects = [eegmat_subject(s) for s in subjects_ids]
    run_scale(
        "mental_focus", "الحالة الذهنية — مرتاح ↔ مركّز",
        f"EEGMAT (PhysioNet) — {len(subjects)} شخصاً، 3 دقائق راحة ثم دقيقة حساب ذهني لكل شخص، 19 قطباً",
        subjects, labels=["راحة", "حساب ذهني"],
        extra_notes=[
            "الراحة والحساب تسجيلان منفصلان بترتيب ثابت؛ جزء من الفرق قد يأتي من التسجيل نفسه لا من الحالة الذهنية.",
            "البيانات مفلترة عند 30Hz قبل نشرها، فلم نستخدم نطاق جاما.",
        ],
    )
    if (ids := sam40.available_subjects()):
        print(f"\n«مسترخٍ ↔ تحت مهمة مجهِدة» — SAM40، {len(ids)} شخصاً", flush=True)
        rated = sam40.ratings()
        run_scale(
            "mental_stress", "الحالة الذهنية — مسترخٍ ↔ تحت مهمة مجهِدة",
            f"SAM40 (figshare، CC BY 4.0) — {len(ids)} شخصاً، 3 محاولات استرخاء و9 محاولات مهام مجهِدة "
            "(ستروب، حساب ذهني، صورة المرآة) لكل شخص، 25 ثانية لكل محاولة، 32 قطباً",
            [sam40_subject(s, rated) for s in ids], labels=["استرخاء", "مهمة مجهِدة"],
            channels=sam40.CHANNELS,
            calibration_text="محاولة الاسترخاء الأولى (25ث) خط أساس شخصي ولا تدخل في الاختبار.",
            extra_notes=[
                "«متوتر» هنا يعني: أثناء مهمة صُمّمت لإحداث التوتر. قد يقيس المقياس الانشغال الذهني لا التوتر بالذات، "
                "لذلك نقارنه بتقييم التوتر الذي أعطاه كل شخص لكل محاولة (rating_agreement).",
                "محاولات SAM40 منفصلة؛ ترتيبها على الخط الزمني في الواجهة للعرض فقط.",
            ],
        )
    else:
        print("\nSAM40 غير موجودة — تخطّي «مسترخٍ ↔ تحت مهمة مجهِدة». نزّلها: python -m scripts.download_data sam40")
    print(f"تم ✔  التقارير في {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
