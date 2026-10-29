"""تدريب مصنّف تخيّل الحركة لكل شخص وتقييمه بصدق.

    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m tracks.motor_imagery.train

لكل شخص:
  1. نجرّب ثلاث طرق على جلسة اليوم الأول فقط (تقسيم داخلي إلى 5 أجزاء):
       csp_lda       : CSP يجد توليفات الأقطاب التي تختلف قدرتها أكثر بين الأوامر، ثم مصنّف خطي
       riemann_ts_lr : مصفوفة التغاير بين الأقطاب كنقطة في فضاء منحنٍ ← "تسطيحها" ← انحدار لوجستي
       riemann_mdm   : أقرب "متوسط" لكل أمر في ذلك الفضاء
  2. نختار الأفضل بحسب اليوم الأول — لا ننظر إلى اليوم الثاني أبداً عند الاختيار
     (اختيار الطريقة بنتيجة الاختبار نفسه تسريب خفيّ يضخّم الأرقام).
  3. ندرّب الطريقة المختارة على اليوم الأول كاملاً ونختبرها على اليوم الثاني.
     هكذا يُستخدم النظام فعلاً: معايرة اليوم، واستخدام في يوم آخر.

المخرجات في models/:
  motor_imagery.json                التقرير
  mi_heldout/subject_<n>.json       ما فهمه النموذج من كل محاولة في اليوم الثاني (للمستخدم المحاكى)
  mi_subject_<n>.joblib             النموذج (لجهاز حقيقي لاحقاً)
"""
import json
import sys
import time

import joblib
import numpy as np
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline

from core.config import MODELS_DIR
from core.evaluation.metrics import classification_summary
from core.evaluation.report import save_report
from core.evaluation.splits import assert_no_leak

from .data import BAND, CLASS_AR, CLASSES, EPOCH_SEC, SESSIONS, SUBJECTS, load_subject
from .speller import COMMAND_AR, MAPPING_PHRASES, choose_mapping, typical_performance

BENCHMARK_PHRASE = "أحتاج ماء"  # جملة ثابتة نقيس بها سرعة الكتابة لكل شخص

HELDOUT_DIR = MODELS_DIR / "mi_heldout"

PIPELINES = {
    "csp_lda": lambda: make_pipeline(CSP(n_components=8, reg="ledoit_wolf", log=True), LinearDiscriminantAnalysis()),
    "riemann_ts_lr": lambda: make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                                           LogisticRegression(C=1.0, max_iter=2000)),
    "riemann_mdm": lambda: make_pipeline(Covariances("oas"), MDM(metric="riemann")),
}
PIPELINE_AR = {
    "csp_lda": "CSP + مصنّف خطي",
    "riemann_ts_lr": "ريماني: فضاء مماسي + انحدار لوجستي",
    "riemann_mdm": "ريماني: أقرب متوسط",
}


INNER_CV = StratifiedKFold(5, shuffle=True, random_state=0)


def choose_pipeline(X: np.ndarray, y: np.ndarray) -> tuple[str, dict[str, float]]:
    """أفضل طريقة بحسب تقسيم داخلي لبيانات التدريب وحدها."""
    scores = {name: float(cross_val_score(make(), X, y, cv=INNER_CV).mean()) for name, make in PIPELINES.items()}
    return max(scores, key=scores.get), scores


def day_one_heldout(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """توقعات اليوم الأول لمحاولات لم يرها النموذج (كل جزء من الخمسة يُتوقَّع بنموذج
    تدرّب على الأربعة الباقية). نختار بها توزيع الأوامر دون لمس اليوم الثاني."""
    probs = cross_val_predict(PIPELINES[name](), X, y, cv=INNER_CV, method="predict_proba")
    return {"y_true": y.tolist(), "probs": probs.tolist()}


def recall_of(heldout: dict) -> dict[str, float]:
    y = np.asarray(heldout["y_true"])
    pred = np.asarray(heldout["probs"]).argmax(axis=1)
    return {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)}


def train_subject(subject: int) -> dict:
    d = load_subject(subject)
    X = d["X"].astype(np.float64)
    y, session = d["y"], d["session"]
    train_idx = np.flatnonzero(session == SESSIONS[0])
    test_idx = np.flatnonzero(session == SESSIONS[1])
    assert_no_leak(session, train_idx, test_idx)

    name, inner = choose_pipeline(X[train_idx], y[train_idx])
    day1 = day_one_heldout(name, X[train_idx], y[train_idx])
    mapping, mapping_search = choose_mapping(day1)
    model = PIPELINES[name]().fit(X[train_idx], y[train_idx])
    probs = model.predict_proba(X[test_idx])
    pred = probs.argmax(axis=1)

    heldout = {"subject": subject, "classes": CLASSES, "pipeline": name, "mapping": mapping,
               "y_true": y[test_idx].tolist(), "probs": np.round(probs, 4).tolist()}
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    (HELDOUT_DIR / f"subject_{subject}.json").write_text(json.dumps(heldout), encoding="utf-8")
    joblib.dump({"model": model, "pipeline": name, "classes": CLASSES, "fs": d["fs"],
                 "ch_names": d["ch_names"], "epoch_sec": EPOCH_SEC, "band": BAND},
                MODELS_DIR / f"mi_subject_{subject}.joblib")

    summary = classification_summary([CLASS_AR[CLASSES[i]] for i in y[test_idx]],
                                     [CLASS_AR[CLASSES[i]] for i in pred],
                                     labels=[CLASS_AR[c] for c in CLASSES])
    return {"subject": subject, "pipeline": name, "pipeline_ar": PIPELINE_AR[name],
            "inner_cv": inner, "test_accuracy": summary["accuracy"], "kappa": summary["kappa"],
            "n_train": len(train_idx), "n_test": len(test_idx),
            "day1_recall": recall_of(day1), "day2_recall": recall_of(heldout),
            "mapping": mapping, "mapping_search_day1": mapping_search,
            "spelling": typical_performance(BENCHMARK_PHRASE, heldout),
            # للمقارنة: نفس النموذج بالتوزيع الافتراضي (القدمان = اختيار)
            "spelling_default_mapping": typical_performance(BENCHMARK_PHRASE, {**heldout, "mapping": None}),
            "_y": y[test_idx], "_pred": pred}


def main() -> int:
    rows = []
    for subject in SUBJECTS:
        t0 = time.time()
        r = train_subject(subject)
        rows.append(r)
        scores = " · ".join(f"{k} {v:.0%}" for k, v in r["inner_cv"].items())
        sp, sd = r["spelling"], r["spelling_default_mapping"]
        speed = f"{sp['median_chars_per_minute']:.1f} حرف/دقيقة" if sp["median_chars_per_minute"] else "لم تكتمل"
        before = f"{sd['median_chars_per_minute']:.1f}" if sd["median_chars_per_minute"] else "لم تكتمل"
        mapping = "، ".join(f"{COMMAND_AR[c]}={CLASS_AR[k]}" for c, k in r["mapping"].items())
        print(f"الشخص {subject}: اليوم الأول [{scores}] ← {r['pipeline']} ← اليوم الثاني "
              f"{r['test_accuracy']:.0%} (kappa {r['kappa']:.2f})\n    توزيع الأوامر: {mapping}\n"
              f"    كتابة «{BENCHMARK_PHRASE}»: {speed}، اكتملت {sp['completion_rate']:.0%} "
              f"(بالتوزيع الافتراضي: {before}، {sd['completion_rate']:.0%}) [{time.time() - t0:.0f}ث]", flush=True)

    y = np.concatenate([r.pop("_y") for r in rows])
    pred = np.concatenate([r.pop("_pred") for r in rows])
    labels = [CLASS_AR[c] for c in CLASSES]
    summary = classification_summary([labels[i] for i in y], [labels[i] for i in pred], labels=labels)
    mean_acc = float(np.mean([r["test_accuracy"] for r in rows]))
    print(f"\nالمتوسط على اليوم الثاني: {mean_acc:.0%} (الصدفة 25%)")

    save_report("motor_imagery", {
        "title": "الكتابة بالدماغ — تمييز 4 أوامر متخيَّلة",
        "track": "motor_imagery",
        "dataset": f"BNCI 2014-001 (BCI Competition IV 2a) — {len(SUBJECTS)} أشخاص، جلستان في يومين، 288 محاولة لكل جلسة",
        "evaluation": "الطريقة تُختار وتُدرَّب على اليوم الأول فقط، وتُختبر على اليوم الثاني الذي لم تره.",
        "summary": summary,
        "per_subject": rows,
        "mean_subject_accuracy": mean_acc,
        "settings": {"window_after_cue_sec": list(EPOCH_SEC), "band_hz": list(BAND),
                     "pipelines": PIPELINE_AR},
        "notes": [
            "مستوى الصدفة 25%: أربعة أوامر متساوية العدد.",
            "يتفاوت الناس كثيراً: بعضهم يُنتج إشارة تخيّل واضحة وبعضهم بالكاد — ظاهرة معروفة في أبحاث BCI.",
            "الدقة هنا لكل محاولة منفردة مدتها ثانيتان. في لوحة الحروف يصحح المستخدم الأخطاء بأمر «رجوع».",
            "توزيع الأوامر: مرشّحان — الافتراضي، أو «الأمر الأهم للتخيّل الأدق». نقارنهما بمحاكاة الكتابة على "
            f"توقعات اليوم الأول فقط وبجمل مختلفة عن جملة التقييم ({'، '.join(MAPPING_PHRASES)})، ولا نترك "
            "الافتراضي إلا بفارق واضح. تجربة التوزيعات الـ24 كلها فشلت: الفائز في اليوم الأول كان غالباً "
            "المحظوظ لا الأفضل (لعنة الفائز). نعرض أيضاً السرعة بالتوزيع الافتراضي للمقارنة.",
            f"سرعة الكتابة: وسيط 20 محاكاة لكتابة «{BENCHMARK_PHRASE}» بمحاولات حقيقية من اليوم الثاني، "
            "بافتراض 6 ثوانٍ لكل أمر. المحاكاة تتوقف بعد 300 أمر (30 دقيقة) إن لم تكتمل.",
            "تنبيه صدق: أثناء التطوير جرّبنا ثلاث طرق لاختيار التوزيع ورأينا نتائجها على اليوم الثاني قبل أن "
            "نستقر على هذه. لذلك قد تكون أرقام الكتابة متفائلة قليلاً؛ التقييم النظيف تماماً يحتاج جلسة ثالثة أو أشخاصاً جدداً.",
            "بعض المشكلات لا تظهر إلا في اليوم الثاني (إشارة الدماغ تتغير بين يوم وآخر)، فلا يمكن توقعها من اليوم الأول. "
            "الأنظمة الحقيقية تعالج هذا بمعايرة قصيرة في بداية كل جلسة.",
        ],
    })
    print(f"تم ✔  التقرير في {MODELS_DIR / 'motor_imagery.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
