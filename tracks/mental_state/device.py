"""الحالة الذهنية بجهازك.

النموذج الأصلي تدرّب على 19 قطباً (EEGMAT) و32 قطباً (SAM40). جهازك له أقطاب أخرى وأقل،
فنعيد تدريب كل مقياس على الأقطاب المشتركة بين جهازك والبيانات فقط، ونعيد تقييمه بنفس
الطريقة (أشخاص لم يرهم النموذج) — فترى بصدق كم خسر حين نقصت الأقطاب.

في القياس المباشر: دقيقة راحة أولاً (خط الأساس الشخصي)، ثم كل ثانيتين قراءة جديدة.
"""
from collections import deque

import joblib
import numpy as np

from core.config import MODELS_DIR
from core.evaluation.report import load_report, save_report
from core.signal.montage import find, key, present

from . import data as eegmat
from . import sam40
from .pipeline import CALIB_SEC, SETTINGS, SETTLE_SEC, STEP_SEC, WIN_SEC, channel_groups, window_features
from .service import SCALES

SMOOTH_WINDOWS = 3  # القراءة المعروضة = متوسط آخر 3 نوافذ (6 ثوانٍ)، كما في صفحة الحالة الذهنية

DATASETS = {
    "mental_focus": {"channels": eegmat.CHANNELS, "name": "EEGMAT", "available": eegmat.available_subjects},
    "mental_stress": {"channels": sam40.CHANNELS, "name": "SAM40", "available": sam40.available_subjects},
}


def model_path(scale_id: str):
    return MODELS_DIR / f"device_{scale_id}.joblib"


def same_channels(a: list[str], b: list[str]) -> bool:
    return [key(x) for x in a] == [key(x) for x in b]


def scale_requirements(scale_id: str, ch_names: list[str]) -> dict:
    """هل يصلح هذا المقياس لأقطاب الجهاز؟ وأي أقطاب سيستخدم؟"""
    ds = DATASETS[scale_id]
    used = present(ch_names, ds["channels"])  # بأسمائها في البيانات وبترتيبها
    groups = channel_groups(used)
    ok = bool(groups["frontal"]) and bool(groups["posterior"])
    path = model_path(scale_id)
    prepared = path.exists() and same_channels(joblib.load(path)["device_channels"], ch_names)
    return {
        "scale": scale_id, "title": SCALES[scale_id]["title"], "dataset": ds["name"],
        "used": used, "n_dataset_channels": len(ds["channels"]), "supported": ok, "prepared": ok and prepared,
        "reason": None if ok else
        f"لا يوجد في جهازك قطب أمامي (مثل Fz أو F3 أو F4) وقطب خلفي (مثل Pz أو O1) مما في بيانات {ds['name']}",
    }


def requirements(ch_names: list[str]) -> dict:
    scales = [scale_requirements(s, ch_names) for s in SCALES]
    supported = [s for s in scales if s["supported"]]
    if not supported:
        status, reason = "unsupported", scales[0]["reason"]
    elif all(s["prepared"] for s in supported):
        status, reason = "ready", None
    else:
        status, reason = "needs_prepare", "يحتاج تدريب المقياس على أقطاب جهازك (زر «جهّز النماذج»)."
    return {"track": "mental_state", "status": status, "reason": reason, "scales": scales}


def _subjects(scale_id: str, used: list[str]):
    from .train import eegmat_subject, sam40_subject

    ids = DATASETS[scale_id]["available"]()
    if scale_id == "mental_focus":
        return [eegmat_subject(s, used) for s in ids]
    rated = sam40.ratings()
    return [sam40_subject(s, rated, used) for s in ids]


def prepare_scale(scale_id: str, ch_names: list[str], log=print) -> dict | None:
    """يدرّب المقياس على أقطاب الجهاز ويقيّمه على أشخاص لم يرهم. يعيد التقرير أو None إن لم يصلح."""
    from .pipeline import make_model
    from .train import (N_FOLDS, cross_subject_predictions, features_of, rating_agreement, summarize,
                        top_features, validity_notes)

    req = scale_requirements(scale_id, ch_names)
    if not req["supported"]:
        log(f"«{req['title']}»: {req['reason']} — تخطّي.")
        return None
    used = req["used"]
    if not DATASETS[scale_id]["available"]():
        log(f"«{req['title']}»: بيانات {req['dataset']} غير منزَّلة — تخطّي.")
        return None
    log(f"«{req['title']}»: حساب الميزات من {req['dataset']} على {len(used)} قطباً مشتركاً مع جهازك…")
    subjects = _subjects(scale_id, used)
    labels = ["راحة", "حساب ذهني"] if scale_id == "mental_focus" else ["استرخاء", "مهمة مجهِدة"]
    probs = cross_subject_predictions(subjects, calibrated=True)
    result = summarize(subjects, probs, labels)
    result["rating_agreement"] = rating_agreement(subjects, probs)
    final = make_model().fit(np.vstack([features_of(s, True) for s in subjects]),
                             np.concatenate([s.y for s in subjects]))
    joblib.dump({"model": final, "channels": used, "device_channels": list(ch_names), "settings": SETTINGS},
                model_path(scale_id))
    full = load_report(scale_id) or {}
    features = top_features(final, used)
    notes = [f"الأقطاب المستخدمة ({len(used)} من {len(DATASETS[scale_id]['channels'])}): {'، '.join(used)}."]
    if full.get("auc") is not None:
        notes.append(f"للمقارنة: النموذج بكل أقطاب {req['dataset']} أعطى AUC {full['auc']:.2f}، "
                     f"وبأقطاب جهازك {result['auc']:.2f}.")
    notes += ["مستوى الصدفة 50% (حالتان). AUC: 0.5 صدفة و1.0 فصل تام.",
              "هذا تقييم على بيانات منشورة بأقطاب في مواضع أقطاب جهازك — لا على جهازك نفسه. جهازك قد يكون أكثر "
              "تشويشاً (أقطاب جافة، حركة)، فاعتبر الرقم حداً أعلى تقريبياً."]
    report = save_report(f"device_{scale_id}", {
        "title": f"{SCALES[scale_id]['title']} — بأقطاب جهازك",
        "track": "mental_state",
        "dataset": f"{req['dataset']} — {len(subjects)} شخصاً، بالأقطاب المشتركة مع جهازك فقط",
        "evaluation": f"اترك أشخاصاً: {len(subjects)} شخصاً في {N_FOLDS} مجموعات، مع معايرة راحة شخصية.",
        "summary": result["summary"], "auc": result["auc"], "subjects_rising": result["subjects_rising"],
        "top_features": features, "validity_notes": validity_notes(result, features),
        "device_channels": list(ch_names), "channels": used, "settings": SETTINGS, "notes": notes,
    })
    log(f"  AUC {result['auc']:.2f} · المقياس يرتفع عند {result['subjects_rising']:.0%} من الأشخاص "
        f"(بكل الأقطاب: {full.get('auc', float('nan')):.2f})")
    return {"report": str(report), "auc": result["auc"]}


def prepare(ch_names: list[str], log=print) -> list[dict]:
    return [r for s in SCALES if (r := prepare_scale(s, ch_names, log))]


# ---------- القياس المباشر ----------

def live_analyzer(ch_names: list[str], fs: float):
    """محلّل مباشر لكل مقياس جاهز لأقطاب هذا الجهاز، أو None إن لم يجهز أي مقياس."""
    models = []
    for scale_id in SCALES:
        path = model_path(scale_id)
        if not path.exists():
            continue
        bundle = joblib.load(path)
        if not same_channels(bundle["device_channels"], ch_names):
            continue
        rows = find(ch_names, bundle["channels"])
        if None in rows:
            continue
        models.append({"id": scale_id, **SCALES[scale_id], "model": bundle["model"], "rows": rows,
                       "channels": bundle["channels"], "baseline": [], "recent": deque(maxlen=SMOOTH_WINDOWS)})
    if not models:
        return None
    win = int(round(WIN_SEC * fs))
    state = {"next_end": SETTLE_SEC + WIN_SEC}  # نهاية النافذة التالية بزمن الجلسة

    def analyze(session) -> dict | None:
        while session.elapsed_sec >= state["next_end"]:
            end = state["next_end"]
            state["next_end"] += STEP_SEC
            lag = int(round((session.elapsed_sec - end) * fs))
            if lag + win > session.buffer.filled:
                continue  # الجلسة تأخرت أكثر من الذاكرة — نتجاوز هذه النافذة
            window = session.buffer.buf[:, session.buffer.buf.shape[1] - lag - win: session.buffer.buf.shape[1] - lag]
            for m in models:
                feats = window_features(window[m["rows"]][None], fs, m["channels"])
                if end <= CALIB_SEC:
                    m["baseline"].append(feats[0])
                else:
                    base = np.asarray(m["baseline"])
                    z = (feats - base.mean(axis=0)) / (base.std(axis=0) + 1e-6)
                    m["recent"].append(float(m["model"].predict_proba(z)[0, 1]))
        if session.elapsed_sec < CALIB_SEC or not models[0]["recent"]:
            return {"calibrating": True, "remaining_sec": round(max(0.0, CALIB_SEC - session.elapsed_sec))}
        return {"scales": [{"id": m["id"], "title": m["title"], "low": m["low"], "high": m["high"],
                            "value": round(float(np.mean(m["recent"])), 3)} for m in models if m["recent"]]}

    return analyze
