"""فحص الصرع بجهازك.

CHB-MIT مسجّل بتركيب ثنائي (18 قناة، كل قناة فرق قطبين متجاورين). جهازك يسجّل كل قطب
مقابل مرجع واحد، فنحسب منه القنوات الثنائية الممكنة بالطرح (core/signal/montage.py)،
ونعيد تدريب النموذج العام على هذه القنوات فقط ونقيّمه بنفس الطريقة: اترك مريضاً.

تنبيه صادق: هذا هو «النموذج العام» نفسه الذي اكتشف 6 من 19 نوبة مع 7 إنذارات كاذبة في الساعة.
لا يوجد نموذج يعرف نوبات شخصك، ولا يمكن صنعه دون تسجيلات طبية لنوباته. لذلك المراقبة بجهازك
للتعلّم فقط، والواجهة تعرض الأرقام المقيسة بجانب كل قراءة.
"""
from collections import deque
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import resample_poly

from core.config import MODELS_DIR
from core.evaluation.report import load_report, save_report
from core.features.seizure import PER_CHANNEL, window_features
from core.signal.montage import bipolar, derivable_pairs, key

from .data import CHANNELS, FS, downloaded_recordings
from .pipeline import SETTINGS, THRESHOLD, VOTE_K, VOTE_N, WIN_SEC, STEP_SEC, decide, fit_balanced, timeline

MIN_PAIRS = 8      # أقل من هذا لا يغطي الرأس بما يكفي لنوبة تبدأ في أي مكان
SETTLE_SEC = 2.0   # ثانيتان يستقر فيهما الفلتر قبل أول نافذة
MODEL_PATH = MODELS_DIR / "device_epilepsy.joblib"
REPORT_ID = "device_epilepsy"


def _same(a: list[str], b: list[str]) -> bool:
    return [key(x) for x in a] == [key(x) for x in b]


def electrodes_needed() -> list[str]:
    """الأقطاب الـ19 التي يُبنى منها تركيب CHB-MIT."""
    out: list[str] = []
    for pair in CHANNELS:
        for e in pair.split("-"):
            if e not in out:
                out.append(e)
    return out


def requirements(ch_names: list[str]) -> dict:
    pairs = derivable_pairs(ch_names, CHANNELS)
    have = {key(n) for n in ch_names}
    missing = [e for e in electrodes_needed() if key(e) not in have]
    prepared = MODEL_PATH.exists() and _same(joblib.load(MODEL_PATH)["device_channels"], ch_names)
    if len(pairs) < MIN_PAIRS:
        status = "unsupported"
        reason = (f"من أقطاب جهازك يمكن حساب {len(pairs)} قناة فقط من قنوات CHB-MIT الـ18 (الحد الأدنى {MIN_PAIRS}). "
                  "كشف النوبات يحتاج أقطاباً تغطي الرأس كله.")
    elif not prepared:
        status, reason = "needs_prepare", "يحتاج تدريب النموذج على قنوات جهازك (زر «جهّز النماذج»)."
    else:
        status, reason = "ready", None
    return {"track": "epilepsy", "status": status, "reason": reason, "pairs": pairs,
            "n_pairs_total": len(CHANNELS), "missing_electrodes": missing}


def feature_columns(pairs: list[str]) -> np.ndarray:
    """أعمدة هذه القنوات في ميزات CHB-MIT المحفوظة (كل قناة كتلة متتالية من الميزات)."""
    n = len(PER_CHANNEL)
    return np.concatenate([np.arange(CHANNELS.index(p) * n, (CHANNELS.index(p) + 1) * n) for p in pairs])


def prepare(ch_names: list[str], log=print) -> dict | None:
    from collections import defaultdict

    from core.evaluation.metrics import combine_event_metrics

    from .pipeline import cached_features
    from .train import evaluate, pretty, window_summary

    req = requirements(ch_names)
    if req["status"] == "unsupported":
        log(f"فحص الصرع: {req['reason']} — تخطّي.")
        return None
    recs = downloaded_recordings()
    if not recs:
        log("فحص الصرع: بيانات CHB-MIT غير منزَّلة — تخطّي. (python -m scripts.download_data chbmit)")
        return None
    pairs = req["pairs"]
    cols = feature_columns(pairs)
    log(f"فحص الصرع: {len(pairs)} قناة ثنائية من أقطاب جهازك — قراءة ميزات {len(recs)} تسجيلاً…")
    feats = {}
    for r in recs:
        f = cached_features(r)
        feats[r.file] = {**f, "X": f["X"][:, cols]}
    by_patient: dict[str, list] = defaultdict(list)
    for r in recs:
        by_patient[r.patient].append(r)

    metrics, windows, per_patient = [], [], []
    for patient, prs in by_patient.items():
        train = [r for p, rs in by_patient.items() if p != patient for r in rs]
        model = fit_balanced(np.vstack([feats[r.file]["X"] for r in train]),
                             np.concatenate([feats[r.file]["y"] for r in train]))
        parts = []
        for test in prs:
            probs = model.predict_proba(feats[test.file]["X"])[:, 1]
            m, _ = evaluate(test, probs, feats)
            parts.append(m)
            windows.append((feats[test.file]["y"], probs))
        agg = combine_event_metrics(parts)
        per_patient.append({"patient": patient, "files": len(prs), **agg})
        metrics.extend(parts)
        log(f"  {patient} (اترك مريضاً): {pretty(agg)}")
    total = combine_event_metrics(metrics)

    joblib.dump({"model": fit_balanced(np.vstack([feats[r.file]["X"] for r in recs]),
                                       np.concatenate([feats[r.file]["y"] for r in recs])),
                 "pairs": pairs, "device_channels": list(ch_names), "fs": FS, "settings": SETTINGS,
                 "train_files": [r.file for r in recs]}, MODEL_PATH)
    full = (load_report("epilepsy_general") or {}).get("seizure_events")
    notes = [f"القنوات المستخدمة ({len(pairs)} من {len(CHANNELS)}): {'، '.join(pairs)}."]
    if full:
        notes.append(f"للمقارنة: بالقنوات الـ18 اكتشف النموذج العام {full['detected']} من {full['n_seizures']} "
                     f"مع {full['false_alarms_per_hour']:.1f} إنذار كاذب/ساعة.")
    notes += ["لا يوجد نموذج يعرف نوبات شخصك: النموذج الجيد (19 من 19) خاص بكل مريض ويحتاج تسجيلات طبية لنوباته.",
              "المراقبة بجهازك للتعلّم فقط — ليست تشخيصاً ولا إنذاراً طبياً."]
    save_report(REPORT_ID, {
        "title": "كشف الصرع — نموذج عام بأقطاب جهازك",
        "track": "epilepsy",
        "dataset": f"CHB-MIT — {len(by_patient)} مرضى، {len(recs)} تسجيلاً، بالقنوات الممكنة من أقطاب جهازك",
        "evaluation": "اترك مريضاً واحداً: ندرّب على المرضى الآخرين ونختبر على كل ملفات هذا المريض.",
        "seizure_events": total, "per_patient": per_patient, "summary": window_summary(windows),
        "device_channels": list(ch_names), "pairs": pairs, "settings": SETTINGS, "notes": notes,
    })
    log(f"  المجموع: {pretty(total)}")
    return {"seizure_events": total}


def _bundle(ch_names: list[str]) -> dict | None:
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle if _same(bundle["device_channels"], ch_names) else None


def reliability_text() -> str | None:
    ev = (load_report(REPORT_ID) or {}).get("seizure_events")
    if not ev:
        return None
    return (f"عند اختباره على أطفال لم يرهم اكتشف {ev['detected']} من {ev['n_seizures']} نوبة مع "
            f"{ev['false_alarms_per_hour']:.1f} إنذار كاذب في الساعة — للتعلّم فقط، ليس تشخيصاً.")


def to_model_rate(x: np.ndarray, fs: float) -> np.ndarray:
    """النموذج تدرّب على 256 عيّنة/ثانية — ميزات مثل line length تتغير بمعدل العيّنات."""
    if fs == FS:
        return x
    ratio = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1, padtype="line")


def live_analyzer(ch_names: list[str], fs: float):
    bundle = _bundle(ch_names)
    if bundle is None:
        return None
    model, pairs = bundle["model"], bundle["pairs"]
    win, win_model = int(round(WIN_SEC * fs)), int(round(WIN_SEC * FS))
    votes: deque[bool] = deque(maxlen=VOTE_N)
    state = {"next_end": SETTLE_SEC + WIN_SEC, "prob": None}
    note = reliability_text()

    def analyze(session) -> dict | None:
        while session.elapsed_sec >= state["next_end"]:
            end = state["next_end"]
            state["next_end"] += STEP_SEC
            lag = int(round((session.elapsed_sec - end) * fs))
            if lag + win > session.buffer.filled:
                continue
            size = session.buffer.buf.shape[1]
            window = session.buffer.buf[:, size - lag - win: size - lag]
            x = to_model_rate(bipolar(window, ch_names, pairs), fs)[:, :win_model]
            prob = float(model.predict_proba(window_features(x[None], FS))[0, 1])
            votes.append(prob >= THRESHOLD)
            state["prob"] = prob
        if state["prob"] is None:
            return None
        return {"prob": round(state["prob"], 3), "alarm": sum(votes) >= VOTE_K, "threshold": THRESHOLD,
                "device": True, "note": note}

    return analyze


def analyze_referential(path: Path, original_name: str) -> dict:
    """تحليل ملف EDF مسجّل بجهازك (كل قطب مقابل مرجع) — مثل تسجيلاتك المحفوظة من البث المباشر."""
    import mne

    from .pipeline import recording_features

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="error")
    names = list(raw.ch_names)
    if not MODEL_PATH.exists():
        raise ValueError("الملف ليس بتركيب CHB-MIT الثنائي، ولا يوجد نموذج مجهّز لأقطاب جهاز. "
                         "جهّز النماذج من صفحة «جهازك» أولاً.")
    bundle = joblib.load(MODEL_PATH)
    pairs = bundle["pairs"]
    if derivable_pairs(names, pairs) != pairs:
        raise ValueError(f"الملف لا يحتوي الأقطاب التي جُهّز عليها النموذج ({'، '.join(pairs)}).")
    fs = float(raw.info["sfreq"])
    data = to_model_rate(bipolar(raw.get_data() * 1e6, names, pairs), fs).astype(np.float32)
    X, starts = recording_features(data, FS)
    probs = bundle["model"].predict_proba(X)[:, 1]
    return {"file": original_name, "duration_sec": data.shape[1] / FS, "seizures": [],
            "general": timeline(starts, probs, decide(starts, probs)),
            "warnings": [w for w in [reliability_text(),
                                     f"حُلّل بالنموذج المجهّز لأقطاب جهازك ({len(pairs)} قناة ثنائية)."] if w]}
