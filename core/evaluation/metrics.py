"""مقاييس صادقة.

1) التصنيف: الدقة وحدها لا تكفي. نعرض دائماً "مستوى الصدفة" — الدقة التي يحصل
   عليها من يخمّن عشوائياً أو يختار الفئة الأكثر دائماً. نموذج دقته 30% في
   مسألة من 4 فئات متساوية أفضل من الصدفة بقليل فقط (25%).

2) الصرع: النوبات أقل من 1% من وقت التسجيل. نموذج يقول "لا نوبة" دائماً
   دقته 99% وهو عديم الفائدة. المقاييس الصحيحة ثلاثة:
   - الحساسية: كم نوبة من النوبات الحقيقية اكتُشفت؟
   - الإنذارات الكاذبة في الساعة: كم مرة صرخ النظام بلا نوبة؟
   - زمن الاكتشاف: كم ثانية بعد بداية النوبة انتبه النظام؟
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix)


def classification_summary(y_true, y_pred, labels: list | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    counts = np.array([(y_true == lab).sum() for lab in labels])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recall = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(len(labels)),
                       where=cm.sum(axis=1) > 0)
    return {
        "n": int(len(y_true)),
        "labels": [str(lab) for lab in labels],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        # الصدفة: تخمين عشوائي متساوٍ، واختيار الفئة الأكثر دائماً
        "chance_uniform": 1.0 / len(labels),
        "chance_majority": float(counts.max() / max(counts.sum(), 1)),
        "per_class_recall": {str(lab): float(r) for lab, r in zip(labels, recall)},
        "confusion_matrix": cm.tolist(),
    }


# ---------- الصرع: من احتمالات النوافذ إلى أحداث ----------

@dataclass
class Event:
    start: float  # ثانية
    end: float
    # متى أطلق النظام الإنذار فعلاً (بعد اكتمال تصويت k من n). البداية قد تكون
    # أبكر لأننا نعود لأول نافذة مشبوهة، لكن الإنذار لا يصل إلا عند هذه اللحظة.
    alarm: float | None = None

    def overlaps(self, other: "Event", tolerance: float = 0.0) -> bool:
        return self.start <= other.end + tolerance and other.start <= self.end + tolerance

    def as_dict(self) -> dict:
        d = {"start": round(self.start, 2), "end": round(self.end, 2)}
        if self.alarm is not None:
            d["alarm"] = round(self.alarm, 2)
        return d


def windows_to_events(starts: np.ndarray, positive: np.ndarray, win_sec: float,
                      k: int = 3, n: int = 4, merge_gap_sec: float = 10.0) -> list[Event]:
    """يحوّل قرارات النوافذ (نوبة/لا) إلى أحداث.

    تنعيم "k من n": نعتبر لحظة ما نوبة فقط إن كانت k من آخر n نوافذ إيجابية.
    هذا يمنع نافذة شاذة واحدة من إطلاق إنذار، مقابل تأخير بسيط في الاكتشاف.
    ثم ندمج الأحداث التي تفصلها فجوة قصيرة.
    """
    positive = np.asarray(positive, dtype=bool)
    if positive.size == 0:
        return []
    kernel = np.ones(n, dtype=int)
    votes = np.convolve(positive.astype(int), kernel)[: len(positive)]  # آخر n نوافذ
    smoothed = votes >= k

    events: list[Event] = []
    in_event = False
    for i, on in enumerate(smoothed):
        if not on:
            in_event = False
            continue
        # حدود الحدث تُؤخذ من النوافذ الإيجابية فعلاً داخل نافذة التصويت،
        # لا من التصويت نفسه (الذي يتأخر n-1 نوافذ بعد آخر نافذة إيجابية)
        first = max(0, i - n + 1)
        hits = first + np.flatnonzero(positive[first : i + 1])
        end = float(starts[hits[-1]] + win_sec)
        if not in_event:
            # الإنذار يُطلق عند نهاية النافذة التي اكتمل بها التصويت
            events.append(Event(float(starts[hits[0]]), end, alarm=float(starts[i] + win_sec)))
            in_event = True
        else:
            events[-1].end = max(events[-1].end, end)

    merged: list[Event] = []
    for ev in events:
        if merged and ev.start - merged[-1].end <= merge_gap_sec:
            merged[-1].end = max(merged[-1].end, ev.end)
        else:
            merged.append(ev)
    return merged


def event_metrics(true_events: list[Event], detected: list[Event], total_hours: float,
                  tolerance_sec: float = 30.0) -> dict:
    """مقاييس الأحداث.

    نوبة حقيقية "مكتشفة" إن تداخل معها اكتشاف (مع سماحية tolerance_sec قبلها).
    اكتشاف "كاذب" إن لم يتداخل مع أي نوبة حقيقية.
    زمن الاكتشاف = لحظة الإنذار الفعلي − بداية النوبة.
    """
    hits, latencies = 0, []
    for t in true_events:
        matching = [d for d in detected if d.overlaps(t, tolerance_sec)]
        if matching:
            hits += 1
            first = min(d.alarm if d.alarm is not None else d.start for d in matching)
            latencies.append(max(0.0, first - t.start))
    false_alarms = sum(1 for d in detected
                       if not any(d.overlaps(t, tolerance_sec) for t in true_events))
    return {
        "n_seizures": len(true_events),
        "detected": hits,
        "false_alarms": false_alarms,
        "hours": total_hours,
        "latencies": latencies,
        **rates(hits, len(true_events), false_alarms, total_hours, latencies),
    }


def rates(hits: int, n_seizures: int, false_alarms: int, hours: float, latencies: list[float]) -> dict:
    return {
        "sensitivity": hits / n_seizures if n_seizures else None,
        "false_alarms_per_hour": false_alarms / hours if hours > 0 else None,
        "mean_latency_sec": float(np.mean(latencies)) if latencies else None,
    }


def combine_event_metrics(parts: list[dict]) -> dict:
    """يجمع مقاييس عدة تسجيلات في رقم واحد (بالعدّ، لا بمتوسط النسب)."""
    hits = sum(p["detected"] for p in parts)
    n = sum(p["n_seizures"] for p in parts)
    fa = sum(p["false_alarms"] for p in parts)
    hours = sum(p["hours"] for p in parts)
    lat = [x for p in parts for x in p["latencies"]]
    return {"n_seizures": n, "detected": hits, "false_alarms": fa, "hours": hours,
            "latencies": lat, **rates(hits, n, fa, hours, lat)}
