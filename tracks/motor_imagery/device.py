"""الكتابة بالدماغ بجهازك: معايرة شخصية ثم كتابة مباشرة.

نماذج BNCI الجاهزة لا تصلح لك: كل منها تعلّم دماغ شخص واحد وأقطاب جهاز واحد. لذلك يبدأ
كل شخص بجلسة معايرة، تماماً كما سجّل أشخاص BNCI يومهم الأول:
  1. يظهر أمر على الشاشة (مثلاً «اليد اليسرى») فيتخيّل الحركة 4 ثوانٍ دون أن يتحرك.
  2. كل أمر من الأربعة يتكرر N مرة بترتيب عشوائي، حتى لا يتوقع الدماغ ما بعده.
  3. نقص النافذة نفسها التي تدرّبت عليها نماذج BNCI: من 0.5 إلى 2.5 ث بعد ظهور الأمر،
     من إشارة مفلترة 8–30Hz بفلتر سببي على البث المتصل (data.py: epochs_from_raw).
  4. ندرّب طريقة واحدة ثابتة: ريماني + انحدار لوجستي، الأفضل عند 6 من 9 أشخاص في BNCI.
     لا نختار بين طرق على بياناتك القليلة — اختيار الأفضل من بين خيارات على بيانات قليلة
     يضخّم الرقم المعروض (لعنة الفائز).
  5. الدقة المعروضة بتقسيم داخلي لجلسة المعايرة نفسها — تقدير متفائل: في BNCI انخفضت الدقة
     من اليوم الأول إلى الثاني عند أغلب الأشخاص. أعد المعايرة في بداية كل جلسة كتابة إن ساءت النتائج.

ثم الكتابة: كل 6 ثوانٍ دورة واحدة — «استعد» ثانيتان، ثم «تخيّل الآن»، وبعد 2.5 ث يقرأ النموذج
الأمر من الإشارة وينفّذه على لوحة الحروف (speller.py).
"""
from datetime import datetime

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from core.config import MAINS_HZ, MODELS_DIR
from core.evaluation.metrics import classification_summary
from core.evaluation.report import save_report
from core.signal.filters import StreamFilter, make_chain
from core.signal.montage import key, missing
from core.signal.windows import RingBuffer

from . import speller
from .data import BAND, CLASS_AR, CLASSES, EPOCH_SEC
from .train import BENCHMARK_PHRASE, PIPELINE_AR, PIPELINES

PIPELINE = "riemann_ts_lr"
PERSON_MODEL = MODELS_DIR / "device_speller.joblib"
REPORT_ID = "device_speller"

LEAD_IN_SEC = 5.0          # قبل أول أمر: يستقر الفلتر ويستعد الشخص
PREPARE_SEC = 2.0          # «استعد»
IMAGINE_SEC = 4.0          # الأمر ظاهر: تخيّل
REST_SEC = (1.5, 2.5)      # راحة عشوائية الطول بين المحاولات
PER_CLASS = (2, 40)        # حدود عدد المحاولات لكل أمر
REQUIRED = ["C3", "C4"]    # تخيّل اليدين يظهر فوق منطقة الحركة في الجهة المعاكسة
RECOMMENDED = ["Cz", "FC3", "FC4", "CP3", "CP4"]

COMMAND_ICON = {"left": "←", "right": "→", "select": "✓", "back": "⌫"}


def _matches(bundle: dict, ch_names: list[str], fs: float, board: str | None) -> bool:
    return ([key(n) for n in bundle["ch_names"]] == [key(n) for n in ch_names]
            and float(bundle["fs"]) == float(fs) and bundle.get("board") == board)


def person_model(ch_names: list[str], fs: float, board: str | None) -> dict | None:
    if not PERSON_MODEL.exists():
        return None
    bundle = joblib.load(PERSON_MODEL)
    return bundle if _matches(bundle, ch_names, fs, board) else None


def requirements(ch_names: list[str], fs: float, board: str | None) -> dict:
    lacking = missing(ch_names, REQUIRED)
    if lacking:
        return {"track": "motor_imagery", "status": "unsupported",
                "reason": f"تخيّل حركة اليدين يظهر فوق {' و'.join(REQUIRED)} — جهازك لا يغطي {'، '.join(lacking)}.",
                "missing_recommended": missing(ch_names, RECOMMENDED)}
    bundle = person_model(ch_names, fs, board)
    return {"track": "motor_imagery", "status": "ready" if bundle else "needs_calibration",
            "reason": None if bundle else "يحتاج جلسة معايرة لك (نحو 5 دقائق) من صفحة الكتابة بالدماغ.",
            "missing_recommended": missing(ch_names, RECOMMENDED),
            "calibrated_at": bundle.get("created_at") if bundle else None,
            "accuracy": bundle.get("accuracy") if bundle else None}


def commands_meta(mapping: dict[str, str]) -> list[dict]:
    """لكل أمر: نوع التخيّل الذي يفعّله ووظيفته على اللوحة."""
    return [{"command": cmd, "class": cls, "imagine": CLASS_AR[cls], "action": speller.COMMAND_AR[cmd],
             "icon": COMMAND_ICON[cmd]} for cmd, cls in mapping.items()]


class _Epochs:
    """فلتر 8–30Hz سببي على البث + ذاكرة لآخر 10 ثوانٍ، لقص نافذة كل أمر بعد اكتمالها."""

    def __init__(self, n_ch: int, fs: float):
        self.fs = fs
        self.filter = StreamFilter(make_chain(fs, *BAND, MAINS_HZ), n_ch)
        self.buffer = RingBuffer(n_ch, int(10 * fs))
        self.n = 0  # عدد العيّنات منذ البداية

    def push(self, raw: np.ndarray) -> None:
        if raw.shape[1]:
            self.buffer.push(self.filter.process(raw))
            self.n += raw.shape[1]

    @property
    def t(self) -> float:
        return self.n / self.fs

    def window_after(self, cue_sec: float) -> np.ndarray | None:
        """النافذة 0.5–2.5 ث بعد ظهور الأمر، إن اكتملت وما زالت في الذاكرة."""
        a = int(round((cue_sec + EPOCH_SEC[0]) * self.fs))
        b = a + int(round((EPOCH_SEC[1] - EPOCH_SEC[0]) * self.fs))
        if self.n < b or self.n - a > self.buffer.filled:
            return None
        size = self.buffer.buf.shape[1]
        end = size - (self.n - b)
        return self.buffer.buf[:, end - (b - a): end].copy()


def schedule(per_class: int, seed: int) -> list[dict]:
    """ترتيب المحاولات وتوقيتها بزمن الجلسة."""
    rng = np.random.default_rng(seed)
    classes = [c for c in CLASSES for _ in range(per_class)]
    rng.shuffle(classes)
    trials, t = [], LEAD_IN_SEC
    for cls in classes:
        cue = t + PREPARE_SEC
        trials.append({"class": cls, "start": t, "cue": cue, "end": cue + IMAGINE_SEC})
        t = cue + IMAGINE_SEC + float(rng.uniform(*REST_SEC))
    return trials


def train_person(X: np.ndarray, y: np.ndarray, fs: float, ch_names: list[str], board: str | None) -> dict:
    """يدرّب نموذج الشخص ويقيس دقته بتقسيم داخلي لجلسة المعايرة."""
    per_class = int(np.bincount(y, minlength=len(CLASSES)).min())
    cv = StratifiedKFold(min(5, per_class), shuffle=True, random_state=0)
    probs = cross_val_predict(PIPELINES[PIPELINE](), X, y, cv=cv, method="predict_proba")
    pred = probs.argmax(axis=1)
    labels = [CLASS_AR[c] for c in CLASSES]
    summary = classification_summary([labels[i] for i in y], [labels[i] for i in pred], labels=labels)
    heldout = {"y_true": y.tolist(), "probs": probs.tolist()}
    mapping, search = speller.choose_mapping(heldout)
    recall = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)}
    spelling = speller.typical_performance(BENCHMARK_PHRASE, {**heldout, "mapping": mapping})
    created = datetime.now().isoformat(timespec="seconds")
    joblib.dump({"model": PIPELINES[PIPELINE]().fit(X, y), "pipeline": PIPELINE, "classes": CLASSES, "fs": fs,
                 "ch_names": list(ch_names), "board": board, "epoch_sec": EPOCH_SEC, "band": BAND,
                 "mapping": mapping, "accuracy": summary["accuracy"], "created_at": created}, PERSON_MODEL)
    save_report(REPORT_ID, {
        "title": "الكتابة بالدماغ — معايرة جهازك",
        "track": "motor_imagery_device",
        "dataset": f"جلسة معايرتك: {len(y)} محاولة ({per_class} لكل أمر)، {len(ch_names)} قطباً، {fs:g} عيّنة/ثانية",
        "evaluation": f"تقسيم داخلي ({cv.get_n_splits()} أجزاء) لجلسة المعايرة نفسها — كل محاولة توقّعها نموذج لم يرها.",
        "summary": summary, "recall": recall, "mapping": mapping, "mapping_search": search,
        "spelling": spelling, "pipeline_ar": PIPELINE_AR[PIPELINE],
        "notes": ["مستوى الصدفة 25%: أربعة أوامر متساوية العدد.",
                  "تقدير متفائل: المحاولات من الجلسة نفسها. في BNCI انخفضت الدقة في اليوم التالي عند أغلب الناس — "
                  "أعد المعايرة في بداية كل جلسة كتابة.",
                  f"سرعة الكتابة المتوقعة: محاكاة كتابة «{BENCHMARK_PHRASE}» بتوقعات هذه المحاولات، 6 ثوانٍ لكل أمر.",
                  "أول مرة تكون الإشارة ضعيفة عادة؛ التخيّل مهارة تتحسن بالتمرين (تخيّل الإحساس بالحركة لا صورتها)."],
    })
    return {"accuracy": summary["accuracy"], "balanced_accuracy": summary["balanced_accuracy"], "chance": 0.25,
            "recall": {CLASS_AR[c]: r for c, r in recall.items()}, "commands": commands_meta(mapping),
            "spelling": spelling, "n_trials": len(y)}


class Calibration:
    """محلّل جلسة المعايرة: يوجّه الشخص أمراً بعد أمر، ويقص المحاولات، ثم يدرّب نموذجه."""

    def __init__(self, ch_names: list[str], fs: float, board: str | None, per_class: int = 10,
                 seed: int | None = None):
        self.ch_names, self.fs, self.board = list(ch_names), fs, board
        self.per_class = int(np.clip(per_class, *PER_CLASS))
        self.trials = schedule(self.per_class, seed if seed is not None else int(datetime.now().timestamp()))
        self.epochs = _Epochs(len(ch_names), fs)
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self._collected = 0
        self._marked = 0
        self.phase = "lead_in"
        self.result: dict | None = None
        self.error: str | None = None

    def __call__(self, session) -> dict:
        self.epochs.push(session.last_raw)
        t = self.epochs.t
        # علامة لكل أمر في ملف EDF المحفوظ — لإعادة التدريب لاحقاً من التسجيل نفسه
        while self._marked < len(self.trials) and t >= self.trials[self._marked]["cue"]:
            tr = self.trials[self._marked]
            session.mark(tr["class"], onset_sec=tr["cue"], duration=IMAGINE_SEC)
            self._marked += 1
        while self._collected < len(self.trials):
            w = self.epochs.window_after(self.trials[self._collected]["cue"])
            if w is None:
                break
            self.X.append(w)
            self.y.append(CLASSES.index(self.trials[self._collected]["class"]))
            self._collected += 1

        if self.result is not None or self.error is not None:
            return self._status(t)
        if self._collected == len(self.trials):
            if self.phase != "training":
                self.phase = "training"  # إطار واحد يقول «جارٍ التدريب» قبل أن يبدأ فعلاً
                return self._status(t)
            try:
                self.result = train_person(np.asarray(self.X, dtype=np.float64), np.asarray(self.y),
                                           self.fs, self.ch_names, self.board)
                self.phase = "done"
            except Exception as exc:  # بيانات غير صالحة (قطب مسطّح مثلاً)
                self.error, self.phase = f"تعذّر التدريب: {exc}", "error"
            return self._status(t)
        return self._status(t)

    def _status(self, t: float) -> dict:
        n = len(self.trials)
        out: dict = {"n_trials": n, "collected": self._collected, "per_class": self.per_class}
        if self.phase in ("training", "done", "error"):
            return {**out, "phase": self.phase, "result": self.result, "error": self.error}
        i = next((k for k, tr in enumerate(self.trials) if t < tr["end"]), None)
        if i is None:  # انتهت آخر محاولة وننتظر اكتمال نافذتها
            return {**out, "phase": "rest", "trial": n, "remaining_sec": 0}
        tr = self.trials[i]
        if t < tr["start"]:
            phase, remaining = ("lead_in" if i == 0 else "rest"), tr["start"] - t
        elif t < tr["cue"]:
            phase, remaining = "prepare", tr["cue"] - t
        else:
            phase, remaining = "imagine", tr["end"] - t
        self.phase = phase
        return {**out, "phase": phase, "trial": i + 1, "remaining_sec": round(remaining, 1),
                "class": tr["class"] if phase == "imagine" else None,
                "imagine": CLASS_AR[tr["class"]] if phase == "imagine" else None}


class LiveSpeller:
    """محلّل الكتابة المباشرة: دورة كل 6 ثوانٍ، والنموذج يقرأ الأمر من الإشارة."""

    CYCLE_SEC = speller.SECONDS_PER_COMMAND  # 2 استعداد + 2.5 حتى القرار + 1.5 لعرض النتيجة

    def __init__(self, bundle: dict, ch_names: list[str], fs: float, min_confidence: float = 0.0,
                 target: str = ""):
        self.model = bundle["model"]
        self.mapping: dict[str, str] = bundle["mapping"]
        self.command_of = {cls: cmd for cmd, cls in self.mapping.items()}
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.target = target.strip()
        if self.target:
            speller.validate_target(self.target)
        self.epochs = _Epochs(len(ch_names), fs)
        self.state = speller.State()
        self.cycle_start = LEAD_IN_SEC
        self.count = 0
        self.last: dict | None = None

    @property
    def cue(self) -> float:
        return self.cycle_start + PREPARE_SEC

    def __call__(self, session) -> dict:
        self.epochs.push(session.last_raw)
        while (w := self.epochs.window_after(self.cue)) is not None:
            probs = self.model.predict_proba(w[None].astype(np.float64))[0]
            decoded = self.command_of[CLASSES[int(np.argmax(probs))]]
            executed = decoded if probs.max() >= self.min_confidence else None
            self.state = speller.apply(self.state, executed)
            self.count += 1
            self.last = {"decoded": decoded, "executed": executed, "confidence": round(float(probs.max()), 3),
                         "probs": {self.command_of[c]: round(float(p), 3) for c, p in zip(CLASSES, probs)}}
            session.mark(f"decoded:{decoded}", onset_sec=self.cue, duration=EPOCH_SEC[1])
            self.cycle_start += self.CYCLE_SEC
        t = self.epochs.t
        decision = self.cue + EPOCH_SEC[1]
        if t < self.cycle_start:
            phase, remaining = ("lead_in" if self.count == 0 else "result"), self.cycle_start - t
        elif t < self.cue:
            phase, remaining = "prepare", self.cue - t
        else:
            phase, remaining = "imagine", max(0.0, decision - t)
        out = {"phase": phase, "remaining_sec": round(remaining, 1), "view": speller.view(self.state),
               "last": self.last, "count": self.count, "commands": commands_meta(self.mapping),
               "minutes": round(self.count * self.CYCLE_SEC / 60, 2)}
        if self.target:
            done = speller.is_done(self.state, self.target)
            out["target"] = self.target
            out["done"] = done
            out["suggested"] = None if done else speller.plan(self.state, self.target)
        return out
