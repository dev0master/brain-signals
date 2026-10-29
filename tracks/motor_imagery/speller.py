"""لوحة الحروف التي تُدار بأربعة أوامر فقط — كل أمر = تخيّل حركة.

    تخيّل اليد اليسرى  ← المؤشر خطوة لليسار   (التالي في ترتيب عربي من اليمين لليسار)
    تخيّل اليد اليمنى  ← المؤشر خطوة لليمين   (السابق)
    تخيّل القدمين      ← اختيار ما تحت المؤشر
    تخيّل اللسان       ← رجوع: من داخل مجموعة إلى المستوى الأعلى، أو حذف آخر حرف

الكتابة على مستويين: نختار مجموعة حروف، ثم الحرف داخلها. وفي المستوى الأعلى تظهر
أيضاً كلمات مقترحة تكمل الكلمة الجارية، فيكفي اختيار واحد بدل عدة حروف.

كل شيء هنا دوال نقية بلا EEG: الحالة تدخل، والأمر يُطبَّق، وحالة جديدة تخرج.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np

from .data import CLASSES

COMMANDS = ["left", "right", "select", "back"]
COMMAND_OF_CLASS = dict(zip(CLASSES, COMMANDS))  # التوزيع الافتراضي: left_hand→left، feet→select …
CLASS_OF_COMMAND = {v: k for k, v in COMMAND_OF_CLASS.items()}
COMMAND_AR = {"left": "يسار", "right": "يمين", "select": "اختيار", "back": "رجوع/حذف"}

# الأوامر مرتبة من الأكثر استخداماً في الكتابة إلى الأقل
COMMAND_PRIORITY = ["select", "left", "right", "back"]

# جمل لاختيار توزيع الأوامر — مختلفة عن جملة التقييم في train.py حتى لا نفصّل
# التوزيع على مقاس الاختبار
MAPPING_PHRASES = ["شكرا لك", "أين الطبيب"]


def ranked_mapping(recall: dict[str, float]) -> dict[str, str]:
    """الأمر الأكثر استخداماً للتخيّل الذي يُفهم صحيحاً أكثر، وهكذا نزولاً."""
    ordered = sorted(CLASSES, key=lambda c: recall[c], reverse=True)
    return dict(zip(COMMAND_PRIORITY, ordered))


def _recall(heldout: dict) -> dict[str, float]:
    y = np.asarray(heldout["y_true"])
    pred = np.asarray(heldout["probs"]).argmax(axis=1)
    return {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES)}


def choose_mapping(day1: dict, runs: int = 30, max_commands: int = 200,
                   margin: float = 0.10) -> tuple[dict[str, str], dict]:
    """توزيع الأوامر على أنواع التخيّل لشخص معيّن.

    المشكلة: عند أحد الأشخاص كان تخيّل القدمين (الافتراضي لـ«اختيار») يُفهم صحيحاً
    32% فقط، ويُقرأ غالباً «لسان» = حذف — فأكثر أمر نحتاجه كان يهدم ما كُتب.

    ما جرّبناه وفشل: تجربة التوزيعات الـ24 كلها واختيار الأفضل في المحاكاة. مع 24
    خياراً وبيانات قليلة "يفوز" غالباً الخيار المحظوظ لا الأفضل (لعنة الفائز) —
    وفي اليوم الثاني كان أسوأ من الافتراضي عند بعض الناس. نفس فخ تجربة استراتيجيات
    تداول كثيرة على نفس البيانات ثم الاحتفاظ بالأفضل.

    ما نفعله: مرشّحان فقط — الافتراضي، أو "الأمر الأهم للتخيّل الأدق" — نقارنهما
    بمحاكاة الكتابة على توقعات اليوم الأول (لم يرها النموذج أثناء تدريبه)، ولا نترك
    الافتراضي (الأسهل للتذكّر: اليد اليسرى = يسار) إلا إن تفوّق البديل بفارق واضح.
    """
    def score(mapping: dict[str, str]) -> dict:
        done, commands = [], []
        for phrase in MAPPING_PHRASES:
            for s in range(runs):
                m = simulate(phrase, {**day1, "mapping": mapping}, seed=s, max_commands=max_commands)["metrics"]
                done.append(m["chars_per_minute"] is not None)
                commands.append(m["commands"])
        return {"completion_rate": float(np.mean(done)), "median_commands": float(np.median(commands))}

    default = dict(CLASS_OF_COMMAND)
    ranked = ranked_mapping(_recall(day1))
    s_default, s_ranked = score(default), score(ranked)
    clearly_better = (
        s_ranked["completion_rate"] >= s_default["completion_rate"] + margin
        or (s_ranked["completion_rate"] >= s_default["completion_rate"]
            and s_ranked["median_commands"] <= (1 - margin) * s_default["median_commands"])
    )
    chosen = ranked if ranked != default and clearly_better else default
    return chosen, {"default": s_default, "ranked": s_ranked, "chose": "ranked" if chosen is ranked else "default"}

GROUPS: list[list[str]] = [
    ["ا", "أ", "إ", "آ", "ب", "ت"],
    ["ث", "ج", "ح", "خ", "د", "ذ"],
    ["ر", "ز", "س", "ش", "ص", "ض"],
    ["ط", "ظ", "ع", "غ", "ف", "ق"],
    ["ك", "ل", "م", "ن", "ه", "و"],
    ["ي", "ى", "ة", "ء", "ؤ", "ئ"],
    [" ", "؟", "،", "."],
]
SUPPORTED = {c for g in GROUPS for c in g}

# كلمات قد يحتاجها شخص لا يستطيع الكلام ولا الحركة — هذا من أهم استخدامات الكتابة بالدماغ
WORDS = [
    "نعم", "لا", "شكرا", "مرحبا", "أريد", "أحتاج", "ماء", "أكل", "ألم", "مساعدة", "طبيب", "ممرض",
    "حمام", "نوم", "تعبان", "بارد", "حار", "عطشان", "جائع", "أمي", "أبي", "أحبك", "الآن", "لاحقا",
    "غير", "مريح", "افتح", "أغلق", "الضوء", "النافذة", "التلفاز", "اقرأ", "اتصل", "أين", "متى", "كيف",
    "رأسي", "صدري", "ظهري", "بطني", "يدي", "رجلي", "خائف", "سعيد", "حزين", "بخير", "قليلا", "كثيرا",
]
MAX_SUGGESTIONS = 3


@dataclass(frozen=True)
class State:
    text: str = ""
    group: int | None = None  # None = المستوى الأعلى
    cursor: int = 0


def current_word(text: str) -> str:
    """الكلمة التي تُكتب الآن (بعد آخر مسافة)."""
    return text.split(" ")[-1]


def suggestions(text: str) -> list[str]:
    prefix = current_word(text)
    if not prefix:
        return []
    return [w for w in WORDS if w.startswith(prefix) and w != prefix][:MAX_SUGGESTIONS]


def items(state: State) -> list[dict]:
    """ما يظهر في الصف الحالي. المجموعات أولاً بترتيب ثابت، ثم الاقتراحات."""
    if state.group is not None:
        return [{"kind": "char", "value": c} for c in GROUPS[state.group]]
    return ([{"kind": "group", "value": i, "label": " ".join("␣" if c == " " else c for c in g)}
             for i, g in enumerate(GROUPS)]
            + [{"kind": "word", "value": w} for w in suggestions(state.text)])


def apply(state: State, command: str | None) -> State:
    """يطبّق أمراً واحداً. None = لا شيء (النموذج لم يكن واثقاً فامتنع)."""
    if command is None:
        return state
    row = items(state)
    n = len(row)
    if command == "left":
        return replace(state, cursor=(state.cursor + 1) % n)
    if command == "right":
        return replace(state, cursor=(state.cursor - 1) % n)
    if command == "back":
        if state.group is not None:
            return State(state.text, None, state.group)
        shorter = replace(state, text=state.text[:-1])
        # بعد الحذف قد تختفي اقتراحات فيقصر الصف — نبقي المؤشر داخله
        return replace(shorter, cursor=min(state.cursor, len(items(shorter)) - 1))
    if command == "select":
        item = row[state.cursor % n]
        if item["kind"] == "group":
            return State(state.text, item["value"], 0)
        if item["kind"] == "char":
            return State(state.text + item["value"], None, state.group)
        # كلمة مقترحة: تستبدل الجزء المكتوب من الكلمة الجارية وتضيف مسافة
        prefix = current_word(state.text)
        return State(state.text[: len(state.text) - len(prefix)] + item["value"] + " ", None, 0)
    raise ValueError(f"أمر غير معروف: {command}")


def validate_target(target: str) -> None:
    bad = sorted({c for c in target if c not in SUPPORTED})
    if bad:
        raise ValueError(f"أحرف غير موجودة في اللوحة: {' '.join(bad)}")


def is_done(state: State, target: str) -> bool:
    return state.text.strip() == target.strip()


def _steps_toward(cursor: int, goal: int, n: int) -> str:
    """أقصر اتجاه في صف دائري."""
    forward = (goal - cursor) % n
    return "left" if forward <= n - forward else "right"


def plan(state: State, target: str) -> str:
    """الأمر التالي الذي سيتخيّله مستخدم يعرف طريقه إلى الهدف بأقصر مسار."""
    text = state.text
    # خطأ سابق؟ نحذف حتى يعود النص بداية صحيحة للهدف (نقبل مسافة زائدة بعد كلمة مقترحة)
    if not (target + " ").startswith(text):
        return "back"
    remaining = target[len(text):]
    if not remaining:
        return "back"  # لا يحدث عادة: is_done يوقف المحاكاة قبله
    if state.group is not None:
        group = GROUPS[state.group]
        if remaining[0] not in group:
            return "back"
        goal = group.index(remaining[0])
        return "select" if state.cursor == goal else _steps_toward(state.cursor, goal, len(group))
    row = items(state)
    goal = None
    prefix = current_word(text)
    for i, item in enumerate(row):
        if item["kind"] == "word":
            completed = text[: len(text) - len(prefix)] + item["value"]
            if (target + " ").startswith(completed + " "):  # الاقتراح يكمل كلمة كاملة من الهدف
                goal = i
                break
    if goal is None:
        goal = next(i for i, g in enumerate(GROUPS) if remaining[0] in g)
    return "select" if state.cursor == goal else _steps_toward(state.cursor, goal, len(row))


def view(state: State) -> dict:
    return {**asdict(state), "items": items(state)}


# ---------- المستخدم المحاكى ----------

SECONDS_PER_COMMAND = 6.0  # تحضير ~2ث + تخيّل حتى القرار 2.5ث + راحة ~1.5ث


def simulate(target: str, heldout: dict, seed: int = 0, max_commands: int = 300,
             min_confidence: float = 0.0) -> dict:
    """محاكاة صادقة للكتابة.

    في كل خطوة يقرر "المستخدم" الأمر الصحيح التالي (plan)، ثم نسحب محاولة حقيقية
    لذلك التخيّل من جلسة الاختبار — اليوم الذي لم يتدرّب عليه النموذج — ونأخذ
    ما فهمه النموذج منها فعلاً. أخطاء النموذج تُطبَّق كما هي، ويصححها المستخدم بعدها.

    heldout: {"y_true": [فئة كل محاولة], "probs": [[احتمال كل فئة]],
              "mapping": {أمر: نوع التخيّل} اختياري — وإلا التوزيع الافتراضي} من train.py
    min_confidence: إن كانت أعلى ثقة أقل منها يمتنع النظام عن التنفيذ (أبطأ لكن أخطاء أقل).
    """
    validate_target(target)
    rng = np.random.default_rng(seed)
    y_true = np.asarray(heldout["y_true"])
    probs = np.asarray(heldout["probs"])
    pool = {c: np.flatnonzero(y_true == i) for i, c in enumerate(CLASSES)}
    class_of = heldout.get("mapping") or CLASS_OF_COMMAND
    command_of = {cls: cmd for cmd, cls in class_of.items()}

    state = State()
    steps = []
    correct = executed = 0
    while not is_done(state, target) and len(steps) < max_commands:
        intended = plan(state, target)
        trial = int(rng.choice(pool[class_of[intended]]))
        p = probs[trial]
        decoded = command_of[CLASSES[int(np.argmax(p))]]
        command = decoded if p.max() >= min_confidence else None
        state = apply(state, command)
        if command is not None:
            executed += 1
            correct += command == intended
        steps.append({"intended": intended, "decoded": decoded, "executed": command,
                      "probs": [round(float(x), 3) for x in p], "trial": trial, "view": view(state)})

    finished = is_done(state, target)
    minutes = len(steps) * SECONDS_PER_COMMAND / 60
    return {
        "target": target,
        "finished": finished,
        "steps": steps,
        "metrics": {
            "commands": len(steps),
            "command_accuracy": correct / executed if executed else None,
            "abstained": len(steps) - executed,
            "minutes": round(minutes, 2),
            "chars_per_minute": round(len(target) / minutes, 2) if finished and minutes else None,
            "ideal_commands": ideal_commands(target),
            "seconds_per_command": SECONDS_PER_COMMAND,
        },
    }


def typical_performance(target: str, heldout: dict, runs: int = 20, max_commands: int = 300) -> dict:
    """محاكاة واحدة قد تكون محظوظة أو منحوسة — خطأ واحد قد يجرّ سلسلة تصحيحات.
    لذلك نكرر المحاكاة ونأخذ الوسيط."""
    results = [simulate(target, heldout, seed=s, max_commands=max_commands)["metrics"] for s in range(runs)]
    done = [r for r in results if r["chars_per_minute"] is not None]
    return {
        "target": target,
        "runs": runs,
        "completion_rate": len(done) / runs,
        "median_commands": float(np.median([r["commands"] for r in results])),
        "median_chars_per_minute": float(np.median([r["chars_per_minute"] for r in done])) if done else None,
        "ideal_commands": ideal_commands(target),
    }


def ideal_commands(target: str) -> int:
    """عدد الأوامر لو لم يخطئ النموذج أبداً — الحد الأدنى الممكن."""
    validate_target(target)
    state, n = State(), 0
    while not is_done(state, target):
        state = apply(state, plan(state, target))
        n += 1
        if n > 1000:
            raise RuntimeError("المخطط لم يصل للهدف")
    return n
