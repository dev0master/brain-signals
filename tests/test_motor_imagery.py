import json

import numpy as np
import pytest

from tracks.motor_imagery import speller, train
from tracks.motor_imagery.data import CLASSES
from tracks.motor_imagery.speller import (GROUPS, State, apply, ideal_commands, is_done, items, plan,
                                          simulate, suggestions)


def run(commands, state=State()):
    for c in commands:
        state = apply(state, c)
    return state


def test_typing_a_letter_goes_into_group_then_back_to_top():
    s = run(["left", "select"])                       # المجموعة الثانية
    assert s.group == 1 and s.cursor == 0
    s = run(["left", "left", "select"], s)            # ث ج ح → ح
    assert s.text == "ح" and s.group is None and s.cursor == 1


def test_cursor_wraps_around_and_right_goes_backwards():
    assert run(["right"]).cursor == len(GROUPS) - 1


def test_back_leaves_group_then_deletes():
    s = run(["select", "select"])                    # "ا"
    s = run(["select"], s)                           # داخل المجموعة الأولى
    s = apply(s, "back")
    assert s.group is None and s.text == "ا"
    assert apply(s, "back").text == ""


def test_word_suggestion_completes_current_word():
    s = State(text="أنا أح")
    assert "أحتاج" in suggestions(s.text) and "أحبك" in suggestions(s.text)
    idx = next(i for i, it in enumerate(items(s)) if it["value"] == "أحتاج")
    s = apply(State(s.text, None, idx), "select")
    assert s.text == "أنا أحتاج "


@pytest.mark.parametrize("target", ["مرحبا", "أحتاج ماء", "ألم شديد؟", "لا"])
def test_planner_reaches_any_target_with_perfect_control(target):
    s, n = State(), 0
    while not is_done(s, target):
        s = apply(s, plan(s, target))
        n += 1
        assert n < 200
    assert n == ideal_commands(target)


def test_planner_uses_suggestion_when_it_saves_steps():
    # بعد "م" يظهر "مرحبا" مقترحاً فيُختار بدل كتابة 4 أحرف أخرى
    assert ideal_commands("مرحبا") < ideal_commands("مرحبب")


def test_planner_corrects_a_wrong_letter():
    s = State(text="مس")
    assert plan(s, "مرحبا") == "back"
    assert plan(State(text="مس", group=2), "مرحبا") == "back"  # يخرج من المجموعة أولاً


def test_validate_target_rejects_unsupported_characters():
    with pytest.raises(ValueError, match="x"):
        speller.validate_target("ماء x")


def heldout(accuracy: float, n_per_class=50, seed=0):
    """محاولات مزيّفة: النموذج يصيب بنسبة accuracy ويخطئ عشوائياً في الباقي."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), n_per_class)
    probs = []
    for t in y:
        guess = t if rng.random() < accuracy else rng.choice([c for c in range(4) if c != t])
        p = np.full(4, 0.1)
        p[guess] = 0.7
        probs.append(p)
    return {"y_true": y.tolist(), "probs": probs}


def test_simulation_with_perfect_decoder_matches_ideal():
    r = simulate("أحتاج ماء", heldout(1.0))
    assert r["finished"]
    assert r["metrics"]["commands"] == r["metrics"]["ideal_commands"]
    assert r["metrics"]["command_accuracy"] == 1.0


def test_simulation_with_imperfect_decoder_corrects_errors_and_is_slower():
    # seed=0 فيه أخطاء فعلاً (بعض البذور قد تسحب محاولات صحيحة كلها بالصدفة)
    r = simulate("مرحبا", heldout(0.75), seed=0)
    assert r["finished"]
    assert r["metrics"]["commands"] > r["metrics"]["ideal_commands"]
    assert any(s["executed"] != s["intended"] for s in r["steps"])
    assert r["steps"][-1]["view"]["text"].strip() == "مرحبا"


def test_simulation_at_chance_level_gives_up_instead_of_looping():
    r = simulate("أحتاج ماء", heldout(0.25), max_commands=60)
    assert not r["finished"]
    assert r["metrics"]["commands"] == 60
    assert r["metrics"]["chars_per_minute"] is None


def test_min_confidence_makes_the_system_abstain():
    r = simulate("لا", heldout(1.0), min_confidence=0.9)  # الثقة 0.7 دائماً < 0.9
    assert r["metrics"]["abstained"] == r["metrics"]["commands"]
    assert not r["finished"] or r["metrics"]["commands"] == 0


# ---------- التدريب على بيانات اصطناعية ----------

def synthetic_subject(seed=0, n_per_class=40, n_ch=8, n_t=200):
    """كل أمر يزيد التباين في زوج أقطاب مختلف — شبيه بأثر التخيّل على قدرة ميو/بيتا."""
    rng = np.random.default_rng(seed)
    X, y, session = [], [], []
    for sess in ("0train", "1test"):
        for c in range(4):
            for _ in range(n_per_class):
                x = rng.standard_normal((n_ch, n_t))
                x[2 * c : 2 * c + 2] *= 2.5
                X.append(x)
                y.append(c)
                session.append(sess)
    return {"X": np.array(X, dtype=np.float32), "y": np.array(y), "session": np.array(session),
            "fs": 100.0, "ch_names": [f"C{i}" for i in range(n_ch)]}


def test_train_subject_selects_on_day_one_and_reports_day_two(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_subject", lambda s: synthetic_subject(s))
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "HELDOUT_DIR", tmp_path / "mi_heldout")
    r = train.train_subject(1)
    assert r["pipeline"] in train.PIPELINES
    assert set(r["inner_cv"]) == set(train.PIPELINES)
    assert r["test_accuracy"] > 0.6  # الصدفة 25%
    saved = json.loads((tmp_path / "mi_heldout" / "subject_1.json").read_text(encoding="utf-8"))
    assert len(saved["y_true"]) == r["n_test"] == 160
    assert np.allclose(np.sum(saved["probs"], axis=1), 1, atol=1e-3)
    assert saved["classes"] == CLASSES
    assert r["spelling"]["runs"] == 20 and "median_chars_per_minute" in r["spelling"]
    assert set(r["mapping"]) == {"select", "left", "right", "back"} and saved["mapping"] == r["mapping"]


def test_typical_performance_reports_median_over_runs():
    from tracks.motor_imagery.speller import typical_performance
    good = typical_performance("لا", heldout(1.0), runs=5)
    assert good["completion_rate"] == 1.0 and good["median_commands"] == good["ideal_commands"]
    bad = typical_performance("أحتاج ماء", heldout(0.25), runs=3, max_commands=40)
    assert bad["completion_rate"] == 0 and bad["median_chars_per_minute"] is None


def confused_feet_heldout(seed=0):
    """حالة الشخص 3 الحقيقية مبسّطة: تخيّل القدمين يُفهم «لسان» غالباً."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(4), 60)
    probs = []
    for t in y:
        p = np.full(4, 0.05)
        if t == 2:  # feet
            p[3 if rng.random() < 0.65 else 2] = 0.85
        else:
            p[t if rng.random() < 0.92 else (t + 1) % 4] = 0.85
        probs.append(p)
    return {"y_true": y.tolist(), "probs": probs}


def test_choose_mapping_moves_select_away_from_unreliable_feet_and_helps_on_new_data():
    from tracks.motor_imagery.speller import choose_mapping, typical_performance
    mapping, search = choose_mapping(confused_feet_heldout(seed=0), runs=6, max_commands=150)
    assert search["chose"] == "ranked" and mapping["select"] != "feet"
    assert search["ranked"]["completion_rate"] > search["default"]["completion_rate"]
    day2 = confused_feet_heldout(seed=1)  # "يوم ثانٍ": محاولات مختلفة بنفس طباع الشخص
    personal = typical_performance("أحتاج ماء", {**day2, "mapping": mapping}, runs=10, max_commands=200)
    default = typical_performance("أحتاج ماء", day2, runs=10, max_commands=200)
    assert personal["completion_rate"] > default["completion_rate"]


def test_choose_mapping_keeps_default_when_all_imagery_is_equally_good():
    from tracks.motor_imagery.speller import CLASS_OF_COMMAND, choose_mapping
    mapping, search = choose_mapping(heldout(0.8), runs=6)
    assert mapping == CLASS_OF_COMMAND and search["chose"] == "default"


def test_ranked_mapping_gives_most_used_command_to_most_reliable_imagery():
    from tracks.motor_imagery.speller import ranked_mapping
    m = ranked_mapping({"left_hand": 0.83, "right_hand": 0.97, "feet": 0.32, "tongue": 0.88})
    assert m == {"select": "right_hand", "left": "tongue", "right": "left_hand", "back": "feet"}
