import numpy as np
import pytest

from tracks.mental_state import data as eegmat
from tracks.mental_state import train
from tracks.mental_state.pipeline import (CHANNELS, calibrate, feature_names, recording_features,
                                          window_features)

FS = 250.0


def eeg(seconds: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    """EEG اصطناعي: ضوضاء + ألفا 10Hz خلفية + بيتا 20Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = 5 * rng.standard_normal((len(CHANNELS), t.size))
    posterior = [CHANNELS.index(c) for c in ("O1", "O2", "P3", "P4", "Pz")]
    x[posterior] += alpha * np.sin(2 * np.pi * 10 * t)
    x += beta * np.sin(2 * np.pi * 20 * t + 1.0)
    return x


def test_window_features_shape_and_names():
    w = eeg(8, 20, 5, 0)[None, :, : int(4 * FS)]
    f = window_features(w, FS)
    assert f.shape == (1, len(feature_names())) == (1, 19 * 4 + 3)


def test_alpha_beta_ratio_drops_when_alpha_drops():
    names = feature_names()
    i = names.index("ألفا/بيتا خلفياً")
    relaxed, _ = recording_features(eeg(20, 30, 5, 1), FS)
    busy, _ = recording_features(eeg(20, 5, 10, 2), FS)
    assert relaxed[:, i].mean() > busy[:, i].mean() + 1


def test_recording_features_skip_filter_settling_period():
    _, starts = recording_features(eeg(20, 20, 5, 3), FS)
    assert starts.min() >= 4.0


def test_calibrate_centers_baseline():
    rng = np.random.default_rng(0)
    base = rng.normal(5, 2, (30, 4))
    np.testing.assert_allclose(calibrate(base, base).mean(axis=0), 0, atol=1e-9)


def fake_subject(i: int, offset: float, rng) -> train.SubjectData:
    """كل شخص له إزاحة خاصة في كل الميزات (مثل اختلاف سُمك الجمجمة) — والمهمة تغيّر ميزتين."""
    n_feat = 6
    shift = offset * rng.standard_normal(n_feat)

    def windows(n, task):
        x = rng.standard_normal((n, n_feat)) + shift
        if task:
            x[:, 0] -= 1.0
            x[:, 1] += 1.0
        return x

    return train.SubjectData(
        subject=f"S{i}", X=np.vstack([windows(40, False), windows(20, True)]),
        y=np.r_[np.zeros(40, int), np.ones(20, int)], t=np.arange(60.0), baseline=windows(25, False))


def test_cross_subject_evaluation_and_why_calibration_helps():
    rng = np.random.default_rng(0)
    subjects = [fake_subject(i, offset=3.0, rng=rng) for i in range(12)]
    raw = train.summarize(subjects, train.cross_subject_predictions(subjects, calibrated=False), ["هادئ", "مهمة"])
    cal = train.summarize(subjects, train.cross_subject_predictions(subjects, calibrated=True), ["هادئ", "مهمة"])
    assert cal["auc"] > 0.8
    assert cal["auc"] > raw["auc"] + 0.05  # الفروق بين الناس تطغى على أثر المهمة بلا معايرة
    assert cal["subjects_rising"] == 1.0
    assert len(cal["per_subject"]) == 12


@pytest.mark.skipif(not eegmat.path_of(0, 2).exists(), reason="EEGMAT غير منزّلة")
def test_eegmat_subject_timeline_has_calibration_then_rest_then_task():
    s = train.eegmat_subject(0)
    kinds = [seg["kind"] for seg in s.segments]
    assert kinds == ["calibration", "calm", "task"]
    assert s.t.min() > train.CALIB_SEC  # دقيقة المعايرة ليست في الاختبار
    assert s.y[0] == 0 and s.y[-1] == 1
    assert np.all(np.diff(s.t) > 0)
    assert len(s.baseline) >= 25


def test_service_timeline_and_live_analyzer(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tracks.mental_state import service

    monkeypatch.setattr(service, "OOF_DIR", tmp_path)
    (tmp_path / "mental_focus_Subject07.json").write_text(json.dumps({
        "subject": "Subject07",
        "segments": [{"kind": "calibration", "label": "معايرة", "start": 0, "end": 60},
                     {"kind": "calm", "label": "راحة", "start": 60, "end": 180},
                     {"kind": "task", "label": "حساب ذهني", "start": 180, "end": 240}],
        "times": [70.0, 100.0, 190.0, 200.0], "calibrated": [0.2, 0.3, 0.8, 0.9], "raw": [0.5, 0.5, 0.6, 0.6],
    }), encoding="utf-8")
    tl = service.timeline("mental_focus", "Subject07")
    assert tl["sources"][1]["source_id"].endswith("Subject07_2.edf")
    assert service.timeline("mental_focus", "../secret") is None
    assert service.timeline("nope", "Subject07") is None

    at = lambda path, t: service.live_analyzer(tmp_path / path)(SimpleNamespace(source=SimpleNamespace(position_sec=t)))
    assert at("Subject07_1.edf", 30)["calibrating"] is True       # أول دقيقة: معايرة
    assert at("Subject07_1.edf", 105)["value"] == 0.3 and at("Subject07_1.edf", 105)["truth"] == "راحة"
    r = at("Subject07_2.edf", 15)                                  # 15ث داخل المهمة = 195 على الخط الموحّد
    assert r["value"] == 0.8 and r["truth"] == "حساب ذهني" and r["high"] == "مركّز"
    assert service.live_analyzer(tmp_path / "chb01_03.edf") is None


def test_rating_agreement_detects_relationship_and_its_absence():
    rng = np.random.default_rng(0)

    def subject(follow: bool, i: int):
        ratings = rng.integers(1, 11, 9).astype(float)
        trial = np.repeat([f"T/{k}" for k in range(9)], 5)
        rating = np.repeat(ratings, 5)
        p = (rating / 10 if follow else rng.random(9).repeat(5)) + 0.02 * rng.standard_normal(45)
        sd = train.SubjectData(subject=f"S{i}", X=np.zeros((45, 1)), y=np.ones(45, int), t=np.arange(45.0),
                               baseline=np.zeros((2, 1)), rating=rating, trial=trial)
        return sd, p

    pairs = [subject(True, i) for i in range(10)]
    ag = train.rating_agreement([s for s, _ in pairs], [p for _, p in pairs])
    assert ag["spearman_pooled"] > 0.9 and ag["mean_within_subject"] > 0.9 and ag["n_trials"] == 90

    pairs = [subject(False, i) for i in range(10)]
    ag = train.rating_agreement([s for s, _ in pairs], [p for _, p in pairs])
    assert abs(ag["spearman_pooled"]) < 0.3
    assert train.rating_agreement([fake_subject(0, 0, np.random.default_rng(1))], [np.zeros(60)]) is None


from tracks.mental_state import sam40  # noqa: E402


@pytest.mark.skipif(not sam40.available_subjects(), reason="SAM40 غير مستخرجة")
def test_sam40_subject_layout():
    s = train.sam40_subject(1, sam40.ratings())
    kinds = [seg["kind"] for seg in s.segments]
    assert kinds == ["calibration", "calm", "calm"] + ["task"] * 9
    assert s.X.shape[1] == len(sam40.CHANNELS) * 4 + 3
    assert np.isnan(s.rating[s.y == 0]).all()           # لا تقييم للاسترخاء
    assert not np.isnan(s.rating[s.y == 1]).any()       # كل محاولة مهمة مقيّمة
    assert len(s.baseline) > 0 and s.t.min() > s.segments[0]["end"]


def test_validity_notes_are_generated_from_numbers():
    feats = [{"feature": "ألفا عند Fp1"}, {"feature": "دلتا عند Cz"}, {"feature": "دلتا عند Fp2"}]
    notes = train.validity_notes({"rating_agreement": {"spearman_pooled": -0.18, "n_trials": 360,
                                                       "mean_within_subject": -0.04}}, feats)
    assert any("حركة العين" in n and "Fp1" in n and "Fp2" in n for n in notes)
    assert any("لا يتبع شدة التوتر" in n and "-0.04" in n for n in notes)
    assert train.validity_notes({"rating_agreement": None}, [{"feature": "ثيتا عند Cz"}]) == []
