import numpy as np
import pytest

from core.evaluation.metrics import (Event, classification_summary, event_metrics,
                                     windows_to_events)
from core.evaluation.splits import LeakageError, assert_no_leak, group_splits
from core.features.bandpower import (BAND_NAMES, band_powers, frontal_alpha_asymmetry,
                                     relative_band_powers)
from core.features.seizure import PER_CHANNEL, feature_names, window_features
from core.signal.filters import apply_causal, bandpass_sos

FS = 256.0


def test_alpha_sine_dominates_alpha_band():
    t = np.arange(int(4 * FS)) / FS
    x = 10 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.default_rng(0).standard_normal(t.size)
    rel = relative_band_powers(x[None, :], FS)[0]
    assert BAND_NAMES[int(np.argmax(rel))] == "alpha"
    assert rel[BAND_NAMES.index("alpha")] > 0.8


def test_band_powers_shape_for_windows():
    w = np.random.default_rng(1).standard_normal((7, 3, 512))
    assert band_powers(w, FS).shape == (7, 3, len(BAND_NAMES))


def test_frontal_alpha_asymmetry_sign_and_missing_channels():
    names = ["F3", "F4", "Cz"]
    assert frontal_alpha_asymmetry(np.array([1.0, 2.0, 5.0]), names) > 0  # يمين أقوى
    assert frontal_alpha_asymmetry(np.array([1.0, 2.0]), ["Cz", "Pz"]) is None


def test_seizure_features_shape_and_line_length_rises_with_rhythmic_burst():
    rng = np.random.default_rng(2)
    t = np.arange(int(2 * FS)) / FS
    # خلفية تشبه EEG: ضوضاء منخفضة التردد (الضوضاء البيضاء الخالصة تعرّجها عالٍ جداً)
    quiet = apply_causal(bandpass_sos(0.5, 20, FS), rng.standard_normal((2, t.size)))
    quiet *= 10 / quiet.std()
    burst = quiet + 150 * np.sin(2 * np.pi * 5 * t)
    feats = window_features(np.stack([quiet, burst]), FS)
    assert feats.shape == (2, 2 * len(PER_CHANNEL))
    ll = PER_CHANNEL.index("line_length")
    assert feats[1, ll] > 3 * feats[0, ll]
    assert len(feature_names(["A", "B"])) == feats.shape[1]


def test_group_splits_never_share_groups():
    groups = np.repeat(["s1", "s2", "s3", "s4"], 10)
    n = 0
    for tr, te in group_splits(groups):
        assert set(groups[tr]).isdisjoint(groups[te])
        n += 1
    assert n == 4


def test_leak_guard_raises():
    groups = np.array(["a", "a", "b", "b"])
    with pytest.raises(LeakageError):
        assert_no_leak(groups, np.array([0, 2]), np.array([1, 3]))


def test_classification_summary_reports_chance():
    s = classification_summary([0, 0, 0, 1], [0, 0, 0, 0])
    assert s["accuracy"] == 0.75
    assert s["chance_majority"] == 0.75  # الدقة لم تتجاوز "اختيار الفئة الأكثر دائماً"
    assert s["chance_uniform"] == 0.5
    assert s["balanced_accuracy"] == 0.5


def test_k_of_n_smoothing_ignores_single_spike():
    starts = np.arange(20, dtype=float)
    pos = np.zeros(20, bool)
    pos[3] = True                 # نافذة شاذة واحدة ← لا حدث
    pos[10:15] = True             # 5 نوافذ متتالية ← حدث واحد
    events = windows_to_events(starts, pos, win_sec=1.0, k=3, n=4)
    assert len(events) == 1
    assert events[0].start == 10.0
    assert events[0].end == 15.0


def test_event_metrics_sensitivity_false_alarms_latency():
    truth = [Event(100, 160), Event(1000, 1050)]
    detected = [Event(108, 150), Event(3000, 3010)]  # يكتشف الأولى بعد 8 ثوان + إنذار كاذب
    m = event_metrics(truth, detected, total_hours=2.0)
    assert m["sensitivity"] == 0.5
    assert m["false_alarms"] == 1
    assert m["false_alarms_per_hour"] == 0.5
    assert m["mean_latency_sec"] == 8.0
