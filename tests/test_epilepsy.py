import joblib
import mne
import numpy as np
import pytest

from core.evaluation.metrics import Event, event_metrics
from tracks.epilepsy import service
from tracks.epilepsy.data import CHANNELS, Recording, choose_files, parse_summary, select_channels
from tracks.epilepsy.pipeline import (WIN_SEC, decide, fit_balanced, recording_features,
                                      window_labels)

SUMMARY = """
File Name: chb01_01.edf
Number of Seizures in File: 0

File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb04_08.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 100 seconds
Seizure 1 End Time: 150 seconds
Seizure 2 Start Time: 900 seconds
Seizure 2 End Time: 990 seconds
"""

FS = 256.0


def test_parse_summary_both_formats():
    recs = parse_summary(SUMMARY, "chb01")
    assert [r.file for r in recs] == ["chb01_01.edf", "chb01_03.edf", "chb04_08.edf"]
    assert recs[0].seizures == []
    assert recs[1].seizures == [(2996.0, 3036.0)]
    assert recs[2].seizures == [(100.0, 150.0), (900.0, 990.0)]


def test_choose_files_keeps_all_seizure_files_and_spreads_free_ones():
    recs = [Recording("p", f"p_{i:02d}.edf", [(1, 2)] if i in (3, 7) else []) for i in range(20)]
    chosen = [r.file for r in choose_files(recs, n_free=3)]
    assert "p_03.edf" in chosen and "p_07.edf" in chosen
    assert len(chosen) == 5
    assert "p_00.edf" in chosen and "p_19.edf" in chosen  # موزّعة من البداية للنهاية


def test_select_channels_handles_mne_duplicate_suffixes():
    names = [c for c in CHANNELS if c != "T8-P8"] + ["T8-P8-0", "T8-P8-1", "--0"]
    idx = select_channels(names)
    assert idx is not None
    assert names[idx[CHANNELS.index("T8-P8")]] == "T8-P8-0"
    assert select_channels(CHANNELS[:-1]) is None


def test_window_labels_by_window_center():
    starts = np.arange(0, 20, 2.0)  # نوافذ 4 ثوان، مراكزها 2,4,6,...
    y = window_labels(starts, [(5.0, 9.0)])
    assert list(np.flatnonzero(y)) == [2, 3]  # المركزان 6 و 8


def fake_recording(seconds: float, seizure: tuple[float, float] | None, seed: int) -> np.ndarray:
    """EEG اصطناعي: ضوضاء منخفضة التردد، ونوبة = موجات إيقاعية كبيرة 4Hz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    from core.signal.filters import apply_causal, bandpass_sos
    x = apply_causal(bandpass_sos(0.5, 20, FS), rng.standard_normal((len(CHANNELS), n)))
    x *= 20 / x.std()
    if seizure:
        t = np.arange(n) / FS
        mask = (t >= seizure[0]) & (t <= seizure[1])
        x[:, mask] += 150 * np.sin(2 * np.pi * 4 * t[mask])
    return x


def test_pipeline_learns_rhythmic_bursts_and_detects_unseen_one():
    """من البداية للنهاية: تدريب على تسجيلين، اختبار على ثالث لم يره النموذج."""
    train = [(fake_recording(300, (100, 140), 1), (100, 140)),
             (fake_recording(300, (200, 230), 2), (200, 230))]
    X, y = [], []
    for data, sz in train:
        feats, starts = recording_features(data, FS)
        X.append(feats)
        y.append(window_labels(starts, [sz]))
    model = fit_balanced(np.vstack(X), np.concatenate(y))

    test = fake_recording(300, (150, 190), 3)
    feats, starts = recording_features(test, FS)
    detected = decide(starts, model.predict_proba(feats)[:, 1])
    m = event_metrics([Event(150, 190)], detected, total_hours=300 / 3600)
    assert m["detected"] == 1
    assert m["false_alarms"] == 0
    assert 0 <= m["mean_latency_sec"] <= 3 * WIN_SEC


def write_edf(path, data_uv: np.ndarray, ch_names: list[str]) -> None:
    info = mne.create_info(ch_names, FS, ch_types="eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose="error")
    mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose="error")


def test_analyze_edf_rejects_wrong_montage(tmp_path, monkeypatch):
    model_path = tmp_path / "m.joblib"
    joblib.dump({"model": None, "train_files": []}, model_path)
    monkeypatch.setattr(service, "GENERAL_MODEL", model_path)
    edf = tmp_path / "x.edf"
    write_edf(edf, np.random.default_rng(0).standard_normal((3, 2560)) * 20, ["Fp1", "Cz", "O1"])
    with pytest.raises(service.AnalysisError, match="18"):
        service.analyze_edf(edf, "x.edf")


def test_analyze_edf_runs_general_model_and_warns_on_training_file(tmp_path, monkeypatch):
    data = fake_recording(120, (40, 70), 4)
    feats, starts = recording_features(data, FS)
    model = fit_balanced(feats, window_labels(starts, [(40, 70)]))
    model_path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "train_files": ["chb99_01.edf"]}, model_path)
    monkeypatch.setattr(service, "GENERAL_MODEL", model_path)
    monkeypatch.setattr(service, "load_report", lambda _id: None)  # بلا تقرير موثوقية
    service._general_bundle.cache_clear()
    edf = tmp_path / "upload.edf"
    write_edf(edf, data, CHANNELS)

    result = service.analyze_edf(edf, "chb99_01.edf")
    assert len(result["general"]["probs"]) == len(starts)
    assert result["general"]["detected"]
    assert any("بيانات تدريب" in w for w in result["warnings"])  # الملف كان ضمن بيانات التدريب
    assert not service.analyze_edf(edf, "new_patient.edf")["warnings"]


def test_analyze_edf_warns_with_measured_general_model_reliability(tmp_path, monkeypatch):
    data = fake_recording(60, (20, 40), 5)
    feats, starts = recording_features(data, FS)
    model_path = tmp_path / "m.joblib"
    joblib.dump({"model": fit_balanced(feats, window_labels(starts, [(20, 40)])), "train_files": []}, model_path)
    monkeypatch.setattr(service, "GENERAL_MODEL", model_path)
    service._general_bundle.cache_clear()
    monkeypatch.setattr(service, "load_report", lambda _id: {"seizure_events": {
        "detected": 6, "n_seizures": 19, "false_alarms_per_hour": 7.23}})
    edf = tmp_path / "u.edf"
    write_edf(edf, data, CHANNELS)
    [warning] = service.analyze_edf(edf, "someone.edf")["warnings"]
    assert "6 من 19" in warning and "7.2" in warning


def test_live_analyzer_reads_held_out_probability_at_current_time(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    oof = tmp_path / "oof"
    oof.mkdir()
    (oof / "chb99_01.edf.json").write_text(json.dumps({
        "file": "chb99_01.edf",
        "patient_specific": {"times": [4.0, 6.0, 8.0, 10.0], "probs": [0.1, 0.2, 0.9, 0.95],
                             "detected": [{"start": 6.0, "end": 14.0, "alarm": 10.0}], "threshold": 0.5},
    }), encoding="utf-8")
    monkeypatch.setattr(service, "OOF_DIR", oof)
    analyze = service.live_analyzer(tmp_path / "chb99_01.edf")

    def at(t):
        return analyze(SimpleNamespace(source=SimpleNamespace(position_sec=t)))

    assert at(3.0) is None                        # لا قرار قبل اكتمال أول نافذة
    assert at(7.0)["prob"] == 0.2                 # آخر قرار متاح حتى هذه اللحظة
    assert at(8.5)["alarm"] is False              # الاحتمال عالٍ لكن الإنذار لم يُطلق بعد
    assert at(10.5) == {"prob": 0.95, "alarm": True, "threshold": 0.5}
    assert service.live_analyzer(tmp_path / "unknown.edf") is None


def test_plan_download_picks_seizure_files_plus_free_hours(monkeypatch):
    """يغطي المسار الذي يستخدمه scripts/download_data.py."""
    from tracks.epilepsy import data as chb
    import scripts.download_data  # noqa: F401 — يفشل الاختبار إن انكسر استيراد السكربت

    monkeypatch.setattr(chb, "fetch_summary", lambda p: parse_summary(SUMMARY, p))
    plan = chb.plan_download(["chb01"], n_free=1)
    assert [r.file for r in plan] == ["chb01_01.edf", "chb01_03.edf", "chb04_08.edf"]
