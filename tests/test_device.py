"""مسار الجهاز: الأسماء والتركيبات، BrainFlow، الإعداد، والمحلّلات المباشرة بلا عتاد."""
import json
import time

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from core.evaluation.report import save_report
from core.signal.montage import bipolar, derivable_pairs, find, key
from core.sources import ReplaySource, SyntheticEEG
from core.sources.base import SignalSource, SourceInfo
from core.sources.brainflow_device import BrainFlowSource, board_spec
from core.sources.replay import save_edf
from server.session import LiveSession
from tracks.epilepsy import device as epi_dev
from tracks.epilepsy.data import CHANNELS as EPI_PAIRS
from tracks.mental_state import device as mental_dev
from tracks.mental_state.pipeline import feature_names, window_features
from tracks.motor_imagery import device as mi_dev
from tracks.motor_imagery import speller
from tracks.motor_imagery.data import CLASSES

CYTON_DAISY = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"]
UNICORN = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
MUSE = ["TP9", "AF7", "AF8", "TP10"]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(session: LiveSession, clock: FakeClock, seconds: float, dt: float = 0.1) -> dict:
    msg = session.step()
    for _ in range(int(round(seconds / dt))):
        clock.t += dt
        msg = session.step()
    return msg


# ---------- الأسماء والتركيبات ----------

def test_old_and_new_10_20_names_match():
    assert key("T3") == key("t7") == "T7"
    assert key("EEG Fp1-REF") == key("FP1") == "FP1"
    assert find(["Fp1", "T5", "Cz"], ["P7", "CZ", "O1"]) == [1, 2, None]


def test_bipolar_is_difference_of_referential_channels():
    data = np.array([[5.0, 6.0], [1.0, 1.0], [2.0, 0.0]])
    out = bipolar(data, ["Fp1", "F7", "T7"], ["FP1-F7", "F7-T7"])
    np.testing.assert_array_equal(out, [[4.0, 5.0], [-1.0, 1.0]])


def test_cyton_daisy_default_wiring_gives_16_of_18_chbmit_channels():
    pairs = derivable_pairs(CYTON_DAISY, EPI_PAIRS)
    assert len(pairs) == 16 and "FZ-CZ" not in pairs and "CZ-PZ" not in pairs


# ---------- BrainFlow ----------

def test_board_specs_come_from_brainflow():
    spec = board_spec("openbci_cyton_daisy")
    assert spec["fs"] == 125 and spec["default_names"] == CYTON_DAISY
    assert board_spec("unicorn")["default_names"] == UNICORN
    assert len(board_spec("openbci_ganglion")["default_names"]) == 4  # بلا أسماء ← EEG1..4


def test_brainflow_synthetic_board_streams_in_microvolts():
    src = BrainFlowSource("synthetic", ch_names=[f"E{i}" for i in range(16)])
    src.start()
    try:
        time.sleep(0.6)
        chunk = src.read_chunk()
    finally:
        src.stop()
    assert chunk.shape[0] == 16 and chunk.shape[1] > 50
    assert src.position_sec == pytest.approx(chunk.shape[1] / 250)
    with pytest.raises(ValueError):
        BrainFlowSource("synthetic", ch_names=["C3"])  # عدد الأسماء ≠ عدد القنوات


# ---------- ماذا يدعم كل جهاز ----------

def test_requirements_follow_the_electrodes(tmp_path, monkeypatch):
    monkeypatch.setattr(mi_dev, "PERSON_MODEL", tmp_path / "none.joblib")
    monkeypatch.setattr(epi_dev, "MODEL_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(mental_dev, "model_path", lambda s: tmp_path / f"{s}.joblib")

    assert mi_dev.requirements(CYTON_DAISY, 125, "openbci_cyton_daisy")["status"] == "needs_calibration"
    assert mi_dev.requirements(MUSE, 256, "muse_2")["status"] == "unsupported"
    assert epi_dev.requirements(CYTON_DAISY)["status"] == "needs_prepare"
    unicorn = epi_dev.requirements(UNICORN)
    assert unicorn["status"] == "unsupported" and unicorn["pairs"] == ["FZ-CZ", "CZ-PZ"]
    assert mental_dev.requirements(UNICORN)["status"] == "needs_prepare"
    assert mental_dev.requirements(MUSE)["status"] == "unsupported"
    focus = mental_dev.scale_requirements("mental_focus", CYTON_DAISY)
    assert "T5" in focus["used"] and "Fz" not in focus["used"]  # P7 في الجهاز = T5 في EEGMAT


def test_mental_features_unchanged_for_full_montage_and_adapt_to_subsets():
    from tracks.mental_state.data import CHANNELS

    rng = np.random.default_rng(0)
    full = window_features(rng.standard_normal((2, 19, 512)), 128, CHANNELS)
    assert full.shape == (2, 19 * 4 + 3) and len(feature_names(CHANNELS)) == 79
    sub = ["Fz", "Cz", "Pz"]  # لا F3/F4 ← بلا عدم تناظر ألفا
    assert window_features(rng.standard_normal((1, 3, 512)), 128, sub).shape == (1, 3 * 4 + 2)
    with pytest.raises(ValueError):
        window_features(rng.standard_normal((1, 2, 512)), 128, ["C3", "C4"])  # لا أمامي ولا خلفي


# ---------- الحالة الذهنية المباشرة ----------

def test_mental_live_analyzer_calibrates_one_minute_then_reads(tmp_path, monkeypatch):
    names = ["Fz", "F3", "F4", "Pz", "O1", "O2"]
    rng = np.random.default_rng(0)
    model = LogisticRegression().fit(rng.standard_normal((40, 27)), np.arange(40) % 2)
    joblib.dump({"model": model, "channels": names, "device_channels": names},
                tmp_path / "mental_focus.joblib")
    monkeypatch.setattr(mental_dev, "model_path", lambda s: tmp_path / f"{s}.joblib")

    clock = FakeClock()
    session = LiveSession(SyntheticEEG(fs=128, ch_names=names, clock=clock))
    session.add_analyzer("mental", mental_dev.live_analyzer(names, 128))
    msg = run(session, clock, 30)
    assert msg["mental"]["calibrating"] and msg["mental"]["remaining_sec"] == 30
    msg = run(session, clock, 40)
    (scale,) = msg["mental"]["scales"]
    assert scale["id"] == "mental_focus" and 0 <= scale["value"] <= 1
    assert mental_dev.live_analyzer(["Fz", "Cz"], 128) is None  # نموذج لأقطاب أخرى لا يُستخدم


# ---------- الكتابة بالدماغ: معايرة ثم كتابة ----------

MI_NAMES = ["C3", "Cz", "C4", "FC3", "FC4", "CP3", "CP4", "Pz"]
# التخيّل يُضعف موجة ميو (10Hz) فوق منطقة الحركة المقابلة
SUPPRESS = {"left_hand": ["C4"], "right_hand": ["C3"], "feet": ["Cz"], "tongue": ["C3", "Cz", "C4"]}


class ImaginingBrain(SignalSource):
    """دماغ اصطناعي "يتخيّل" الأمر الذي يطلبه منه imagined(t) — لنختبر المسار كاملاً بلا إنسان."""

    def __init__(self, fs, clock, seed=0):
        self.info = SourceInfo(fs=fs, ch_names=MI_NAMES, kind="device", description="test")
        self._clock, self._t_last, self._n = clock, None, 0
        self._rng = np.random.default_rng(seed)
        self.imagined = lambda t: None

    def read_chunk(self):
        now = self._clock()
        if self._t_last is None:
            self._t_last = now
        n = int(round((now - self._t_last) * self.info.fs))
        self._t_last = now
        t = (self._n + np.arange(n)) / self.info.fs
        self._n += n
        x = 4.0 * self._rng.standard_normal((len(MI_NAMES), n))
        cls = self.imagined(t[0]) if n else None
        gain = np.ones(len(MI_NAMES))
        if cls:
            gain[[MI_NAMES.index(c) for c in SUPPRESS[cls]]] = 0.15
        phase = np.arange(len(MI_NAMES))[:, None]
        return x + 20.0 * gain[:, None] * np.sin(2 * np.pi * 10.0 * t[None, :] + phase)


def test_calibration_learns_the_person_and_live_speller_types_with_it(tmp_path, monkeypatch):
    monkeypatch.setattr(mi_dev, "PERSON_MODEL", tmp_path / "speller.joblib")
    monkeypatch.setattr(mi_dev, "save_report", lambda mid, r: save_report(mid, r, tmp_path))
    clock = FakeClock()
    brain = ImaginingBrain(250, clock)
    session = LiveSession(brain)
    calib = mi_dev.Calibration(MI_NAMES, 250, "synthetic", per_class=8, seed=1)
    session.add_analyzer("calibration", calib)

    def during_cue(t):
        tr = next((tr for tr in calib.trials if tr["cue"] <= t < tr["end"]), None)
        return tr["class"] if tr else None

    brain.imagined = during_cue
    seen = set()
    msg = session.step()
    while msg["calibration"]["phase"] not in ("done", "error") and clock.t < 400:
        clock.t += 0.1
        msg = session.step()
        seen.add(msg["calibration"]["phase"])
    result = msg["calibration"]["result"]
    assert {"prepare", "imagine", "rest", "training"} <= seen
    assert result["n_trials"] == 32 and result["accuracy"] > 0.8  # الصدفة 25%
    assert len(session.markers) == 32  # علامة لكل أمر في ملف EDF
    bundle = mi_dev.person_model(MI_NAMES, 250, "synthetic")
    assert bundle is not None and mi_dev.person_model(MI_NAMES, 125, "synthetic") is None
    assert json.loads((tmp_path / "device_speller.json").read_text(encoding="utf-8"))["summary"]["n"] == 32

    # الكتابة: "المستخدم" يتخيّل الأمر الذي يقترحه المخطط نحو «لا»
    clock2 = FakeClock()
    brain2 = ImaginingBrain(250, clock2, seed=7)
    live = mi_dev.LiveSpeller(bundle, MI_NAMES, 250, target="لا")
    brain2.imagined = lambda t: (bundle["mapping"][speller.plan(live.state, "لا")]
                                 if live.cue <= t < live.cue + 4 and not speller.is_done(live.state, "لا") else None)
    session2 = LiveSession(brain2)
    session2.add_analyzer("speller", live)
    msg = session2.step()
    while not msg["speller"].get("done") and clock2.t < 25 * 6:
        clock2.t += 0.1
        msg = session2.step()
    assert msg["speller"]["done"] and msg["speller"]["view"]["text"].strip() == "لا"
    assert msg["speller"]["count"] >= speller.ideal_commands("لا")
    assert {c["command"] for c in msg["speller"]["commands"]} == set(speller.COMMANDS)


def test_calibration_schedule_is_balanced_and_shuffled():
    trials = mi_dev.schedule(5, seed=3)
    assert sorted(t["class"] for t in trials) == sorted(CLASSES * 5)
    assert [t["class"] for t in trials] != sorted(t["class"] for t in trials)
    assert all(b["start"] - a["end"] >= 1.5 for a, b in zip(trials, trials[1:]))


# ---------- الصرع بجهازك ----------

def _dummy_epilepsy_bundle(path, names):
    pairs = derivable_pairs(names, EPI_PAIRS)
    rng = np.random.default_rng(0)
    n = len(pairs) * 12
    model = LogisticRegression().fit(rng.standard_normal((40, n)), np.arange(40) % 2)
    joblib.dump({"model": model, "pairs": pairs, "device_channels": names, "fs": 256.0}, path)


def test_epilepsy_live_analyzer_resamples_device_rate_to_model_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(epi_dev, "MODEL_PATH", tmp_path / "epi.joblib")
    _dummy_epilepsy_bundle(epi_dev.MODEL_PATH, CYTON_DAISY)
    assert epi_dev.to_model_rate(np.zeros((2, 500)), 125).shape == (2, 1024)
    clock = FakeClock()
    session = LiveSession(SyntheticEEG(fs=125, ch_names=CYTON_DAISY, clock=clock))
    session.add_analyzer("seizure", epi_dev.live_analyzer(CYTON_DAISY, 125))
    msg = run(session, clock, 12)
    assert 0 <= msg["seizure"]["prob"] <= 1 and msg["seizure"]["device"] is True
    assert epi_dev.live_analyzer(UNICORN, 250) is None


def test_device_recording_can_be_analyzed_on_the_epilepsy_page(tmp_path, monkeypatch):
    from tracks.epilepsy import service

    monkeypatch.setattr(epi_dev, "MODEL_PATH", tmp_path / "epi.joblib")
    _dummy_epilepsy_bundle(epi_dev.MODEL_PATH, CYTON_DAISY)
    data = SyntheticEEG(fs=125, ch_names=CYTON_DAISY).generate(125 * 60)
    path = save_edf(tmp_path / "rec.edf", data, 125, CYTON_DAISY)
    result = service.analyze_edf(path, "rec.edf")
    assert result["duration_sec"] == pytest.approx(60, abs=0.1)
    assert len(result["general"]["times"]) > 20
    with pytest.raises(service.AnalysisError):
        p2 = save_edf(tmp_path / "unicorn.edf", np.zeros((8, 2500)), 250, UNICORN)
        service.analyze_edf(p2, "unicorn.edf")


# ---------- الخادم ----------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from server import device, sources
    from server.main import app

    monkeypatch.setattr(device, "DEVICE_FILE", tmp_path / "device.json")
    monkeypatch.setattr(device, "RECORDINGS_DIR", tmp_path / "recordings")
    monkeypatch.setattr(sources, "DATA_DIR", tmp_path)
    return TestClient(app)


def test_device_config_api_validates_and_lists_the_device(client):
    assert client.get("/api/device").json()["config"] is None
    bad = client.put("/api/device", json={"board": "synthetic", "ch_names": ["C3"] * 16})
    assert bad.status_code == 422 and "مكرر" in bad.json()["detail"]
    assert client.put("/api/device", json={"board": "openbci_cyton"}).status_code == 422  # بلا منفذ
    state = client.put("/api/device", json={"board": "synthetic"}).json()
    assert len(state["config"]["ch_names"]) == 16
    assert [r["track"] for r in state["readiness"]] == ["live", "motor_imagery", "mental_state", "epilepsy"]
    assert client.get("/api/sources").json()[0]["id"] == "device"


def test_device_stream_records_to_edf_that_replays(client, tmp_path):
    client.put("/api/device", json={"board": "synthetic"})
    with client.websocket_connect("/ws?source=device&record=1") as ws:
        hello = ws.receive_json()
        assert hello["kind"] == "device" and hello["recording"].startswith("session_")
        for _ in range(20):
            frame = ws.receive_json()
        assert frame["recording_sec"] > 0.5
    deadline = time.time() + 10
    while not list((tmp_path / "recordings").glob("*.edf")) and time.time() < deadline:
        time.sleep(0.1)
    (path,) = list((tmp_path / "recordings").glob("*.edf"))
    replay = ReplaySource.from_edf(path)
    assert len(replay.info.ch_names) == 16 and replay.info.fs == 250
    ids = [s["id"] for s in client.get("/api/sources").json()]
    assert f"edf:recordings/{path.name}" in ids


def test_tasks_need_a_device_and_a_calibration(client):
    with client.websocket_connect("/ws?source=synthetic&task=mi_calibration") as ws:
        assert "جهاز" in ws.receive_json()["message"]
    client.put("/api/device", json={"board": "synthetic"})
    with client.websocket_connect("/ws?source=device&task=speller") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error" and "معايرة" in msg["message"]
