import numpy as np
from fastapi.testclient import TestClient

from core.sources import SyntheticEEG
from server.main import app
from server.session import LiveSession

client = TestClient(app)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_health_and_sources():
    assert client.get("/api/health").json() == {"ok": True}
    ids = [s["id"] for s in client.get("/api/sources").json()]
    assert "synthetic" in ids


def test_live_session_frames_have_display_samples_bands_and_quality():
    clock = FakeClock()
    session = LiveSession(SyntheticEEG(fs=256, clock=clock))
    hello = session.hello()
    assert hello["display_fs"] == 128
    msg = session.step()  # أول خطوة تبدأ ساعة المصدر عند الزمن 0 (بلا عيّنات بعد)
    total = len(msg["samples"][0])
    for _ in range(25):  # 2.5 ثانية من الإطارات
        clock.t += 0.1
        msg = session.step()
        total += len(msg["samples"][0])
    assert len(msg["samples"]) == len(hello["ch_names"])
    assert abs(total - 2.5 * 128) <= 1  # التخفيض منتظم عبر القطع
    assert len(msg["bands"]) == len(hello["ch_names"])
    assert np.allclose(np.sum(msg["bands"], axis=1), 1, atol=1e-3)
    assert set(msg["quality"]) <= {"good", "flat", "noisy", "saturated"}


def test_websocket_sends_hello_then_frames():
    with client.websocket_connect("/ws?source=synthetic") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert "O1" in hello["ch_names"]
        frame = ws.receive_json()
        assert frame["type"] == "frame"


def test_websocket_rejects_unknown_source_and_paths_outside_data_dir():
    for bad in ["nope", "edf:../../secret.edf"]:
        with client.websocket_connect(f"/ws?source={bad}") as ws:
            assert ws.receive_json()["type"] == "error"


def test_speller_endpoints(tmp_path, monkeypatch):
    import json
    from tracks.motor_imagery import service as sp

    monkeypatch.setattr(sp, "HELDOUT_DIR", tmp_path)
    monkeypatch.setattr(sp, "load_report", lambda _id: None)
    (tmp_path / "subject_3.json").write_text(json.dumps(
        {"y_true": [0, 1, 2, 3], "probs": [[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7]]}),
        encoding="utf-8")

    info = client.get("/api/speller/info").json()
    assert len(info["commands"]) == 4 and info["initial"]["text"] == ""
    assert [s["subject"] for s in client.get("/api/speller/subjects").json()] == [3]

    r = client.post("/api/speller/simulate", json={"subject": 3, "target": "لا"}).json()
    assert r["finished"] and r["steps"][-1]["view"]["text"].strip() == "لا"
    assert client.post("/api/speller/simulate", json={"subject": 3, "target": "hello"}).status_code == 422
    assert client.post("/api/speller/simulate", json={"subject": 9, "target": "لا"}).status_code == 422

    v = client.post("/api/speller/step", json={"state": info["initial"], "command": "select"}).json()
    assert v["group"] == 0
    assert client.post("/api/speller/step", json={"state": v, "command": "jump"}).status_code == 422
