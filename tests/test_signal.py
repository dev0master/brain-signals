import numpy as np
import pytest

from core.signal.filters import StreamFilter, apply_causal, bandpass_sos, make_chain
from core.signal.quality import channel_quality
from core.signal.reference import apply_matrix, car, laplacian_matrix, neighbors_from_montage
from core.signal.windows import RingBuffer, sliding_windows

FS = 256.0


def sine(freq, sec=10.0, amp=10.0, fs=FS):
    t = np.arange(int(sec * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def steady_amplitude(x, fs=FS):
    """سعة الإشارة بعد تجاهل أول ثانيتين (فترة استقرار الفلتر)."""
    return np.abs(x[int(2 * fs):]).max()


def test_bandpass_keeps_10hz_and_chain_removes_50hz():
    chain = make_chain(FS, low=8, high=30, mains_hz=50)
    passed = apply_causal(chain, sine(10)[None, :])[0]
    removed = apply_causal(chain, sine(50)[None, :])[0]
    assert steady_amplitude(passed) > 0.9 * 10
    assert steady_amplitude(removed) < 0.05 * 10


def test_bandpass_rejects_slow_drift():
    sos = bandpass_sos(8, 30, FS)
    out = apply_causal(sos, sine(1)[None, :])[0]
    assert steady_amplitude(out) < 0.1 * 10


def test_stream_filter_in_chunks_equals_whole_recording():
    """البث قطعةً قطعة يجب أن يعطي نفس نتيجة الفلترة على التسجيل كاملاً."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 2000))
    chain = make_chain(FS)
    whole = apply_causal(chain, data)
    sf = StreamFilter(chain, 3)
    pieces = [sf.process(data[:, i : i + 37]) for i in range(0, 2000, 37)]
    np.testing.assert_allclose(np.hstack(pieces), whole, atol=1e-9)


def test_car_makes_channel_sum_zero():
    data = np.random.default_rng(2).standard_normal((5, 100)) + 7.0
    np.testing.assert_allclose(car(data).sum(axis=0), 0, atol=1e-9)


def test_laplacian_subtracts_mean_of_neighbors():
    names = ["A", "B", "C", "D"]
    m = laplacian_matrix(names, {"A": ["B", "C"], "Z": ["A"]})  # Z غير موجودة فتُتجاهل
    data = np.array([[10.0], [2.0], [4.0], [99.0]])
    out = apply_matrix(m, data)
    assert out[0, 0] == pytest.approx(10 - (2 + 4) / 2)
    np.testing.assert_allclose(out[1:, 0], data[1:, 0])  # بلا جيران ← بلا تغيير


def test_laplacian_is_not_just_sign_flip():
    """الخطأ الذي وجدناه في Data1a.ipynb: np.eye(n) * -1 يعكس الإشارة فقط."""
    names = ["A", "B", "C"]
    data = np.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])  # إشارة مشتركة بين كل القنوات
    lap = apply_matrix(laplacian_matrix(names, {"A": ["B", "C"]}), data)
    wrong = apply_matrix(np.eye(3) * -1, data)
    assert np.allclose(lap[0], 0)        # الصحيح يزيل المشترك
    assert np.allclose(wrong[0], -data[0])  # الخاطئ يبقيه معكوساً


def test_neighbors_from_montage_finds_motor_neighbors():
    names = ["C3", "C1", "C5", "FC3", "CP3", "Cz", "O1"]
    nb = neighbors_from_montage(names, k=4)
    assert set(nb["C3"]) == {"C1", "C5", "FC3", "CP3"}


def test_sliding_windows_shapes_and_starts():
    data = np.arange(3 * 1000, dtype=float).reshape(3, 1000)
    w, starts = sliding_windows(data, fs=100, win_sec=2, step_sec=1)
    assert w.shape == (9, 3, 200)
    np.testing.assert_allclose(starts[:3], [0, 1, 2])
    np.testing.assert_array_equal(w[1, 0], data[0, 100:300])


def test_ring_buffer_keeps_latest_samples():
    rb = RingBuffer(2, 5)
    rb.push(np.array([[1, 2, 3], [1, 2, 3]], dtype=float))
    assert not rb.full
    rb.push(np.array([[4, 5, 6, 7], [4, 5, 6, 7]], dtype=float))
    assert rb.full
    np.testing.assert_array_equal(rb.latest(5)[0], [3, 4, 5, 6, 7])


def test_channel_quality_labels():
    rng = np.random.default_rng(3)
    good = 20 * rng.standard_normal(512)
    flat = 0.01 * rng.standard_normal(512)
    noisy = 400 * rng.standard_normal(512) * 0.5
    saturated = good.copy()
    saturated[100] = 5000
    assert channel_quality(np.vstack([good, flat, noisy, saturated])) == [
        "good", "flat", "noisy", "saturated"]
