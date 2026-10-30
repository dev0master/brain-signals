"""الاتصال بجهاز EEG حقيقي عبر مكتبة BrainFlow.

BrainFlow تدعم أجهزة كثيرة بواجهة واحدة (OpenBCI، g.tec Unicorn، Muse، وغيرها)، ولها جهاز
"محاكى" (synthetic) يعمل بلا أي عتاد — نجرّب به مسار الجهاز كاملاً قبل شرائه.

الإعداد (أي جهاز، أي منفذ، وأي قطب على أي قناة) يُحفظ من صفحة «جهازك» — انظر server/device.py.
"""
import time

import numpy as np

from .base import SignalSource, SourceInfo

# كل جهاز: معرّفه في BrainFlow، اسمه للعرض، وطريقة توصيله (ما يحتاج المستخدم أن يكتبه)
BOARDS: dict[str, dict] = {
    "synthetic": {"brainflow": "SYNTHETIC_BOARD", "label": "جهاز BrainFlow محاكى (للتجربة بلا عتاد)",
                  "connection": "none"},
    "openbci_cyton_daisy": {"brainflow": "CYTON_DAISY_BOARD", "label": "OpenBCI Cyton + Daisy — 16 قطباً",
                            "connection": "serial"},
    "openbci_cyton": {"brainflow": "CYTON_BOARD", "label": "OpenBCI Cyton — 8 أقطاب", "connection": "serial"},
    "unicorn": {"brainflow": "UNICORN_BOARD", "label": "g.tec Unicorn Hybrid Black — 8 أقطاب",
                "connection": "bluetooth"},
    "openbci_ganglion": {"brainflow": "GANGLION_BOARD", "label": "OpenBCI Ganglion — 4 أقطاب",
                         "connection": "serial"},
    "muse_2": {"brainflow": "MUSE_2_BOARD", "label": "Muse 2 — 4 أقطاب", "connection": "bluetooth"},
    "muse_s": {"brainflow": "MUSE_S_BOARD", "label": "Muse S — 4 أقطاب", "connection": "bluetooth"},
}


class DeviceError(RuntimeError):
    """تعذّر الاتصال بالجهاز — رسالة مفهومة للمستخدم."""


def _board_id(board: str) -> int:
    from brainflow.board_shim import BoardIds

    if board not in BOARDS:
        raise ValueError(f"جهاز غير معروف: {board}")
    return getattr(BoardIds, BOARDS[board]["brainflow"]).value


def board_spec(board: str) -> dict:
    """معدل العيّنات وعدد قنوات EEG وأسماؤها الافتراضية — من BrainFlow نفسها، بلا اتصال."""
    from brainflow.board_shim import BoardShim

    BoardShim.disable_board_logger()
    bid = _board_id(board)
    n = len(BoardShim.get_eeg_channels(bid))
    try:
        names = list(BoardShim.get_eeg_names(bid))
    except Exception:  # بعض الأجهزة (Ganglion) لا تعرّف أسماء القنوات — يسمّيها المستخدم
        names = [f"EEG{i + 1}" for i in range(n)]
    return {"fs": float(BoardShim.get_sampling_rate(bid)), "n_channels": n, "default_names": names}


class BrainFlowSource(SignalSource):
    def __init__(self, board: str = "synthetic", serial_port: str = "", mac_address: str = "",
                 ch_names: list[str] | None = None):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams

        BoardShim.disable_board_logger()
        spec = board_spec(board)
        names = list(ch_names) if ch_names else spec["default_names"]
        if len(names) != spec["n_channels"]:
            raise ValueError(f"الجهاز له {spec['n_channels']} قناة لكن أُعطيت {len(names)} اسماً")
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        self.board = board
        self._board_id = _board_id(board)
        self._shim = BoardShim(self._board_id, params)
        self._eeg_rows = BoardShim.get_eeg_channels(self._board_id)
        self._n_read = 0
        self.info = SourceInfo(
            fs=spec["fs"],
            ch_names=names,
            kind="device",
            description=BOARDS[board]["label"],
        )

    def start(self) -> None:
        try:
            # عند الانتقال بين الصفحات قد يصل الاتصال الجديد قبل أن يحرّر القديم الجهاز — ننتظر قليلاً
            for attempt in range(8):
                try:
                    self._shim.prepare_session()
                    break
                except Exception as exc:
                    if "ANOTHER_BOARD" not in str(exc) or attempt == 7:
                        raise
                    time.sleep(0.5)
            self._shim.start_stream()
        except Exception as exc:
            hint = "" if BOARDS[self.board]["connection"] == "none" else (
                " — تأكد أن الجهاز مشغّل ومشحون، وأن المنفذ صحيح في صفحة «جهازك»"
                + (" (مثل COM3)" if BOARDS[self.board]["connection"] == "serial" else ""))
            raise DeviceError(f"تعذّر الاتصال بـ {BOARDS[self.board]['label']}{hint}. ({exc})") from exc

    def stop(self) -> None:
        if self._shim.is_prepared():
            self._shim.stop_stream()
            self._shim.release_session()

    def read_chunk(self) -> np.ndarray:
        # get_board_data تعيد كل ما تجمّع في الذاكرة منذ آخر قراءة ثم تفرغها
        data = self._shim.get_board_data()
        eeg = data[self._eeg_rows, :]  # BrainFlow يعطي EEG بالميكروفولت أصلاً
        self._n_read += eeg.shape[1]
        return eeg

    @property
    def position_sec(self) -> float:
        return self._n_read / self.info.fs
