"""الاتصال بجهاز EEG حقيقي عبر مكتبة BrainFlow.

BrainFlow تدعم أجهزة كثيرة بواجهة واحدة (OpenBCI، Muse، وغيرها)، ولها جهاز
"محاكى" (synthetic) يعمل بلا أي عتاد — نستخدمه الآن لنتأكد أن المسار جاهز.

عندما تشتري جهازاً: غيّر board إلى اسمه وأعطِ serial_port أو mac_address.
"""
import numpy as np

from .base import SignalSource, SourceInfo

# أسماء مختصرة لأشهر الأجهزة ← معرّفها في BrainFlow
BOARDS = {
    "synthetic": "SYNTHETIC_BOARD",
    "openbci_cyton": "CYTON_BOARD",
    "openbci_ganglion": "GANGLION_BOARD",
    "muse_s": "MUSE_S_BOARD",
    "muse_2": "MUSE_2_BOARD",
}


class BrainFlowSource(SignalSource):
    def __init__(self, board: str = "synthetic", serial_port: str = "", mac_address: str = ""):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        self._board_id = getattr(BoardIds, BOARDS[board]).value
        self._shim = BoardShim(self._board_id, params)
        self._eeg_rows = BoardShim.get_eeg_channels(self._board_id)
        try:
            names = BoardShim.get_eeg_names(self._board_id)
        except Exception:  # بعض الأجهزة لا تعرّف أسماء القنوات
            names = [f"EEG{i + 1}" for i in range(len(self._eeg_rows))]
        self.info = SourceInfo(
            fs=float(BoardShim.get_sampling_rate(self._board_id)),
            ch_names=list(names),
            kind="synthetic" if board == "synthetic" else "device",
            description=f"BrainFlow: {board}",
        )

    def start(self) -> None:
        self._shim.prepare_session()
        self._shim.start_stream()

    def stop(self) -> None:
        if self._shim.is_prepared():
            self._shim.stop_stream()
            self._shim.release_session()

    def read_chunk(self) -> np.ndarray:
        # get_board_data تعيد كل ما تجمّع في الذاكرة منذ آخر قراءة ثم تفرغها
        data = self._shim.get_board_data()
        return data[self._eeg_rows, :]  # BrainFlow يعطي EEG بالميكروفولت أصلاً
