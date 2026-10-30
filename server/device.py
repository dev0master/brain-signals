"""إعداد جهازك: أي جهاز، أي منفذ، وأي قطب على أي قناة — وماذا يستطيع النظام أن يفعل بأقطابه.

الإعداد يُحفظ في DATA_DIR/device.json (خارج المشروع: هو خاص بجهازك لا بالكود).
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from core.config import DATA_DIR
from core.signal.montage import STANDARD_10_20, key
from core.sources.brainflow_device import BOARDS, BrainFlowSource, board_spec
from tracks.epilepsy import device as epilepsy_device
from tracks.mental_state import device as mental_device
from tracks.motor_imagery import device as mi_device

DEVICE_FILE = DATA_DIR / "device.json"
RECORDINGS_DIR = DATA_DIR / "recordings"


class ConfigError(ValueError):
    """إعداد غير صالح — رسالة مفهومة للمستخدم."""


def catalog() -> list[dict]:
    out = []
    for board_id, meta in BOARDS.items():
        out.append({"id": board_id, "label": meta["label"], "connection": meta["connection"], **board_spec(board_id)})
    return out


def load_config() -> dict | None:
    if not DEVICE_FILE.exists():
        return None
    return json.loads(DEVICE_FILE.read_text(encoding="utf-8"))


def validate(cfg: dict) -> dict:
    board = cfg.get("board")
    if board not in BOARDS:
        raise ConfigError(f"جهاز غير معروف: {board}")
    spec = board_spec(board)
    names = [str(n).strip() for n in (cfg.get("ch_names") or spec["default_names"])]
    if len(names) != spec["n_channels"]:
        raise ConfigError(f"هذا الجهاز له {spec['n_channels']} قناة — اكتب اسم قطب لكل قناة")
    if any(not n for n in names):
        raise ConfigError("اكتب اسم القطب لكل قناة (مثل C3)")
    keys = [key(n) for n in names]
    if len(set(keys)) != len(keys):
        dup = sorted({n for n in names if keys.count(key(n)) > 1})
        raise ConfigError(f"اسم مكرر: {'، '.join(dup)} — كل قناة تقيس قطباً مختلفاً")
    serial_port = str(cfg.get("serial_port") or "").strip()
    if BOARDS[board]["connection"] == "serial" and not serial_port:
        raise ConfigError("اكتب منفذ الجهاز (مثل COM3) — تجده في «إدارة الأجهزة» تحت Ports بعد توصيل الـ dongle")
    return {"board": board, "serial_port": serial_port, "mac_address": str(cfg.get("mac_address") or "").strip(),
            "ch_names": names}


def save_config(cfg: dict) -> dict:
    clean = validate(cfg)
    DEVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_FILE.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


def delete_config() -> None:
    DEVICE_FILE.unlink(missing_ok=True)


def open_device() -> BrainFlowSource:
    cfg = load_config()
    if cfg is None:
        raise ConfigError("لم يُعدّ جهاز بعد — افتح صفحة «جهازك»")
    return BrainFlowSource(cfg["board"], cfg["serial_port"], cfg["mac_address"], cfg["ch_names"])


def readiness(cfg: dict) -> list[dict]:
    """ماذا يستطيع النظام أن يفعل بأقطاب هذا الجهاز — لكل مسار: جاهز، يحتاج تجهيزاً، أو غير مدعوم."""
    fs = board_spec(cfg["board"])["fs"]
    names = cfg["ch_names"]
    return [
        {"title": "البث المباشر وجودة الأقطاب", "track": "live", "status": "ready", "reason": None},
        {"title": "الكتابة بالدماغ", **mi_device.requirements(names, fs, cfg["board"])},
        {"title": "الحالة الذهنية", **mental_device.requirements(names)},
        {"title": "فحص الصرع", **epilepsy_device.requirements(names)},
    ]


def label() -> str | None:
    cfg = load_config()
    return BOARDS[cfg["board"]]["label"] if cfg else None


def recording_path(prefix: str) -> Path:
    """تسجيلاتك تُحفظ داخل DATA_DIR فتظهر تلقائياً في قائمة المصادر لإعادة تشغيلها."""
    return RECORDINGS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.edf"


# ---------- تجهيز النماذج لأقطاب الجهاز (عمل خلفي يأخذ دقائق) ----------

class PrepareJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"   # idle | running | done | error
        self.log: list[str] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def write(self, line: str) -> None:
        with self.lock:
            self.log.append(line)

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = None
            if self.started_at:
                elapsed = round((self.finished_at or time.time()) - self.started_at)
            return {"status": self.status, "log": list(self.log), "elapsed_sec": elapsed}

    def start(self, ch_names: list[str]) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status, self.log = "running", []
            self.started_at, self.finished_at = time.time(), None
        threading.Thread(target=self._run, args=(list(ch_names),), daemon=True).start()
        return True

    def _run(self, ch_names: list[str]) -> None:
        try:
            run_prepare(ch_names, self.write)
            status = "done"
        except Exception as exc:
            self.write(f"خطأ: {exc}")
            status = "error"
        with self.lock:
            self.status, self.finished_at = status, time.time()


def run_prepare(ch_names: list[str], log=print) -> None:
    log(f"تجهيز النماذج لأقطاب جهازك ({len(ch_names)}): {'، '.join(ch_names)}")
    mental_device.prepare(ch_names, log)
    epilepsy_device.prepare(ch_names, log)
    log("تم ✔ — التقارير في صفحة «النماذج ونتائجها». الكتابة بالدماغ تحتاج معايرتك أنت (من صفحتها).")


JOB = PrepareJob()


def state() -> dict:
    cfg = load_config()
    return {
        "config": cfg,
        "boards": catalog(),
        "standard_names": STANDARD_10_20,
        "readiness": readiness(cfg) if cfg else None,
        "prepare": JOB.snapshot(),
    }
