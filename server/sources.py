"""سجل مصادر الإشارة المتاحة: ما الذي يمكن للواجهة أن تختار بثّه."""
from pathlib import Path

from core.config import DATA_DIR
from core.sources import ReplaySource, SyntheticEEG
from core.sources.base import SignalSource
from tracks.epilepsy.data import seizures_for_path

from . import device

# مزوّدو الأحداث المعروفة لكل ملف (مثل أوقات نوبات الصرع)
EVENT_PROVIDERS: list = [seizures_for_path]


def find_edf_files(limit: int = 200) -> list[Path]:
    """تسجيلاتك أولاً (الأحدث فالأقدم)، ثم ملفات البيانات العامة."""
    if not DATA_DIR.exists():
        return []
    mine = sorted(device.RECORDINGS_DIR.glob("*.edf"), reverse=True) if device.RECORDINGS_DIR.exists() else []
    public = [p for p in sorted(DATA_DIR.rglob("*.edf")) if device.RECORDINGS_DIR not in p.parents]
    return (mine + public)[:limit]


def is_recording(path: Path) -> bool:
    """تسجيل من جهازك (لا من البيانات العامة)."""
    return device.RECORDINGS_DIR.resolve() in path.resolve().parents


def known_events(path: Path) -> list[dict]:
    for provider in EVENT_PROVIDERS:
        events = provider(path)
        if events:
            return events
    return []


def list_sources() -> list[dict]:
    items = [{"id": "synthetic", "label": "إشارة اصطناعية للتجربة", "kind": "synthetic"}]
    if (name := device.label()):
        items.insert(0, {"id": "device", "label": f"جهازك: {name}", "kind": "device"})
    for path in find_edf_files():
        rel = path.relative_to(DATA_DIR).as_posix()
        if is_recording(path):
            items.append({"id": f"edf:{rel}", "label": f"تسجيلك: {path.stem}", "kind": "recording"})
            continue
        events = known_events(path)
        label = path.name + (f" — {len(events)} نوبة مسجّلة" if events else "")
        items.append({"id": f"edf:{rel}", "label": label, "kind": "replay",
                      "n_events": len(events)})
    return items


def source_path(source_id: str) -> Path | None:
    """مسار ملف EDF لمصدر من نوع edf:، مع منع الخروج من مجلد البيانات."""
    if not source_id.startswith("edf:"):
        return None
    path = (DATA_DIR / source_id[4:]).resolve()
    if DATA_DIR.resolve() not in path.parents:
        raise ValueError("مسار غير مسموح")
    return path


def open_source(source_id: str, speed: float = 1.0, start_sec: float = 0.0) -> SignalSource:
    if source_id == "synthetic":
        return SyntheticEEG()
    if source_id == "device":
        return device.open_device()
    if source_id == "brainflow-synthetic":
        from core.sources.brainflow_device import BrainFlowSource
        return BrainFlowSource("synthetic")
    path = source_path(source_id)
    if path is not None:
        return ReplaySource.from_edf(path, events=known_events(path), speed=speed, start_sec=start_sec)
    raise ValueError(f"مصدر غير معروف: {source_id}")
