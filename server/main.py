"""خادم NeuroLens المحلي.

التشغيل:
    C:\\Users\\MSI\\neurolens-venv\\Scripts\\python -m uvicorn server.main:app --port 8770

ثم افتح الواجهة (npm run dev داخل web/) أو http://127.0.0.1:8770 بعد بنائها.
"""
import asyncio
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import ensure_dirs
from core.evaluation.report import load_report, load_reports
from tracks.epilepsy import device as epilepsy_device
from tracks.epilepsy import service as epilepsy
from tracks.mental_state import device as mental_device
from tracks.mental_state import service as mental
from tracks.motor_imagery import device as mi_device
from tracks.motor_imagery import service as speller

from . import device
from .session import LiveSession
from .sources import is_recording, list_sources, open_source, source_path

FRAME_SEC = 0.1  # الواجهة تُحدَّث 10 مرات في الثانية

ensure_dirs()
app = FastAPI(title="NeuroLens")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


# ---------- الحالة الذهنية ----------

@app.get("/api/mental/scales")
def mental_scales() -> list[dict]:
    return mental.scales()


@app.get("/api/mental/{scale_id}/{subject}")
def mental_timeline(scale_id: str, subject: str) -> dict:
    tl = mental.timeline(scale_id, subject)
    if tl is None:
        raise HTTPException(404, "لا توجد نتائج لهذا الشخص — شغّل التدريب أولاً")
    return tl


# ---------- الكتابة بالدماغ ----------

class SimulateRequest(BaseModel):
    subject: int
    target: str
    seed: int = 0
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class StepRequest(BaseModel):
    state: dict
    command: str


@app.get("/api/speller/info")
def speller_info() -> dict:
    return speller.info()


@app.get("/api/speller/subjects")
def speller_subjects() -> list[dict]:
    return speller.subjects()


@app.post("/api/speller/simulate")
def speller_simulate(req: SimulateRequest) -> dict:
    try:
        return speller.simulate(req.subject, req.target, req.seed, req.min_confidence)
    except speller.SpellerError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/speller/step")
def speller_step(req: StepRequest) -> dict:
    try:
        return speller.step(req.state, req.command)
    except speller.SpellerError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/sources")
def sources() -> list[dict]:
    return list_sources()


# ---------- جهازك ----------

class DeviceConfig(BaseModel):
    board: str
    serial_port: str = ""
    mac_address: str = ""
    ch_names: list[str] | None = None


@app.get("/api/device")
def device_state() -> dict:
    return device.state()


@app.put("/api/device")
def device_save(cfg: DeviceConfig) -> dict:
    try:
        device.save_config(cfg.model_dump())
    except device.ConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return device.state()


@app.delete("/api/device")
def device_delete() -> dict:
    device.delete_config()
    return device.state()


@app.post("/api/device/prepare")
def device_prepare() -> dict:
    cfg = device.load_config()
    if cfg is None:
        raise HTTPException(409, "أعدّ الجهاز أولاً")
    if not device.JOB.start(cfg["ch_names"]):
        raise HTTPException(409, "التجهيز يعمل الآن")
    return device.state()


@app.get("/api/models")
def models() -> list[dict]:
    return load_reports()


@app.get("/api/models/{model_id}")
def model(model_id: str) -> dict:
    report = load_report(model_id)
    if report is None:
        raise HTTPException(404, "لا يوجد تقرير بهذا الاسم")
    return report


@app.get("/api/epilepsy/recordings")
def epilepsy_recordings() -> list[dict]:
    return epilepsy.list_recordings()


@app.get("/api/epilepsy/recordings/{file}")
def epilepsy_recording(file: str) -> dict:
    tl = epilepsy.recording_timeline(file)
    if tl is None:
        raise HTTPException(404, "لا توجد نتائج لهذا التسجيل — شغّل التدريب أولاً")
    return tl


@app.post("/api/epilepsy/analyze")
def epilepsy_analyze(file: UploadFile) -> dict:
    """يحلّل ملف EDF مرفوعاً بالنموذج العام. (دالة عادية لا async: التحليل يأخذ ثوانٍ
    فيشغّله FastAPI في خيط منفصل ولا يتوقف البث المباشر.)"""
    name = Path(file.filename or "upload.edf").name
    if not name.lower().endswith(".edf"):
        raise HTTPException(400, "الملف يجب أن يكون بصيغة EDF")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.edf"
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            return epilepsy.analyze_edf(path, name)
        except epilepsy.AnalysisError as exc:
            raise HTTPException(422, str(exc)) from exc


TASKS = ("", "mi_calibration", "speller")


def make_session(source: str, speed: float, start: float, task: str, record: bool,
                 per_class: int, min_confidence: float, target: str) -> LiveSession:
    if task not in TASKS:
        raise ValueError(f"مهمة غير معروفة: {task}")
    if task and source != "device":
        raise ValueError("المعايرة والكتابة تحتاجان جهازاً — أعدّه من صفحة «جهازك» "
                         "(يمكن اختيار «جهاز BrainFlow محاكى» للتجربة بلا عتاد)")
    src = open_source(source, speed=speed, start_sec=start)
    record_to = None
    if task == "mi_calibration":
        record_to = device.recording_path("calibration")  # تُحفظ دائماً مع علامة كل أمر
    elif record and src.info.kind == "device":
        record_to = device.recording_path("session")
    session = LiveSession(src, record_to=record_to)
    info = src.info
    board = (device.load_config() or {}).get("board")
    if task == "mi_calibration":
        session.add_analyzer("calibration", mi_device.Calibration(info.ch_names, info.fs, board, per_class))
        return session
    if task == "speller":
        bundle = mi_device.person_model(info.ch_names, info.fs, board)
        if bundle is None:
            raise ValueError("لا توجد معايرة لك على هذا الجهاز بهذه الأقطاب — ابدأ بالمعايرة.")
        session.add_analyzer("speller", mi_device.LiveSpeller(bundle, info.ch_names, info.fs, min_confidence, target))
        return session
    path = source_path(source)
    if path is not None and (analyzer := epilepsy.live_analyzer(path)):
        session.add_analyzer("seizure", analyzer)
    if path is not None and (analyzer := mental.live_analyzer(path)):
        session.add_analyzer("mental", analyzer)
    if source == "device" or (path is not None and is_recording(path)):
        # جهازك أو تسجيل منه: نماذج مجهّزة لأقطابه (إن وُجدت)
        if (analyzer := mental_device.live_analyzer(info.ch_names, info.fs)):
            session.add_analyzer("mental", analyzer)
        if (analyzer := epilepsy_device.live_analyzer(info.ch_names, info.fs)):
            session.add_analyzer("seizure", analyzer)
    return session


@app.websocket("/ws")
async def stream(ws: WebSocket, source: str = "synthetic", speed: float = 1.0,
                 start: float = 0.0, task: str = "", record: int = 0, per_class: int = 10,
                 min_confidence: float = 0.0, target: str = "") -> None:
    """بث مباشر: رسالة hello ثم رسالة frame كل 100ms.

    الواجهة تغيّر المصدر بإغلاق الاتصال وفتح اتصال جديد بمعامل source مختلف.
    start: من أي ثانية في التسجيل نبدأ (للقفز إلى ما قبل نوبة مثلاً).
    task: "" بث عادي، "mi_calibration" جلسة معايرة الكتابة بالدماغ، "speller" كتابة مباشرة بجهازك.
    record: 1 = احفظ الجلسة كملف EDF عند الإغلاق (للجهاز فقط).
    """
    await ws.accept()
    try:
        session = await asyncio.to_thread(make_session, source, speed, start, task, bool(record),
                                          per_class, min_confidence, target)
    except Exception as exc:  # ملف تالف، مصدر غير موجود، أو جهاز غير معَدّ
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close()
        return
    try:
        await ws.send_json(session.hello())
        while True:
            # المعالجة في خيط منفصل: تدريب نموذج المعايرة أو اتصال الجهاز قد يأخذ ثوانٍ
            await ws.send_json(await asyncio.to_thread(session.step))
            await asyncio.sleep(FRAME_SEC)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # انقطع الجهاز أو تعذّر الاتصال به
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
        except Exception:
            pass
    finally:
        # الإغلاق (وحفظ التسجيل) في خيط مستقل: يكتمل حتى لو أُلغيت هذه المهمة عند انقطاع الاتصال
        threading.Thread(target=close_quietly, args=(session,)).start()


def close_quietly(session: LiveSession) -> None:
    try:
        session.close()
    except Exception as exc:
        print(f"تعذّر إغلاق الجلسة أو حفظ تسجيلها: {exc}")


# بعد بناء الواجهة (npm run build) يخدمها الخادم نفسه على نفس المنفذ
_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
