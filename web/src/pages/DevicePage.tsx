import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Cpu, Play, RotateCcw, Save, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getJson, sendJson } from "../lib/api";
import { CONNECTION_HINT, READINESS, nameProblems } from "../lib/device";
import { formatClock } from "../lib/liveStore";
import type { BoardInfo, DeviceConfig, DeviceState, Readiness } from "../lib/types";

const PREFERRED_BOARD = "openbci_cyton_daisy"; // يغطي المسارات الثلاثة — انظر README

export function DevicePage() {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["device"],
    queryFn: () => getJson<DeviceState>("/api/device"),
    // أثناء تجهيز النماذج نتابع سجلّه كل ثانية ونصف
    refetchInterval: (query) => (query.state.data?.prepare.status === "running" ? 1500 : false),
  });
  const update = (s: DeviceState) => {
    qc.setQueryData(["device"], s);
    qc.invalidateQueries({ queryKey: ["sources"] });
  };

  return (
    <div className="page">
      <header className="page-head">
        <h1>جهازك</h1>
        <p className="muted">
          هنا تربط جهاز EEG حقيقياً بالنظام: تختار نوعه، وتكتب أي قطب وضعته على كل قناة. بعدها يعرف النظام ماذا يستطيع أن
          يفعل بأقطابك، ويعيد تدريب نماذجه عليها. لا تملك جهازاً بعد؟ اختر «جهاز BrainFlow محاكى» لتجرّب كل الخطوات.
        </p>
      </header>
      <div className="banner banner-warn">
        <strong>السلامة:</strong> جهاز EEG يقيس فقط ولا يرسل كهرباء للجسم. استعمل جهازاً يعمل بالبطارية، ولا توصل أقطاباً برأس
        أحد عبر دائرة مصنوعة يدوياً موصولة بكهرباء الحائط. افصل شاحن اللابتوب أثناء القياس. النتائج للتعلّم — ليست تشخيصاً.
      </div>
      {q.error && <div className="banner banner-bad">{(q.error as Error).message}</div>}
      {q.data && <DeviceForm state={q.data} onSaved={update} />}
      {q.data?.config && q.data.readiness && <ReadinessView items={q.data.readiness} />}
      {q.data?.config && <PrepareView state={q.data} onStarted={update} />}
      <SetupGuide />
    </div>
  );
}

function DeviceForm({ state, onSaved }: { state: DeviceState; onSaved: (s: DeviceState) => void }) {
  const initial: DeviceConfig = state.config ?? (() => {
    const b = state.boards.find((x) => x.id === PREFERRED_BOARD) ?? state.boards[0];
    return { board: b.id, serial_port: "", mac_address: "", ch_names: b.default_names };
  })();
  const [cfg, setCfg] = useState<DeviceConfig>(initial);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  // بالمحتوى لا بهوية الكائن: المتابعة الدورية أثناء التجهيز تعيد كائناً جديداً بنفس الإعداد
  const savedKey = JSON.stringify(state.config);
  useEffect(() => { if (state.config) setCfg(state.config); }, [savedKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const board = state.boards.find((b) => b.id === cfg.board) as BoardInfo;
  const problems = useMemo(() => nameProblems(cfg.ch_names), [cfg.ch_names]);
  const dirty = JSON.stringify(cfg) !== JSON.stringify(state.config);

  const chooseBoard = (id: string) => {
    const b = state.boards.find((x) => x.id === id)!;
    setCfg({ ...cfg, board: id, ch_names: b.default_names });
    setSaved(false);
  };
  const setName = (i: number, v: string) => {
    setCfg({ ...cfg, ch_names: cfg.ch_names.map((n, k) => (k === i ? v : n)) });
    setSaved(false);
  };
  const save = async () => {
    setError(null);
    try {
      onSaved(await sendJson<DeviceState>("PUT", "/api/device", cfg));
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const remove = async () => {
    onSaved(await sendJson<DeviceState>("DELETE", "/api/device"));
    setSaved(false);
  };

  return (
    <section className="card">
      <h2>1. نوع الجهاز وأقطابه</h2>
      <div className="controls">
        <label className="grow">
          الجهاز
          <select value={cfg.board} onChange={(e) => chooseBoard(e.target.value)}>
            {state.boards.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
          </select>
        </label>
        {board.connection === "serial" && (
          <label>
            المنفذ
            <input className="ltr" placeholder="COM3" value={cfg.serial_port}
              onChange={(e) => { setCfg({ ...cfg, serial_port: e.target.value }); setSaved(false); }} />
          </label>
        )}
        {board.connection === "bluetooth" && (
          <label>
            عنوان البلوتوث (اختياري)
            <input className="ltr" placeholder="00:55:DA:B0:12:34" value={cfg.mac_address}
              onChange={(e) => { setCfg({ ...cfg, mac_address: e.target.value }); setSaved(false); }} />
          </label>
        )}
      </div>
      <p className="small muted">
        {CONNECTION_HINT[board.connection]} · <span className="num">{board.n_channels}</span> قناة ·{" "}
        <span className="num">{board.fs}</span> عيّنة/ثانية
      </p>

      <h3>أي قطب على كل قناة؟</h3>
      <p className="small muted">
        الأسماء الافتراضية هي التركيب الذي يقترحه صانع الجهاز. إن وضعت قطباً في مكان آخر غيّر اسمه هنا — النظام يبني كل
        نماذجه على هذه الأسماء. (T3/T4/T5/T6 أسماء قديمة لـ T7/T8/P7/P8 ويعرفها النظام.)
      </p>
      <div className="channels">
        {cfg.ch_names.map((n, i) => (
          <label key={i} className="channel">
            <span className="small faint">القناة <span className="num">{i + 1}</span></span>
            <input className="ltr" list="std-names" value={n} onChange={(e) => setName(i, e.target.value)} />
          </label>
        ))}
      </div>
      <datalist id="std-names">{state.standard_names.map((n) => <option key={n} value={n} />)}</datalist>
      {problems.map((p) => <div key={p} className="banner banner-warn small">{p}</div>)}
      {error && <div className="banner banner-bad">{error}</div>}
      <div className="row-actions">
        <button className="btn btn-primary" onClick={save} disabled={problems.length > 0}>
          <Save size={15} aria-hidden /> احفظ الإعداد
        </button>
        <button className="btn" onClick={() => chooseBoard(cfg.board)}>
          <RotateCcw size={15} aria-hidden /> الأسماء الافتراضية
        </button>
        {state.config && (
          <button className="btn" onClick={remove}><Trash2 size={15} aria-hidden /> احذف الإعداد</button>
        )}
        {saved && !dirty && <span className="pill pill-ok">حُفظ</span>}
        {dirty && state.config && <span className="pill pill-warn">تغييرات غير محفوظة</span>}
      </div>
    </section>
  );
}

const LINKS: Record<Readiness["track"], { to: string; text: string }> = {
  live: { to: "/?source=device", text: "افتح البث المباشر وافحص جودة الأقطاب" },
  motor_imagery: { to: "/speller?mode=device", text: "المعايرة والكتابة" },
  mental_state: { to: "/?source=device", text: "قِس حالتك في البث المباشر" },
  epilepsy: { to: "/?source=device", text: "راقب في البث المباشر" },
};

function ReadinessView({ items }: { items: Readiness[] }) {
  return (
    <section className="card">
      <h2>2. ماذا يستطيع النظام بأقطابك؟</h2>
      <div className="readiness">
        {items.map((r) => {
          const s = READINESS[r.status];
          return (
            <div key={r.track} className="ready-card">
              <div className="ready-head">
                <strong>{r.title}</strong>
                <span className={`pill pill-${s.tone}`}>{s.text}</span>
              </div>
              {r.reason && <p className="small muted">{r.reason}</p>}
              <ReadinessDetails r={r} />
              {r.status !== "unsupported" && r.status !== "needs_prepare" && (
                <Link className="small" to={LINKS[r.track].to}>{LINKS[r.track].text} ←</Link>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function ReadinessDetails({ r }: { r: Readiness }) {
  if (r.track === "motor_imagery") {
    return (
      <p className="small faint">
        {r.accuracy != null && <>آخر معايرة: دقة <span className="num">{Math.round(r.accuracy * 100)}%</span> (الصدفة 25%). </>}
        {!!r.missing_recommended?.length && <>أقطاب تحسّن النتيجة إن أضفتها: <span className="ltr">{r.missing_recommended.join(" ")}</span></>}
      </p>
    );
  }
  if (r.track === "mental_state" && r.scales) {
    return (
      <ul className="small faint plain">
        {r.scales.map((s) => (
          <li key={s.scale}>
            {s.title}: {s.supported
              ? <><span className="num">{s.used.length}</span> من <span className="num">{s.n_dataset_channels}</span> قطباً {s.prepared ? "· مجهّز" : ""}</>
              : "غير مدعوم"}
          </li>
        ))}
      </ul>
    );
  }
  if (r.track === "epilepsy" && r.pairs) {
    return (
      <p className="small faint">
        <span className="num">{r.pairs.length}</span> من <span className="num">{r.n_pairs_total}</span> قناة ثنائية ممكنة.
        {!!r.missing_electrodes?.length && <> ينقص: <span className="ltr">{r.missing_electrodes.join(" ")}</span>.</>}
        {" "}تذكير: النموذج العام ضعيف (انظر صفحة النماذج) — للتعلّم فقط.
      </p>
    );
  }
  return null;
}

function PrepareView({ state, onStarted }: { state: DeviceState; onStarted: (s: DeviceState) => void }) {
  const job = state.prepare;
  const [error, setError] = useState<string | null>(null);
  const needs = state.readiness?.some((r) => r.status === "needs_prepare");
  const start = async () => {
    setError(null);
    try {
      onStarted(await sendJson<DeviceState>("POST", "/api/device/prepare"));
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <section className="card">
      <h2>3. تجهيز النماذج لأقطابك</h2>
      <p className="small muted">
        يعيد تدريب مقياسَي الحالة الذهنية ونموذج الصرع على الأقطاب المشتركة بين جهازك والبيانات العامة، ويقيسها على
        أشخاص لم يرهم — فترى بصدق كم تخسر بأقطاب أقل. يأخذ بضع دقائق، ويحتاج البيانات المنزَّلة. (الكتابة بالدماغ لا تُجهَّز
        هنا: تحتاج معايرتك أنت.)
      </p>
      <div className="row-actions">
        <button className={`btn ${needs ? "btn-primary" : ""}`} onClick={start} disabled={job.status === "running"}>
          <Play size={15} aria-hidden /> {job.status === "running" ? "يعمل…" : "جهّز النماذج"}
        </button>
        {job.status !== "idle" && (
          <span className={`pill pill-${job.status === "done" ? "ok" : job.status === "error" ? "bad" : "info"}`}>
            {{ running: "يعمل", done: "اكتمل", error: "خطأ", idle: "" }[job.status]}
            {job.elapsed_sec != null && <> · <span className="num">{formatClock(job.elapsed_sec)}</span></>}
          </span>
        )}
        {job.status === "done" && <Link className="small" to="/models">النتائج في صفحة النماذج ←</Link>}
      </div>
      {error && <div className="banner banner-bad">{error}</div>}
      {job.log.length > 0 && <pre className="log">{job.log.join("\n")}</pre>}
    </section>
  );
}

function SetupGuide() {
  return (
    <section className="card">
      <h2><Cpu size={18} aria-hidden /> قبل كل جلسة</h2>
      <ol className="steps">
        <li>اشحن الجهاز أو ضع بطاريات جديدة. افصل شاحن اللابتوب أثناء القياس (يقلّل تشويش كهرباء البيت 50Hz).</li>
        <li>قِس محيط الرأس واختر مقاس القبعة. ضع Cz في منتصف المسافة بين الأنف ونتوء مؤخرة الرأس، وبين الأذنين.</li>
        <li>نظّف مواضع الأقطاب بمنديل كحول، ثم ضع الجل بمحقنة بطرف غير حاد حتى يلامس القطب الجلد (للأقطاب الرطبة).</li>
        <li>ثبّت المرجع (SRB/REF) والأرضي (BIAS/GND) على شحمتَي الأذن أو خلفهما كما في دليل جهازك.</li>
        <li>
          افتح <Link to="/?source=device">البث المباشر</Link> وانتظر حتى تصبح كل الأقطاب «جيدة». «مسطّحة» = لا تلامس،
          «مشوّشة» = حركة أو جل ناقص.
        </li>
        <li>اجلس مرتاحاً، لا تشدّ فكّك ولا ترمش كثيراً أثناء التخيّل — العضلات والعينان أقوى من إشارة الدماغ بعشرات المرات.</li>
        <li>للكتابة بالدماغ: معايرة لك أولاً (نحو 5 دقائق)، ويُفضّل إعادتها في بداية كل يوم.</li>
      </ol>
    </section>
  );
}
