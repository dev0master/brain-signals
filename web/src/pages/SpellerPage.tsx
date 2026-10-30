import { useQuery } from "@tanstack/react-query";
import { Pause, Play, RotateCcw, Square } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { QualityChips } from "../components/QualityChips";
import { getJson } from "../lib/api";
import { calibrationProgress, calibrationPrompt, spellerPrompt } from "../lib/device";
import { formatClock } from "../lib/liveStore";
import { commandForKey, commandsFor, errorsUpTo, isDefaultMapping, simulatedSeconds, speedLabel } from "../lib/speller";
import type {
  CalibrationLive, CommandMeta, DeviceState, SimResult, SpellerCommand, SpellerInfo, SpellerLive, SpellerSubject,
  SpellerView,
} from "../lib/types";
import { useLiveStream } from "../lib/useLiveStream";

const EXAMPLES = ["أحتاج ماء", "نعم", "ألم في رأسي", "شكرا"];
const SPEEDS = [
  { label: "سريع", ms: 350 },
  { label: "متوسط", ms: 900 },
  { label: "بطيء", ms: 1800 },
];
const ICON: Record<SpellerCommand, string> = { left: "←", right: "→", select: "✓", back: "⌫" };
// ترتيب احتمالات كل محاولة كما يرسلها الخادم (tracks/motor_imagery/data.py: CLASSES)
const CLASS_ORDER = ["left_hand", "right_hand", "feet", "tongue"];

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `خطأ (${res.status})`);
  return data as T;
}

export function SpellerPage() {
  const info = useQuery({ queryKey: ["speller-info"], queryFn: () => getJson<SpellerInfo>("/api/speller/info") });
  const subjects = useQuery({ queryKey: ["speller-subjects"], queryFn: () => getJson<SpellerSubject[]>("/api/speller/subjects") });
  const [params] = useSearchParams();
  const [mode, setMode] = useState<"sim" | "manual" | "device">(params.get("mode") === "device" ? "device" : "sim");

  return (
    <div className="page">
      <header className="page-head">
        <h1>الكتابة بالدماغ</h1>
        <p className="muted">
          أربعة أوامر فقط، كل أمر = تخيّل حركة دون تحريك الجسم. بها نتنقل في لوحة حروف عربية: نختار مجموعة ثم حرفاً منها،
          أو كلمة مقترحة تكمل ما نكتبه. هكذا يكتب أشخاص لا يستطيعون الكلام ولا الحركة.
        </p>
      </header>

      <div className="seg">
        <button className={mode === "sim" ? "active" : ""} onClick={() => setMode("sim")}>محاكاة بإشارات حقيقية</button>
        <button className={mode === "manual" ? "active" : ""} onClick={() => setMode("manual")}>جرّب اللوحة بنفسك</button>
        <button className={mode === "device" ? "active" : ""} onClick={() => setMode("device")}>بجهازك</button>
      </div>

      {info.error && <div className="banner banner-bad">{(info.error as Error).message}</div>}
      {info.data && mode === "sim" && <Simulation info={info.data} subjects={subjects.data ?? []} />}
      {info.data && mode === "manual" && <Manual info={info.data} />}
      {info.data && mode === "device" && <DeviceSpeller />}
    </div>
  );
}

// ---------- بجهازك: معايرة ثم كتابة ----------

const PER_CLASS_OPTIONS = [6, 10, 15, 20];
const SECONDS_PER_TRIAL = 8; // استعداد 2 + تخيّل 4 + راحة 2 تقريباً

function DeviceSpeller() {
  const device = useQuery({ queryKey: ["device"], queryFn: () => getJson<DeviceState>("/api/device") });
  const [stage, setStage] = useState<"idle" | "calibrating" | "typing">("idle");
  const [perClass, setPerClass] = useState(10);
  const [target, setTarget] = useState("");
  const [minConf, setMinConf] = useState(0);
  const mi = device.data?.readiness?.find((r) => r.track === "motor_imagery");
  const back = () => { setStage("idle"); device.refetch(); };

  if (device.isLoading) return <p className="muted">جارٍ التحميل…</p>;
  if (!device.data?.config) {
    return (
      <section className="card">
        <p>لم تُعِدّ جهازاً بعد. افتح صفحة <Link to="/device">جهازك</Link> واختر جهازك (أو الجهاز المحاكى للتجربة).</p>
      </section>
    );
  }
  if (mi?.status === "unsupported") return <div className="banner banner-bad">{mi.reason}</div>;
  if (stage === "calibrating") return <CalibrationRun perClass={perClass} onExit={back} onType={() => setStage("typing")} />;
  if (stage === "typing") return <LiveTyping target={target} minConfidence={minConf} onExit={back} />;

  const minutes = Math.round((perClass * 4 * SECONDS_PER_TRIAL) / 60);
  return (
    <>
      <div className="banner banner-info">
        نماذج الأشخاص التسعة لا تصلح لك: كل نموذج تعلّم دماغ صاحبه. لذلك تبدأ بـ<strong>معايرة</strong>: يظهر أمر على الشاشة
        فتتخيّل الحركة 4 ثوانٍ دون أن تتحرك، وكل أمر يتكرر بترتيب عشوائي. ثم يتدرّب نموذجك وتبدأ الكتابة.
      </div>
      <div className="grid-2">
        <section className="card">
          <h2>1. المعايرة</h2>
          <div className="controls">
            <label>
              محاولات لكل أمر
              <select value={perClass} onChange={(e) => setPerClass(Number(e.target.value))}>
                {PER_CLASS_OPTIONS.map((n) => <option key={n} value={n}>{n} (نحو {Math.round((n * 4 * SECONDS_PER_TRIAL) / 60)} دقائق)</option>)}
              </select>
            </label>
            <button className="btn btn-primary" onClick={() => setStage("calibrating")}>
              <Play size={15} aria-hidden /> ابدأ المعايرة
            </button>
          </div>
          <p className="small muted">
            {minutes} دقائق تقريباً. محاولات أكثر = نموذج أدق. قبل البدء تأكد في{" "}
            <Link to="/?source=device">البث المباشر</Link> أن الأقطاب «جيدة»، خصوصاً C3 وC4.
          </p>
          {mi?.status === "ready" && mi.accuracy != null && (
            <p className="small">آخر معايرة: دقة <span className="num">{Math.round(mi.accuracy * 100)}%</span> (الصدفة 25%).</p>
          )}
        </section>
        <section className="card">
          <h2>2. الكتابة</h2>
          {mi?.status !== "ready" ? (
            <p className="muted">تُفتح بعد أول معايرة.</p>
          ) : (
            <>
              <div className="controls">
                <label className="grow">
                  جملة تتدرّب على كتابتها (اختياري)
                  <input value={target} onChange={(e) => setTarget(e.target.value)} maxLength={40} placeholder="مثل: نعم" />
                </label>
                <label>
                  لا تنفّذ إن كانت الثقة أقل من
                  <select value={minConf} onChange={(e) => setMinConf(Number(e.target.value))}>
                    {[0, 0.4, 0.5, 0.6].map((v) => <option key={v} value={v}>{v ? `${v * 100}%` : "نفّذ دائماً"}</option>)}
                  </select>
                </label>
                <button className="btn btn-primary" onClick={() => setStage("typing")}>
                  <Play size={15} aria-hidden /> ابدأ الكتابة
                </button>
              </div>
              <p className="small muted">كل 6 ثوانٍ أمر واحد: «استعد» ثم «تخيّل الآن»، والنموذج يقرأ ما تخيّلته وينفّذه.</p>
            </>
          )}
        </section>
      </div>
    </>
  );
}

function CalibrationRun({ perClass, onExit, onType }: { perClass: number; onExit: () => void; onType: () => void }) {
  const live = useLiveStream("device", 1, 0, { task: "mi_calibration", per_class: perClass });
  const c = live.last?.calibration as CalibrationLive | undefined;
  const prompt = c ? calibrationPrompt(c) : { title: "جارٍ الاتصال بالجهاز…", hint: "", tone: "info" };
  const result = c?.result;
  return (
    <>
      {live.error && <div className="banner banner-bad">{live.error}</div>}
      <section className={`card cue cue-${prompt.tone}`} aria-live="assertive">
        <div className="cue-title">{prompt.title}</div>
        {prompt.hint && <div className="cue-hint">{prompt.hint}</div>}
        {c && c.remaining_sec !== undefined && c.phase !== "done" && <div className="num faint">{Math.ceil(c.remaining_sec)}</div>}
      </section>
      {c && (
        <section className="card">
          <div className="seizure-row">
            <span className="small">المحاولة <span className="num">{Math.min(c.trial ?? c.collected, c.n_trials)}</span> من <span className="num">{c.n_trials}</span></span>
            <div className="meter grow-meter"><div className="meter-fill" style={{ width: `${Math.round(calibrationProgress(c) * 100)}%` }} /></div>
            <button className="btn" onClick={onExit}><Square size={14} aria-hidden /> {c.phase === "done" ? "رجوع" : "أوقف"}</button>
          </div>
          <div className="small muted" style={{ marginTop: 10 }}>جودة الأقطاب الآن:</div>
          <QualityChips names={live.hello?.ch_names ?? []} quality={live.last?.quality} />
        </section>
      )}
      {result && (
        <section className="card">
          <h2>نتيجة معايرتك</h2>
          <div className="vs-chance">
            <div className="metric"><div className="metric-value num">{Math.round(result.accuracy * 100)}%</div><div className="small muted">الدقة</div></div>
            <div className="metric metric-muted"><div className="metric-value num">25%</div><div className="small muted">مستوى الصدفة</div></div>
            <div className="metric metric-muted"><div className="metric-value num">{result.n_trials}</div><div className="small muted">محاولة</div></div>
          </div>
          {result.accuracy < 0.4 ? (
            <div className="banner banner-warn">
              قريبة من الصدفة — النموذج بالكاد يميّز أوامرك. جرّب: تحقق من جودة C3 وC4، تخيّل الإحساس بالحركة لا صورتها،
              وأعد المعايرة بمحاولات أكثر. كثيرون يحتاجون عدة جلسات تدريب. (مع الجهاز المحاكى هذه النتيجة متوقعة: لا دماغ فيه.)
            </div>
          ) : (
            <p className="small">
              سرعة الكتابة المتوقعة: {speedLabel(result.spelling.median_chars_per_minute, result.spelling.completion_rate)}.
            </p>
          )}
          <p className="small faint">
            قيست بتقسيم داخلي لجلستك نفسها — تقدير متفائل: في يوم آخر تنخفض الدقة عادة، فأعد المعايرة في بداية كل جلسة.
          </p>
          <CommandLegend commands={result.commands} title="توزيع الأوامر الذي اختير لك (من معايرتك فقط):" />
          <div className="row-actions">
            <button className="btn btn-primary" onClick={onType}><Play size={15} aria-hidden /> ابدأ الكتابة</button>
            <button className="btn" onClick={onExit}><RotateCcw size={15} aria-hidden /> رجوع</button>
          </div>
        </section>
      )}
    </>
  );
}

function LiveTyping({ target, minConfidence, onExit }: { target: string; minConfidence: number; onExit: () => void }) {
  const extra: Record<string, string | number> = { task: "speller", min_confidence: minConfidence };
  if (target.trim()) extra.target = target.trim();
  const live = useLiveStream("device", 1, 0, extra);
  const s = live.last?.speller as SpellerLive | undefined;
  const prompt = s ? spellerPrompt(s) : { title: "جارٍ الاتصال بالجهاز…", tone: "info" };
  const byCommand = Object.fromEntries((s?.commands ?? []).map((c) => [c.command, c]));
  return (
    <>
      {live.error && <div className="banner banner-bad">{live.error}</div>}
      <section className={`card cue cue-${prompt.tone}`} aria-live="assertive">
        <div className="cue-title">{prompt.title}</div>
        {s?.suggested && byCommand[s.suggested] && (
          <div className="cue-hint">للوصول إلى «{s.target}»: تخيّل {byCommand[s.suggested].imagine} ({byCommand[s.suggested].action})</div>
        )}
        {s?.done && <div className="cue-hint">اكتملت الجملة ✔</div>}
      </section>
      {s && <CommandLegend commands={s.commands} title="أوامرك:" />}
      <Board view={s?.view ?? { text: "", group: null, cursor: 0, items: [] }} target={s?.target} />
      <div className="grid-2">
        <section className="card">
          <h2>آخر أمر</h2>
          {s?.last ? (
            <>
              <p>
                فهم النموذج: <strong>تخيّل {byCommand[s.last.decoded]?.imagine}</strong> ({byCommand[s.last.decoded]?.action}){" "}
                {s.last.executed === null ? <span className="pill">امتنع — ثقة منخفضة</span> : <span className="pill pill-info">نُفّذ</span>}
              </p>
              <div className="probs">
                {s.commands.map((c) => {
                  const p = s.last!.probs[c.command] ?? 0;
                  return (
                    <div key={c.command} className="prob-row">
                      <span className="prob-label">{c.imagine} <span className="faint small">({c.action})</span></span>
                      <div className="meter"><div className="meter-fill" style={{ width: `${Math.round(p * 100)}%` }} /></div>
                      <span className="num small">{Math.round(p * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            </>
          ) : <p className="muted">أول أمر بعد لحظات.</p>}
        </section>
        <section className="card">
          <h2>الجلسة</h2>
          <dl className="facts">
            <dt>الأوامر</dt><dd className="num">{s?.count ?? 0}</dd>
            <dt>الوقت</dt><dd className="num">{formatClock((s?.minutes ?? 0) * 60)}</dd>
          </dl>
          <QualityChips names={live.hello?.ch_names ?? []} quality={live.last?.quality} />
          <div className="row-actions">
            <button className="btn" onClick={onExit}><Square size={14} aria-hidden /> أوقف</button>
          </div>
        </section>
      </div>
    </>
  );
}

function CommandLegend({ commands, title }: { commands: CommandMeta[]; title: string }) {
  return (
    <div className="legend">
      <div className="small muted">{title}</div>
    <div className="legend-cards">
      {commands.map((c) => (
        <div key={c.command} className="legend-card">
          <span className="legend-icon">{ICON[c.command]}</span>
          <div>
            <div><strong>تخيّل {c.imagine}</strong></div>
            <div className="small muted">{c.action}</div>
          </div>
        </div>
      ))}
    </div>
    </div>
  );
}

function Simulation({ info, subjects }: { info: SpellerInfo; subjects: SpellerSubject[] }) {
  const [subject, setSubject] = useState<number | null>(null);
  const [target, setTarget] = useState(EXAMPLES[0]);
  const [speed, setSpeed] = useState(SPEEDS[1].ms);
  const [result, setResult] = useState<SimResult | null>(null);
  const [index, setIndex] = useState(-1); // آخر خطوة معروضة
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // أفضل شخص افتراضياً — ويمكن اختيار الأضعف لترى الفرق
  useEffect(() => {
    if (subject === null && subjects.length) {
      setSubject([...subjects].sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))[0].subject);
    }
  }, [subjects, subject]);

  useEffect(() => {
    if (!playing || !result) return;
    if (index >= result.steps.length - 1) { setPlaying(false); return; }
    const t = setTimeout(() => setIndex((i) => i + 1), speed);
    return () => clearTimeout(t);
  }, [playing, index, result, speed]);

  const start = async () => {
    if (subject === null) return;
    setLoading(true);
    setError(null);
    try {
      const r = await postJson<SimResult>("/api/speller/simulate",
        { subject, target, seed: Math.floor(Math.random() * 1_000_000) });
      setResult(r);
      setIndex(-1);
      setPlaying(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const step = result && index >= 0 ? result.steps[index] : null;
  const view: SpellerView = step?.view ?? info.initial;
  const done = result !== null && index === result.steps.length - 1;
  const current = subjects.find((s) => s.subject === subject);
  const commands = commandsFor(info, current?.mapping);

  if (!subjects.length) {
    return (
      <section className="card">
        <p>لا توجد نماذج مدرَّبة بعد. شغّل التدريب:</p>
        <code className="ltr">python -m tracks.motor_imagery.train</code>
      </section>
    );
  }

  return (
    <>
      <div className="banner banner-info">
        محاكاة صادقة: «المستخدم» يعرف الأمر الصحيح التالي، لكن ما يُنفَّذ هو ما فهمه النموذج من محاولة <strong>حقيقية</strong> لهذا
        الشخص من يوم لم يتدرّب عليه النموذج. أخطاء النموذج تُطبَّق كما هي، ثم يصححها المستخدم بأمر «رجوع».
      </div>

      <CommandLegend commands={commands}
        title={!isDefaultMapping(info, current?.mapping)
          ? "توزيع مخصّص لهذا الشخص: «اختيار» (أكثر أمر نستخدمه) لنوع التخيّل الأوضح عنده — اختير على يومه الأول فقط:"
          : "التوزيع الافتراضي — بقي لهذا الشخص لأن البديل لم يتفوق عليه بوضوح في يومه الأول:"} />

      <section className="card controls">
        <label>
          الشخص
          <select value={subject ?? ""} onChange={(e) => { setSubject(Number(e.target.value)); setResult(null); }}>
            {subjects.map((s) => (
              <option key={s.subject} value={s.subject}>
                الشخص {s.subject} — دقة {s.accuracy !== null ? Math.round(s.accuracy * 100) : "?"}%
              </option>
            ))}
          </select>
        </label>
        <label className="grow">
          ماذا يريد أن يكتب؟
          <input value={target} onChange={(e) => setTarget(e.target.value)} maxLength={40} />
        </label>
        <label>
          سرعة العرض
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => <option key={s.ms} value={s.ms}>{s.label}</option>)}
          </select>
        </label>
        <button className="btn btn-primary" onClick={start} disabled={loading || subject === null}>
          <Play size={15} aria-hidden /> {loading ? "…" : result ? "محاكاة جديدة" : "ابدأ"}
        </button>
        {result && !done && (
          <button className="btn" onClick={() => setPlaying((p) => !p)}>
            {playing ? <><Pause size={15} aria-hidden /> إيقاف</> : <><Play size={15} aria-hidden /> متابعة</>}
          </button>
        )}
      </section>
      <div className="chips">
        {EXAMPLES.map((e) => <button key={e} className="chip chip-btn" onClick={() => setTarget(e)}>{e}</button>)}
      </div>
      {current?.spelling && (
        <p className="small muted">
          في 20 محاكاة لكتابة «{current.spelling.target}» كان وسيط سرعة هذا الشخص{" "}
          <strong>{speedLabel(current.spelling.median_chars_per_minute, current.spelling.completion_rate)}</strong>
          {current.spelling_default_mapping && (
            <> (بالتوزيع الافتراضي حيث القدمان = اختيار:{" "}
              {speedLabel(current.spelling_default_mapping.median_chars_per_minute, current.spelling_default_mapping.completion_rate)})</>
          )}
          {" "}— بنموذج مثالي يحتاج {current.spelling.ideal_commands} أمراً فقط. {current.pipeline_ar && `الطريقة: ${current.pipeline_ar}.`}
        </p>
      )}
      {error && <div className="banner banner-bad">{error}</div>}

      <Board view={view} target={result?.target} />

      {result && (
        <div className="grid-2">
          <section className="card">
            <h2>الخطوة {index + 1} من {result.steps.length}</h2>
            {step ? <DecodeView step={step} commands={commands} /> : <p className="muted">تبدأ الآن…</p>}
          </section>
          <section className="card">
            <h2>حتى الآن</h2>
            <dl className="facts">
              <dt>الأوامر</dt><dd className="num">{index + 1}</dd>
              <dt>أخطاء النموذج</dt><dd className="num">{errorsUpTo(result.steps, index)}</dd>
              <dt>الوقت الحقيقي</dt>
              <dd className="num">{formatClock(simulatedSeconds(index + 1, result.metrics.seconds_per_command))}</dd>
            </dl>
            {done && <Summary result={result} />}
          </section>
        </div>
      )}
    </>
  );
}

function DecodeView({ step, commands }: { step: SimResult["steps"][number]; commands: CommandMeta[] }) {
  const byCommand = Object.fromEntries(commands.map((c) => [c.command, c]));
  const ok = step.executed === step.intended;
  return (
    <>
      <p>
        أراد المستخدم: <strong>تخيّل {byCommand[step.intended].imagine}</strong> ({byCommand[step.intended].action})
      </p>
      <p>
        فهم النموذج: <strong>{byCommand[step.decoded].imagine}</strong>{" "}
        {step.executed === null ? <span className="pill">امتنع — ثقة منخفضة</span>
          : ok ? <span className="pill pill-ok">صحيح</span> : <span className="pill pill-bad">خطأ</span>}
      </p>
      <div className="probs">
        {commands.map((c) => {
          const p = step.probs[CLASS_ORDER.indexOf(c.class)];
          return (
            <div key={c.command} className="prob-row">
              <span className="prob-label">{c.imagine} <span className="faint small">({c.action})</span></span>
              <div className="meter"><div className={`meter-fill ${c.command === step.intended ? "" : "dim"}`}
                style={{ width: `${Math.round(p * 100)}%` }} /></div>
              <span className="num small">{Math.round(p * 100)}%</span>
            </div>
          );
        })}
      </div>
      <p className="small faint">ثقة النموذج في كل أمر لهذه المحاولة (محاولة رقم {step.trial} من اليوم الثاني).</p>
    </>
  );
}

function Summary({ result }: { result: SimResult }) {
  const m = result.metrics;
  if (!result.finished) {
    return (
      <div className="banner banner-warn">
        توقفت المحاكاة بعد {m.commands} أمراً ({Math.round(m.minutes)} دقيقة) دون إكمال الجملة — أخطاء النموذج أكثر من
        قدرة المستخدم على تصحيحها. هذا يحدث فعلاً مع بعض الناس.
      </div>
    );
  }
  return (
    <div className="banner banner-ok">
      اكتملت في {m.commands} أمراً (المثالي {m.ideal_commands}) خلال {formatClock(m.minutes * 60)} ·{" "}
      {m.chars_per_minute?.toFixed(1)} حرف في الدقيقة.
    </div>
  );
}

function Manual({ info }: { info: SpellerInfo }) {
  const [view, setView] = useState<SpellerView>(info.initial);
  const [error, setError] = useState<string | null>(null);
  const [count, setCount] = useState(0);
  // الأوامر تُنفَّذ بالترتيب، كل أمر على نتيجة الذي قبله — وإلا ضاع أحدها عند الضغط السريع
  const viewRef = useRef(info.initial);
  const queue = useRef<Promise<void>>(Promise.resolve());

  const send = useCallback((command: SpellerCommand) => {
    queue.current = queue.current.then(async () => {
      try {
        const next = await postJson<SpellerView>("/api/speller/step", { state: viewRef.current, command });
        viewRef.current = next;
        setView(next);
        setCount((c) => c + 1);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    });
  }, []);

  const reset = () => {
    viewRef.current = info.initial;
    setView(info.initial);
    setCount(0);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      const c = commandForKey(e.key);
      if (c) { e.preventDefault(); send(c); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [send]);

  const seconds = useMemo(() => count * info.seconds_per_command, [count, info.seconds_per_command]);

  return (
    <>
      <CommandLegend commands={info.commands} title="التوزيع الافتراضي للأوامر:" />
      <div className="banner banner-info">
        هنا أنت «النموذج المثالي»: كل ضغطة تُنفَّذ كما أردت. استعمل الأزرار أو الأسهم ← → و Enter و Backspace.
        بالدماغ يأخذ كل أمر نحو {info.seconds_per_command} ثوانٍ — لو كتبت هذا بالتخيّل لاستغرق {formatClock(seconds)}.
      </div>
      <Board view={view} />
      <section className="card manual-buttons">
        {(["right", "left", "select", "back"] as SpellerCommand[]).map((c) => {
          const meta = info.commands.find((x) => x.command === c)!;
          return (
            <button key={c} className="btn btn-big" onClick={() => send(c)}>
              <span className="legend-icon">{ICON[c]}</span> {meta.action}
              <span className="small muted">تخيّل {meta.imagine}</span>
            </button>
          );
        })}
        <button className="btn" onClick={reset}>
          <RotateCcw size={15} aria-hidden /> من جديد
        </button>
      </section>
      {error && <div className="banner banner-bad">{error}</div>}
    </>
  );
}

function Board({ view, target }: { view: SpellerView; target?: string }) {
  return (
    <section className="card board">
      <div className="typed" aria-live="polite">
        {view.text || <span className="faint">…</span>}<span className="caret" />
      </div>
      {target && <div className="small faint">الهدف: {target}</div>}
      <div className="small muted board-level">
        {view.group === null ? "اختر مجموعة حروف أو كلمة مقترحة" : "اختر حرفاً — «رجوع» يعيدك للمجموعات"}
      </div>
      <div className="tiles">
        {view.items.map((it, i) => (
          <div key={`${it.kind}-${it.value}`}
            className={`tile tile-${it.kind} ${i === view.cursor ? "tile-cursor" : ""}`}>
            {it.kind === "group" ? it.label : it.kind === "char" ? (it.value === " " ? "مسافة" : it.value) : it.value}
          </div>
        ))}
      </div>
    </section>
  );
}
