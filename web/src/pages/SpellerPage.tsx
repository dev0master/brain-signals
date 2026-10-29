import { useQuery } from "@tanstack/react-query";
import { Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getJson } from "../lib/api";
import { formatClock } from "../lib/liveStore";
import { commandForKey, commandsFor, errorsUpTo, isDefaultMapping, simulatedSeconds, speedLabel } from "../lib/speller";
import type { CommandMeta, SimResult, SpellerCommand, SpellerInfo, SpellerSubject, SpellerView } from "../lib/types";

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
  const [mode, setMode] = useState<"sim" | "manual">("sim");

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
      </div>

      {info.error && <div className="banner banner-bad">{(info.error as Error).message}</div>}
      {info.data && (mode === "sim"
        ? <Simulation info={info.data} subjects={subjects.data ?? []} />
        : <Manual info={info.data} />)}
    </div>
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
