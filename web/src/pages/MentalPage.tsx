import { useQuery } from "@tanstack/react-query";
import { Pause, Play, Radio } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Gauge } from "../components/Gauge";
import { StateTimeline } from "../components/StateTimeline";
import { getJson } from "../lib/api";
import { formatClock } from "../lib/liveStore";
import { describe, segmentAt, smooth, valueAt } from "../lib/mental";
import type { MentalScale, MentalTimeline } from "../lib/types";

const PLAY_RATE = 10; // كل ثانية عرض = 10 ثوانٍ من التسجيل
const SMOOTH_WINDOWS = 3; // المؤشر يعرض متوسط آخر 3 نوافذ (6 ثوانٍ)

const pct = (v: number) => `${Math.round(v * 100)}%`;

export function MentalPage() {
  const scales = useQuery({ queryKey: ["mental-scales"], queryFn: () => getJson<MentalScale[]>("/api/mental/scales") });
  const [scaleId, setScaleId] = useState<string | null>(null);
  const scale = scales.data?.find((s) => s.id === scaleId) ?? scales.data?.[0];

  return (
    <div className="page">
      <header className="page-head">
        <h1>الحالة الذهنية</h1>
        <p className="muted">
          مقياسان يقرآن من إشارة الدماغ: هل الشخص مرتاح أم منشغل بمهمة ذهنية، وهل هو مسترخٍ أم تحت ضغط مهمة مجهِدة.
          يبدأ كل قياس بـ«معايرة»: لحظات راحة نعرف بها إشارة هذا الشخص وهو هادئ، ثم نقيس كم ابتعد عنها.
        </p>
      </header>
      <div className="banner banner-warn">
        هذا ليس قراءة للمشاعر ولا للأفكار. المقياس يقول فقط: «إشارتك الآن تشبه إشارة من ينشغل بمهمة ذهنية أكثر مما تشبه
        الراحة». ويخطئ — انظر أرقامه أدناه.
      </div>

      {scales.data && scales.data.length === 0 && (
        <section className="card">
          <p>لا توجد نماذج مدرَّبة بعد. شغّل:</p>
          <code className="ltr">python -m tracks.mental_state.train</code>
        </section>
      )}
      {scales.data && scales.data.length > 1 && (
        <div className="seg">
          {scales.data.map((s) => (
            <button key={s.id} className={s.id === scale?.id ? "active" : ""} onClick={() => setScaleId(s.id)}>{s.title}</button>
          ))}
        </div>
      )}
      {scale && <ScaleView key={scale.id} scale={scale} />}
    </div>
  );
}

function ScaleView({ scale }: { scale: MentalScale }) {
  // الشخص الافتراضي: أقرب شخص إلى الوسيط — لا أفضل حالة منتقاة
  const typical = useMemo(() => {
    const rows = Object.values(scale.per_subject).sort((a, b) => a.auc - b.auc);
    return rows[Math.floor(rows.length / 2)]?.subject ?? scale.subjects[0];
  }, [scale]);
  const [subject, setSubject] = useState(typical);
  const tl = useQuery({
    queryKey: ["mental-tl", scale.id, subject],
    queryFn: () => getJson<MentalTimeline>(`/api/mental/${scale.id}/${subject}`),
  });

  return (
    <>
      <ScaleSummary scale={scale} />
      <section className="card controls">
        <label>
          الشخص
          <select value={subject} onChange={(e) => setSubject(e.target.value)}>
            {scale.subjects.map((s) => {
              const r = scale.per_subject[s];
              return <option key={s} value={s}>{s} — AUC {r ? r.auc.toFixed(2) : "?"}{s === typical ? " (نموذجي)" : ""}</option>;
            })}
          </select>
        </label>
        <span className="small muted">كل الأشخاص هنا اختُبروا بنموذج لم يرَ إشارتهم. «نموذجي» = في منتصف الترتيب، لا الأفضل.</span>
      </section>
      {tl.data && <Player key={tl.data.subject} tl={tl.data} scale={scale} />}
      {scale.top_features.length > 0 && <Features scale={scale} />}
    </>
  );
}

function ScaleSummary({ scale }: { scale: MentalScale }) {
  const wc = scale.without_calibration;
  return (
    <section className="card">
      <h2>{scale.title}: ما مدى صحة المقياس؟</h2>
      <p className="small muted">{scale.dataset}</p>
      <div className="vs-chance">
        <div className="metric"><div className="metric-value num">{pct(scale.subjects_rising)}</div>
          <div className="small muted">يرتفع المقياس أثناء المهمة عند هذه النسبة من الأشخاص</div></div>
        <div className="metric"><div className="metric-value num">{scale.auc.toFixed(2)}</div>
          <div className="small muted">AUC (0.5 صدفة · 1.0 فصل تام)</div></div>
        <div className="metric"><div className="metric-value num">{pct(scale.balanced_accuracy)}</div>
          <div className="small muted">الدقة المتوازنة لكل نافذة 4 ثوانٍ (الصدفة 50%)</div></div>
        {wc && (
          <div className="metric metric-muted"><div className="metric-value num">{wc.auc.toFixed(2)}</div>
            <div className="small muted">AUC بلا معايرة — للمقارنة</div></div>
        )}
        {scale.rating_agreement && (
          <div className="metric metric-muted"><div className="metric-value num">{scale.rating_agreement.mean_within_subject >= 0 ? "+" : ""}{scale.rating_agreement.mean_within_subject.toFixed(2)}</div>
            <div className="small muted">توافقه مع التوتر الذي قال الشخص إنه شعر به (0 = لا علاقة)</div></div>
        )}
      </div>
      {scale.validity_notes.map((n) => <div key={n} className="banner banner-warn small">{n}</div>)}
    </section>
  );
}

function Player({ tl, scale }: { tl: MentalTimeline; scale: MentalScale }) {
  const values = useMemo(() => smooth(tl.calibrated, SMOOTH_WINDOWS), [tl]);
  const calibEnd = tl.segments.find((s) => s.kind === "calibration")?.end ?? 0;
  const end = Math.max(...tl.segments.map((s) => s.end));
  // نبدأ عند أول قراءة متاحة (أول نافذة تكتمل بعد المعايرة) لا عند نهاية المعايرة بالضبط
  const [cursor, setCursor] = useState(tl.times[0] ?? calibEnd);
  const [playing, setPlaying] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setCursor((c) => {
        if (c >= end) { setPlaying(false); return end; }
        return Math.min(end, c + PLAY_RATE / 10);
      });
    }, 100);
    return () => clearInterval(id);
  }, [playing, end]);

  const value = valueAt(tl.times, values, cursor);
  const seg = segmentAt(tl.segments, cursor);
  const caption = seg?.kind === "calibration"
    ? "معايرة — يتعرّف النظام على إشارتك وأنت هادئ"
    : value === null ? "…" : `القراءة: ${describe(value, scale.low, scale.high)}`;

  return (
    <section className="card">
      <div className="player">
        <div className="player-gauge">
          <Gauge value={seg?.kind === "calibration" ? null : value} low={scale.low} high={scale.high} caption={caption} />
          <div className="truth-box">
            <div className="small muted">ما كان يفعله الشخص فعلاً</div>
            <div className={`pill ${seg?.kind === "task" ? "pill-warn" : seg?.kind === "calm" ? "pill-ok" : ""}`}>{seg?.label ?? "—"}</div>
            <div className="num small faint">{formatClock(cursor)}</div>
          </div>
        </div>
        <div className="player-controls">
          <button className="btn btn-primary" onClick={() => {
            if (cursor >= end) setCursor(calibEnd);
            setPlaying((p) => !p);
          }}>
            {playing ? <><Pause size={15} aria-hidden /> إيقاف</> : <><Play size={15} aria-hidden /> شغّل ({PLAY_RATE}×)</>}
          </button>
          <label className="check">
            <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
            أظهر النموذج بلا معايرة (خط متقطع)
          </label>
          <span className="spacer" />
          {tl.sources?.map((s) => (
            <Link key={s.source_id} className="btn" to={`/?source=${encodeURIComponent(s.source_id)}`}>
              <Radio size={15} aria-hidden /> {s.label} في البث المباشر
            </Link>
          ))}
        </div>
      </div>
      <StateTimeline times={tl.times} values={values} compare={showRaw ? smooth(tl.raw, SMOOTH_WINDOWS) : undefined}
        segments={tl.segments} cursor={cursor} onSeek={(t) => { setCursor(t); setPlaying(false); }} />
      <p className="small faint">
        اضغط على الخط الزمني للانتقال إلى أي لحظة. الخط الأزرق: قراءة المقياس (متوسط آخر {SMOOTH_WINDOWS * 2} ثوانٍ)؛
        الخط الأصفر المتقطع: المنتصف. {scale.id === "mental_stress" && "محاولات هذه البيانات منفصلة، وترتيبها هنا للعرض فقط."}
      </p>
    </section>
  );
}

function Features({ scale }: { scale: MentalScale }) {
  return (
    <section className="card">
      <h2>ما الذي يحرّك المقياس؟</h2>
      <p className="small muted">أقوى ما اعتمد عليه النموذج النهائي (بعد المعايرة)، مرتبة حسب قوة التأثير:</p>
      <ul className="features">
        {scale.top_features.map((f) => (
          <li key={f.feature}>
            <span className={f.weight > 0 ? "up" : "down"}>{f.weight > 0 ? "▲" : "▼"}</span>
            <strong>{f.feature}</strong> <span className="muted small">— {f.direction}</span>
          </li>
        ))}
      </ul>
      <p className="small faint">
        الارتباط ليس سبباً: النموذج يلتقط ما يختلف بين الحالتين في هذه البيانات، وقد يشمل ذلك أثر الحركة أو التسجيل
        لا الدماغ وحده.
      </p>
    </section>
  );
}
