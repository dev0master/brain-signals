import { useQuery } from "@tanstack/react-query";
import { FileUp, Radio } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { SeizureTimeline } from "../components/SeizureTimeline";
import { getJson } from "../lib/api";
import { formatClock } from "../lib/liveStore";
import { zoomAround, type Range } from "../lib/timeline";
import type { EpiRecording, EpiResult, EventMetrics, ModelTimeline } from "../lib/types";

type ModelKey = "patient_specific" | "general";

const MODEL_INFO: Record<ModelKey, { name: string; hint: string }> = {
  patient_specific: {
    name: "نموذج خاص بالمريض",
    hint: "تدرّب على تسجيلات أخرى لنفس الطفل — لم يرَ هذا التسجيل أبداً.",
  },
  general: {
    name: "نموذج عام (شخص جديد)",
    hint: "تدرّب على مرضى آخرين فقط — كأنه يرى هذا الطفل لأول مرة. أصعب بكثير.",
  },
};

export function EpilepsyPage() {
  const recordings = useQuery({ queryKey: ["epi-recordings"], queryFn: () => getJson<EpiRecording[]>("/api/epilepsy/recordings") });
  const [selected, setSelected] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<EpiResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // أول تسجيل فيه نوبة يُفتح تلقائياً
  useEffect(() => {
    if (!selected && !uploaded && recordings.data?.length) {
      setSelected((recordings.data.find((r) => r.n_seizures > 0 && r.analyzed) ?? recordings.data[0]).file);
    }
  }, [recordings.data, selected, uploaded]);

  const detail = useQuery({
    queryKey: ["epi-recording", selected],
    queryFn: () => getJson<EpiResult>(`/api/epilepsy/recordings/${selected}`),
    enabled: !!selected && !uploaded,
  });
  const result = uploaded ?? detail.data ?? null;

  const upload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/epilepsy/analyze", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `خطأ (${res.status})`);
      setUploaded(data as EpiResult);
      setSelected(null);
    } catch (e) {
      setUploadError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const byPatient = useMemo(() => {
    const groups: Record<string, EpiRecording[]> = {};
    for (const r of recordings.data ?? []) (groups[r.patient] ??= []).push(r);
    return groups;
  }, [recordings.data]);

  return (
    <div className="page">
      <header className="page-head">
        <h1>فحص الصرع</h1>
        <p className="muted">
          يمرّ النموذج على التسجيل كل ثانيتين ويقدّر احتمال أن تكون اللحظة نوبة صرع، ثم يُطلق إنذاراً إذا تكرر
          الاشتباه في 3 من آخر 4 نوافذ. بجانب ذلك نعرض النوبات التي حددها الأطباء لتقارن بنفسك.
        </p>
      </header>
      <div className="banner banner-warn">
        أداة تعليمية وبحثية — ليست تشخيصاً. تشخيص الصرع يقوم به طبيب أعصاب بعد فحص كامل وتسجيلات أطول بكثير.
      </div>
      <p className="small muted">
        بجهازك: بعد «تجهيز النماذج» في صفحة <Link to="/device">جهازك</Link> يعمل الكاشف في{" "}
        <Link to="/?source=device">البث المباشر</Link>، ويمكنك رفع تسجيلاتك المحفوظة هنا. يستخدم النموذج العام فقط —
        لا يوجد نموذج يعرف نوبات شخصك.
      </p>

      <div className="epi-layout">
        <aside className="card epi-list">
          <h2>التسجيلات المحفوظة</h2>
          {recordings.data?.length === 0 && (
            <p className="small muted">لا توجد تسجيلات. نزّلها: <code className="ltr">python -m scripts.download_data chbmit</code></p>
          )}
          {Object.entries(byPatient).map(([patient, recs]) => (
            <div key={patient} className="epi-group">
              <div className="small faint">المريض <span className="ltr">{patient}</span></div>
              {recs.map((r) => (
                <button key={r.file} className={`epi-item ${selected === r.file && !uploaded ? "active" : ""}`}
                  disabled={!r.analyzed} title={r.analyzed ? "" : "لم يُحلَّل بعد — شغّل التدريب"}
                  onClick={() => { setUploaded(null); setSelected(r.file); }}>
                  <span className="ltr">{r.file.replace(".edf", "")}</span>
                  {r.n_seizures > 0 ? <span className="pill pill-bad">{r.n_seizures} نوبة</span> : <span className="faint small">بلا نوبات</span>}
                </button>
              ))}
            </div>
          ))}
          <hr className="sep" />
          <label className="upload">
            <FileUp size={16} aria-hidden /> {uploading ? "جارٍ التحليل… (ساعة تسجيل ≈ 20 ثانية)" : "ارفع ملف EDF للتحليل"}
            <input type="file" accept=".edf" disabled={uploading}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = ""; }} />
          </label>
          <p className="small faint">يجب أن يحتوي الأقطاب الثنائية الـ18 القياسية (مثل تسجيلات CHB-MIT). يُحلَّل بالنموذج العام.</p>
          {uploadError && <div className="banner banner-bad small">{uploadError}</div>}
        </aside>

        <div className="epi-main">
          {detail.error && !uploaded && <div className="banner banner-bad">{(detail.error as Error).message}</div>}
          {result ? <ResultView key={result.file + (uploaded ? "-up" : "")} result={result} /> : (
            <section className="card"><p className="muted">اختر تسجيلاً من القائمة.</p></section>
          )}
        </div>
      </div>
    </div>
  );
}

function ResultView({ result }: { result: EpiResult }) {
  const available = (["patient_specific", "general"] as ModelKey[]).filter((k) => result[k]);
  const [modelKey, setModelKey] = useState<ModelKey>(available[0]);
  const full: Range = { from: 0, to: result.duration_sec };
  const [view, setView] = useState<Range>(full);
  const model = result[modelKey] as ModelTimeline;
  const firstSeizure = result.seizures[0];
  const zoomed = view.from !== 0 || view.to !== result.duration_sec;

  return (
    <section className="card">
      <div className="result-head">
        <h2 className="ltr">{result.file}</h2>
        <span className="muted">{formatClock(result.duration_sec)} من التسجيل</span>
        <span className="spacer" />
        {available.length > 1 && (
          <div className="seg">
            {available.map((k) => (
              <button key={k} className={k === modelKey ? "active" : ""} onClick={() => setModelKey(k)}>{MODEL_INFO[k].name}</button>
            ))}
          </div>
        )}
      </div>
      <p className="small muted">{MODEL_INFO[modelKey].hint}</p>
      {result.warnings?.map((w) => <div key={w} className="banner banner-warn small">{w}</div>)}

      {model.metrics ? <MetricsLine m={model.metrics} /> : <UploadSummary model={model} />}

      <div className="tl-controls">
        {zoomed && <button className="btn" onClick={() => setView(full)}>عرض التسجيل كاملاً</button>}
        {!zoomed && firstSeizure && <button className="btn" onClick={() => setView(zoomAround(firstSeizure, result.duration_sec))}>تكبير حول النوبة</button>}
        <span className="small faint">اضغط على أي شريط ملوّن للتكبير حوله.</span>
        <span className="spacer" />
        {result.source_id && (
          <Link className="btn" to={`/?source=${encodeURIComponent(result.source_id)}&start=${Math.max(0, Math.floor((firstSeizure?.start ?? 0) - 30))}`}>
            <Radio size={15} aria-hidden /> {firstSeizure ? "شاهد النوبة في البث المباشر" : "افتح في البث المباشر"}
          </Link>
        )}
      </div>
      <SeizureTimeline model={model} seizures={result.seizures} view={view}
        onZoom={(ev) => setView(zoomAround(ev, result.duration_sec))} />
    </section>
  );
}

function MetricsLine({ m }: { m: EventMetrics }) {
  const parts: string[] = [];
  if (m.n_seizures === 0) parts.push("لا توجد نوبات في هذا التسجيل");
  else parts.push(`اكتُشفت ${m.detected} من ${m.n_seizures} نوبة`);
  if (m.mean_latency_sec !== null) parts.push(`بعد ${Math.round(m.mean_latency_sec)} ثانية من بدايتها`);
  parts.push(m.false_alarms === 0 ? "بلا إنذارات كاذبة" : `${m.false_alarms} إنذار كاذب`);
  const good = m.detected === m.n_seizures && m.false_alarms === 0;
  return <div className={`banner ${good ? "banner-ok" : "banner-warn"}`}>{parts.join(" · ")}</div>;
}

function UploadSummary({ model }: { model: ModelTimeline }) {
  const n = model.detected.length;
  return (
    <div className={`banner ${n ? "banner-bad" : "banner-ok"}`}>
      {n === 0 ? "لم يُطلق النموذج أي إنذار في هذا التسجيل." :
        `أطلق النموذج ${n} إنذار: ${model.detected.map((e) => formatClock(e.alarm ?? e.start)).join("، ")}. لا نعرف الإجابة الصحيحة لهذا الملف للمقارنة.`}
    </div>
  );
}
