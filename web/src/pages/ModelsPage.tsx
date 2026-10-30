import { useQuery } from "@tanstack/react-query";
import { getJson } from "../lib/api";
import { unusable } from "../lib/metrics";
import { speedLabel } from "../lib/speller";
import type { EventMetrics, ModelReport, SpellingStats } from "../lib/types";

type ClassSummary = {
  n: number;
  labels: string[];
  accuracy: number;
  balanced_accuracy: number;
  kappa: number;
  chance_uniform: number;
  chance_majority: number;
  confusion_matrix: number[][];
};

const pct = (v: number) => `${Math.round(v * 100)}%`;

// ترتيب العرض: الأقوى والأبسط أولاً، ثم الأصعب
const DISPLAY_ORDER = ["epilepsy_patient_specific", "epilepsy_general", "motor_imagery", "mental_focus", "mental_stress",
  // نماذج جهازك (بعد «جهّز النماذج» والمعايرة)
  "device_speller", "device_mental_focus", "device_mental_stress", "device_epilepsy"];
const rank = (id: string) => (DISPLAY_ORDER.includes(id) ? DISPLAY_ORDER.indexOf(id) : DISPLAY_ORDER.length);
const byDisplayOrder = (a: ModelReport, b: ModelReport) => rank(a.model_id) - rank(b.model_id);

/** صفحة النماذج: تعرض تقارير JSON المحفوظة كما هي — لا أرقام مكتوبة باليد. */
export function ModelsPage() {
  const q = useQuery({ queryKey: ["models"], queryFn: () => getJson<ModelReport[]>("/api/models") });

  return (
    <div className="page">
      <header className="page-head">
        <h1>النماذج ونتائجها</h1>
        <p className="muted">
          كل نموذج مدرَّب يظهر هنا بنتائجه الحقيقية من الاختبار على بيانات لم يرها أثناء التدريب، وبجانبها
          "مستوى الصدفة" — النتيجة التي يحققها التخمين العشوائي. النموذج المفيد يتفوق عليه بفرق واضح.
        </p>
      </header>
      {q.isLoading && <p className="muted">جارٍ التحميل…</p>}
      {q.error && <div className="banner banner-bad">{(q.error as Error).message}</div>}
      {q.data && q.data.length === 0 && (
        <section className="card">
          <p>لا توجد نماذج مدرَّبة بعد. ستظهر هنا بعد تشغيل التدريب في كل مرحلة.</p>
        </section>
      )}
      {q.data && [...q.data].sort(byDisplayOrder).map((r) => <ReportCard key={r.model_id} report={r} />)}
    </div>
  );
}

function ReportCard({ report }: { report: ModelReport }) {
  const summary = report.summary as ClassSummary | undefined;
  const events = report.seizure_events as EventMetrics | undefined;
  const notes = report.notes as string[] | undefined;
  return (
    <section className="card">
      <h2>{report.title ?? report.model_id}</h2>
      <dl className="facts">
        {report.dataset && (<><dt>البيانات</dt><dd>{report.dataset}</dd></>)}
        {report.evaluation && (<><dt>طريقة الاختبار</dt><dd>{report.evaluation}</dd></>)}
        <dt>تاريخ التدريب</dt>
        <dd><span className="num">{new Date(report.created_at).toLocaleString("en-GB", { dateStyle: "short", timeStyle: "short" })}</span></dd>
      </dl>
      {events ? (
        <SeizureEventsView m={events} perPatient={report.per_patient as PatientRow[] | undefined} />
      ) : summary && <ClassificationView s={summary} />}
      {events && summary && (
        <details className="more">
          <summary>على مستوى النوافذ (كل 4 ثوانٍ وحدها، قبل التنعيم)</summary>
          <ClassificationView s={summary} />
        </details>
      )}
      {report.track === "motor_imagery" && Array.isArray(report.per_subject) && (
        <SubjectsTable rows={report.per_subject as SubjectRow[]} />
      )}
      {Array.isArray(report.validity_notes) &&
        (report.validity_notes as string[]).map((n) => <div key={n} className="banner banner-warn small">{n}</div>)}
      {notes && <ul className="notes small muted">{notes.map((n) => <li key={n}>{n}</li>)}</ul>}
    </section>
  );
}

type SubjectRow = {
  subject: number;
  pipeline_ar: string;
  inner_cv: Record<string, number>;
  test_accuracy: number;
  kappa: number;
  spelling?: SpellingStats;
  spelling_default_mapping?: SpellingStats;
};

/** نتائج كل شخص في الكتابة بالدماغ: اليوم الأول (اختيار الطريقة) ثم اليوم الثاني (الاختبار). */
function SubjectsTable({ rows }: { rows: SubjectRow[] }) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>الشخص</th><th>الطريقة المختارة</th><th>اليوم الأول (تقسيم داخلي)</th><th>اليوم الثاني</th>
          <th>كتابة «{rows[0]?.spelling?.target}»</th><th>بالتوزيع الافتراضي</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.subject}>
            <td className="num">{r.subject}</td>
            <td className="small">{r.pipeline_ar}</td>
            <td className="num">{pct(Math.max(...Object.values(r.inner_cv)))}</td>
            <td className="num"><strong>{pct(r.test_accuracy)}</strong></td>
            <td className="small">{speedLabel(r.spelling?.median_chars_per_minute, r.spelling?.completion_rate)}</td>
            <td className="small faint">
              {speedLabel(r.spelling_default_mapping?.median_chars_per_minute, r.spelling_default_mapping?.completion_rate)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type PatientRow = EventMetrics & { patient: string; files: number };

function SeizureEventsView({ m, perPatient }: { m: EventMetrics; perPatient?: PatientRow[] }) {
  const lat = (v: number | null) => (v === null ? "—" : `${Math.round(v)} ث`);
  const fa = (v: number | null) => (v === null ? "—" : v.toFixed(2));
  return (
    <>
      <div className="vs-chance">
        <Metric label="الحساسية (نوبات اكتُشفت)" value={m.sensitivity === null ? "—" : pct(m.sensitivity)} />
        <Metric label="إنذارات كاذبة في الساعة" value={fa(m.false_alarms_per_hour)} />
        <Metric label="متوسط زمن الاكتشاف" value={lat(m.mean_latency_sec)} />
        <Metric label="ساعات الاختبار" value={m.hours.toFixed(1)} muted />
      </div>
      <p className="small muted">
        اكتُشفت {m.detected} من {m.n_seizures} نوبة، مع {m.false_alarms} إنذار كاذب في {m.hours.toFixed(1)} ساعة.
        للمقارنة: نموذج يقول «لا نوبة» دائماً حساسيته 0%.
      </p>
      {unusable(m) && (
        <div className="banner banner-warn">
          غير صالح للاستخدام العملي: {(m.sensitivity ?? 0) < 0.5 && "يفوّت أكثر من نصف النوبات"}
          {(m.sensitivity ?? 0) < 0.5 && (m.false_alarms_per_hour ?? 0) > 1 && "، و"}
          {(m.false_alarms_per_hour ?? 0) > 1 && "يُنذر بلا سبب أكثر من مرة في الساعة"}.
        </div>
      )}
      {perPatient && (
        <table className="table">
          <thead><tr><th>المريض</th><th>التسجيلات</th><th>اكتُشفت</th><th>إنذارات كاذبة/ساعة</th><th>زمن الاكتشاف</th></tr></thead>
          <tbody>
            {perPatient.map((p) => (
              <tr key={p.patient}>
                <td className="ltr">{p.patient}</td>
                <td className="num">{p.files}</td>
                <td className="num">{p.detected}/{p.n_seizures}</td>
                <td className="num">{fa(p.false_alarms_per_hour)}</td>
                <td className="num">{lat(p.mean_latency_sec)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

function ClassificationView({ s }: { s: ClassSummary }) {
  // الدقة المتوازنة (متوسط الدقة لكل فئة) لا تخدعها الفئة النادرة، ومستوى صدفتها 1/عدد الفئات
  return (
    <>
      <div className="vs-chance">
        <Metric label="الدقة المتوازنة" value={pct(s.balanced_accuracy)} />
        <Metric label="مستوى الصدفة" value={pct(s.chance_uniform)} muted />
        <Metric label="الدقة" value={pct(s.accuracy)} />
        <Metric label="عدد عيّنات الاختبار" value={String(s.n)} muted />
      </div>
      {s.balanced_accuracy <= s.chance_uniform + 0.05 && (
        <div className="banner banner-warn">الدقة المتوازنة قريبة من مستوى الصدفة — هذا النموذج لا يكاد يتعلم شيئاً مفيداً.</div>
      )}
      <ConfusionMatrix labels={s.labels} cm={s.confusion_matrix} />
    </>
  );
}

function Metric({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className={`metric ${muted ? "metric-muted" : ""}`}>
      <div className="metric-value num">{value}</div>
      <div className="small muted">{label}</div>
    </div>
  );
}

function ConfusionMatrix({ labels, cm }: { labels: string[]; cm: number[][] }) {
  const max = Math.max(1, ...cm.flat());
  return (
    <div className="cm-wrap">
      <p className="small muted">مصفوفة الالتباس: الصفوف = الحقيقة، الأعمدة = ما قاله النموذج. القطر = الإجابات الصحيحة.</p>
      <table className="cm" dir="ltr">
        <thead>
          <tr><th />{labels.map((l) => <th key={l}>{l}</th>)}</tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={labels[i]}>
              <th>{labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} className={i === j ? "cm-diag" : ""} style={{ ["--a" as string]: v / max }}>{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
