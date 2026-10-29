import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { BandBars } from "../components/BandBars";
import { QualityChips } from "../components/QualityChips";
import { SignalCanvas } from "../components/SignalCanvas";
import { getJson } from "../lib/api";
import { SOURCE_KIND } from "../lib/labels";
import { eventAt, formatClock, meanBands } from "../lib/liveStore";
import type { MentalLive, SeizureLive, SourceItem } from "../lib/types";
import { useLiveStream, type StreamStatus } from "../lib/useLiveStream";

const STATUS: Record<StreamStatus, { text: string; tone: string }> = {
  connecting: { text: "جارٍ الاتصال…", tone: "info" },
  open: { text: "متصل", tone: "ok" },
  closed: { text: "انقطع الاتصال", tone: "warn" },
  error: { text: "خطأ", tone: "bad" },
};

export function LivePage() {
  const sources = useQuery({ queryKey: ["sources"], queryFn: () => getJson<SourceItem[]>("/api/sources") });
  // المصدر ونقطة البدء في الرابط، فتستطيع صفحة الصرع فتح نوبة معيّنة هنا مباشرة
  const [params, setParams] = useSearchParams();
  const source = params.get("source") ?? "synthetic";
  const startSec = Number(params.get("start") ?? 0) || 0;
  const [speed, setSpeed] = useState(1);
  const [scale, setScale] = useState(100);
  const live = useLiveStream(source, speed, startSec);
  const { hello, last } = live;
  const t = last?.t ?? startSec;
  const inEvent = hello ? eventAt(hello.events, t) : null;
  const bands = useMemo(() => (last?.bands ? meanBands(last.bands, last.quality) : []), [last]);
  const seizure = last?.seizure as SeizureLive | undefined;
  const mental = last?.mental as MentalLive | undefined;
  const status = STATUS[live.status];
  const options = sources.data ?? [];
  const known = options.some((s) => s.id === source);

  return (
    <div className="page">
      <header className="page-head">
        <h1>البث المباشر</h1>
        <p className="muted">
          هنا ترى إشارات الدماغ كما تصل لحظة بلحظة: كل سطر قطب على الرأس، والزمن يتقدم نحو اليمين.
          لا يوجد جهاز متصل الآن، فنعيد تشغيل تسجيلات حقيقية بسرعة الزمن الفعلي — وسيعمل كل شيء بنفس الطريقة مع جهاز.
        </p>
      </header>

      <section className="card controls">
        <label>
          المصدر
          <select value={source} onChange={(e) => setParams({ source: e.target.value })}>
            {!known && <option value={source}>{source.split("/").pop()}</option>}
            {options.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        </label>
        <label>
          السرعة
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {[1, 2, 5, 10].map((v) => <option key={v} value={v}>{v === 1 ? "الزمن الحقيقي" : `${v}×`}</option>)}
          </select>
        </label>
        <label>
          التكبير
          <select value={scale} onChange={(e) => setScale(Number(e.target.value))}>
            {[25, 50, 100, 200, 500].map((v) => <option key={v} value={v}>{v} µV</option>)}
          </select>
        </label>
        <div className="spacer" />
        <span className={`pill pill-${status.tone}`}>{status.text}</span>
        {hello && <span className="pill">{SOURCE_KIND[hello.kind]}</span>}
        <span className="num pill">{formatClock(t)}{hello?.duration_sec ? ` / ${formatClock(hello.duration_sec)}` : ""}</span>
      </section>

      {live.error && <div className="banner banner-bad">{live.error}</div>}
      {(seizure || inEvent) && (
        <section className="card seizure-live">
          <div>
            <div className="small muted">كاشف الصرع — نموذج لم يرَ هذا التسجيل أثناء تدريبه</div>
            {seizure ? (
              <div className="seizure-row">
                <div className="meter"><div className={`meter-fill ${seizure.prob >= seizure.threshold ? "hot" : ""}`}
                  style={{ width: `${Math.round(seizure.prob * 100)}%` }} /></div>
                <span className="num">{Math.round(seizure.prob * 100)}%</span>
                <span className={`pill ${seizure.alarm ? "pill-bad" : "pill-ok"}`}>{seizure.alarm ? "إنذار: يشتبه بنوبة" : "لا إنذار"}</span>
              </div>
            ) : <p className="small faint">لا توجد توقعات لهذا التسجيل — شغّل التدريب.</p>}
          </div>
          <div className="truth">
            <div className="small muted">ما حدده الأطباء</div>
            {inEvent
              ? <span className="pill pill-bad">نوبة الآن ({formatClock(inEvent.start)} – {formatClock(inEvent.end)})</span>
              : <span className="pill">لا نوبة في هذه اللحظة</span>}
          </div>
        </section>
      )}

      {mental && (
        <section className="card seizure-live">
          <div>
            <div className="small muted">المقياس الذهني — نموذج لم يرَ هذا الشخص</div>
            {mental.calibrating ? (
              <p className="small">معايرة: النظام يتعرّف على إشارة الشخص وهو مرتاح… باقٍ <span className="num">{mental.remaining_sec}</span> ث</p>
            ) : mental.value !== undefined && (
              <div className="seizure-row">
                <span className="small">{mental.low}</span>
                <div className="meter"><div className="meter-fill" style={{ width: `${Math.round(mental.value * 100)}%` }} /></div>
                <span className="small">{mental.high}</span>
              </div>
            )}
          </div>
          {mental.truth && (
            <div className="truth">
              <div className="small muted">ما كان يفعله فعلاً</div>
              <span className="pill">{mental.truth}</span>
            </div>
          )}
        </section>
      )}

      <section className="card">
        {hello ? (
          <SignalCanvas
            bufferRef={live.bufferRef}
            timeRef={live.timeRef}
            chNames={hello.ch_names}
            viewSec={live.viewSec}
            scaleUv={scale}
            events={hello.events}
          />
        ) : (
          <p className="muted">ننتظر الخادم… تأكد أنه يعمل: <code className="ltr">python -m uvicorn server.main:app --port 8770</code></p>
        )}
        {hello && (
          <p className="faint small">
            {hello.description} · {hello.ch_names.length} قناة · {hello.fs} عيّنة/ثانية · الإشارة مفلترة 0.5–45Hz مع حذف تشويش الكهرباء
          </p>
        )}
      </section>

      <div className="grid-2">
        <section className="card">
          <h2>قدرة نطاقات الترددات</h2>
          <p className="muted small">نصيب كل نطاق من طاقة الإشارة في آخر ثانيتين (متوسط الأقطاب السليمة).</p>
          <BandBars names={hello?.bands ?? []} values={bands} />
        </section>
        <section className="card">
          <h2>جودة الأقطاب</h2>
          <QualityChips names={hello?.ch_names ?? []} quality={last?.quality} />
        </section>
      </div>
    </div>
  );
}
