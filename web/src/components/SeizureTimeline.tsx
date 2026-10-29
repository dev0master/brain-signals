import { useState } from "react";
import { formatClock } from "../lib/liveStore";
import { nearestIndex, timeTicks, visibleIndices, type Range } from "../lib/timeline";
import type { DetectedEvent, KnownEvent, ModelTimeline } from "../lib/types";

type Props = {
  model: ModelTimeline;
  seizures: KnownEvent[];
  view: Range;
  onZoom: (ev: { start: number; end: number }) => void;
};

// إحداثيات داخلية (viewBox) — يتمدد الرسم مع عرض الصفحة
const W = 1000;
const LEFT = 8;
const RIGHT = 8;
const ROW_H = 18;
const DET_Y = 6;            // صف "ما اكتشفه النموذج"
const TRUE_Y = DET_Y + 26;  // صف "ما حدده الأطباء"
const PLOT_Y = TRUE_Y + 34; // منحنى الاحتمال
const PLOT_H = 150;
const AXIS_Y = PLOT_Y + PLOT_H + 16;
const H = AXIS_Y + 8;

/**
 * الخط الزمني لتسجيل كامل:
 *  - أعلاه: الأحداث التي أنذر بها النموذج (أحمر) والنوبات التي حددها الأطباء (أخضر)
 *  - أسفله: احتمال "نوبة" لكل نافذة، مع خط العتبة
 * النقر على حدث يكبّر حوله.
 */
export function SeizureTimeline({ model, seizures, view, onZoom }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const span = view.to - view.from;
  const x = (t: number) => LEFT + ((t - view.from) / span) * (W - LEFT - RIGHT);
  const y = (p: number) => PLOT_Y + PLOT_H - p * PLOT_H;
  const tOf = (px: number) => view.from + ((px - LEFT) / (W - LEFT - RIGHT)) * span;

  const [a, b] = visibleIndices(model.times, view);
  let path = "";
  for (let i = a; i < b; i++) path += `${i === a ? "M" : "L"}${x(model.times[i]).toFixed(1)},${y(model.probs[i]).toFixed(1)}`;
  const area = path ? `${path}L${x(model.times[b - 1]).toFixed(1)},${y(0)}L${x(model.times[a]).toFixed(1)},${y(0)}Z` : "";

  const bar = (ev: { start: number; end: number }, top: number, cls: string, key: string, title: string) => {
    if (ev.end < view.from || ev.start > view.to) return null;
    const x1 = Math.max(LEFT, x(ev.start));
    const x2 = Math.min(W - RIGHT, x(ev.end));
    return (
      <rect key={key} className={cls} x={x1} y={top} width={Math.max(3, x2 - x1)} height={ROW_H} rx={3}
        onClick={() => onZoom(ev)}>
        <title>{title}</title>
      </rect>
    );
  };

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    setHover(px < LEFT || px > W - RIGHT ? null : nearestIndex(model.times, tOf(px)));
  };

  return (
    <div className="timeline" dir="ltr">
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
        aria-label="الخط الزمني: احتمال النوبة والأحداث المكتشفة والحقيقية">
        {timeTicks(view).map((t) => (
          <g key={t}>
            <line className="tl-grid" x1={x(t)} x2={x(t)} y1={DET_Y} y2={PLOT_Y + PLOT_H} />
            <text className="tl-tick" x={x(t)} y={AXIS_Y}
              textAnchor={x(t) < LEFT + 24 ? "start" : x(t) > W - RIGHT - 24 ? "end" : "middle"}>{formatClock(t)}</text>
          </g>
        ))}
        <rect className="tl-row" x={LEFT} y={DET_Y} width={W - LEFT - RIGHT} height={ROW_H} />
        <rect className="tl-row" x={LEFT} y={TRUE_Y} width={W - LEFT - RIGHT} height={ROW_H} />
        {model.detected.map((ev: DetectedEvent, i) =>
          bar(ev, DET_Y, "tl-detected", `d${i}`, `إنذار النموذج: ${formatClock(ev.alarm ?? ev.start)} — اضغط للتكبير`))}
        {seizures.map((ev, i) =>
          bar(ev, TRUE_Y, "tl-true", `s${i}`, `نوبة حددها الأطباء: ${formatClock(ev.start)}–${formatClock(ev.end)} — اضغط للتكبير`))}

        <rect className="tl-plot" x={LEFT} y={PLOT_Y} width={W - LEFT - RIGHT} height={PLOT_H} />
        {seizures.map((ev, i) => ev.end >= view.from && ev.start <= view.to && (
          <rect key={`band${i}`} className="tl-true-band" x={Math.max(LEFT, x(ev.start))} y={PLOT_Y}
            width={Math.max(2, Math.min(W - RIGHT, x(ev.end)) - Math.max(LEFT, x(ev.start)))} height={PLOT_H} />
        ))}
        {area && <path className="tl-area" d={area} />}
        {path && <path className="tl-line" d={path} />}
        <line className="tl-threshold" x1={LEFT} x2={W - RIGHT} y1={y(model.threshold)} y2={y(model.threshold)} />

        {hover !== null && hover < model.times.length && (
          <g>
            <line className="tl-cursor" x1={x(model.times[hover])} x2={x(model.times[hover])} y1={DET_Y} y2={PLOT_Y + PLOT_H} />
            <circle className="tl-dot" cx={x(model.times[hover])} cy={y(model.probs[hover])} r={4} />
          </g>
        )}
      </svg>
      <div className="tl-legend" dir="rtl">
        <span><i className="sw sw-detected" /> إنذار النموذج</span>
        <span><i className="sw sw-true" /> نوبة حددها الأطباء</span>
        <span><i className="sw sw-line" /> احتمال النوبة لكل نافذة 4 ثوانٍ</span>
        <span><i className="sw sw-threshold" /> العتبة ({Math.round(model.threshold * 100)}%)</span>
        <span className="spacer" />
        {hover !== null && hover < model.times.length && (
          <span className="num">{formatClock(model.times[hover])} · {Math.round(model.probs[hover] * 100)}%</span>
        )}
      </div>
    </div>
  );
}
