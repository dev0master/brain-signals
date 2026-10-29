import { formatClock } from "../lib/liveStore";
import { timeTicks } from "../lib/timeline";
import type { MentalSegment } from "../lib/types";

type Props = {
  times: number[];
  values: number[];
  compare?: number[];      // خط مقارنة (بلا معايرة) — متقطع
  segments: MentalSegment[];
  cursor: number;
  onSeek: (t: number) => void;
};

const W = 1000;
const H = 200;
const PAD = 8;
const PLOT_TOP = 26;
const PLOT_H = 140;

/** خط زمني لمقياس الحالة الذهنية فوق خلفية الحالات الحقيقية (معايرة / هدوء / مهمة). */
export function StateTimeline({ times, values, compare, segments, cursor, onSeek }: Props) {
  const end = Math.max(...segments.map((s) => s.end));
  const x = (t: number) => PAD + (t / end) * (W - 2 * PAD);
  const y = (v: number) => PLOT_TOP + PLOT_H - v * PLOT_H;
  const path = (vs: number[]) => vs.map((v, i) => `${i ? "L" : "M"}${x(times[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const seek = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    onSeek(Math.max(0, Math.min(end, ((px - PAD) / (W - 2 * PAD)) * end)));
  };

  return (
    <div className="timeline" dir="ltr">
      <svg viewBox={`0 0 ${W} ${H}`} onClick={seek} role="img" aria-label="المقياس عبر الزمن مع الحالات الحقيقية">
        {segments.map((s, i) => (
          <g key={i}>
            <rect className={`seg-${s.kind}`} x={x(s.start)} y={PLOT_TOP} width={x(s.end) - x(s.start)} height={PLOT_H} />
            <text className="seg-label" x={(x(s.start) + x(s.end)) / 2} y={PLOT_TOP - 8} textAnchor="middle">{s.label}</text>
          </g>
        ))}
        <line className="tl-threshold" x1={PAD} x2={W - PAD} y1={y(0.5)} y2={y(0.5)} />
        {compare && <path className="tl-compare" d={path(compare)} />}
        <path className="tl-line" d={path(values)} />
        <line className="tl-cursor-strong" x1={x(cursor)} x2={x(cursor)} y1={PLOT_TOP} y2={PLOT_TOP + PLOT_H} />
        {timeTicks({ from: 0, to: end }, 8).map((t) => (
          <text key={t} className="tl-tick" x={x(t)} y={H - 10}
            textAnchor={x(t) < PAD + 24 ? "start" : x(t) > W - PAD - 24 ? "end" : "middle"}>{formatClock(t)}</text>
        ))}
      </svg>
    </div>
  );
}
