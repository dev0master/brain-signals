import { BAND_INFO } from "../lib/labels";

/** أعمدة القدرة النسبية لكل نطاق تردد (متوسط القنوات الجيدة). */
export function BandBars({ names, values }: { names: string[]; values: number[] }) {
  if (!values.length) return <p className="muted small">ننتظر ثانيتين من الإشارة لحساب النطاقات…</p>;
  const max = Math.max(...values, 0.01);
  return (
    <div className="bands">
      {names.map((band, i) => {
        const info = BAND_INFO[band] ?? { name: band, range: "", meaning: "" };
        return (
          <div key={band} className="band-row">
            <div className="band-label">
              <strong>{info.name}</strong> <span className="faint small num">{info.range}</span>
              <div className="faint small">{info.meaning}</div>
            </div>
            <div className="band-track">
              <div className={`band-fill band-${band}`} style={{ width: `${(values[i] / max) * 100}%` }} />
            </div>
            <span className="num band-pct">{Math.round(values[i] * 100)}%</span>
          </div>
        );
      })}
    </div>
  );
}
