/** مقياس نصف دائري بإبرة: 0 = الطرف الأول (يميناً، بداية القراءة العربية)، 1 = الطرف الثاني (يساراً). */
export function Gauge({ value, low, high, caption }: { value: number | null; low: string; high: string; caption?: string }) {
  const R = 90;
  const cx = 110;
  const cy = 105;
  // الزاوية من 180° إلى 0° في الرسم قبل العكس — بعد العكس تبدأ القيمة 0 من اليمين
  const angle = Math.PI * (1 - (value ?? 0.5));
  const nx = cx + (R - 14) * Math.cos(angle);
  const ny = cy - (R - 14) * Math.sin(angle);
  const arc = (from: number, to: number) => {
    const a0 = Math.PI * (1 - from);
    const a1 = Math.PI * (1 - to);
    return `M ${cx + R * Math.cos(a0)} ${cy - R * Math.sin(a0)} A ${R} ${R} 0 0 1 ${cx + R * Math.cos(a1)} ${cy - R * Math.sin(a1)}`;
  };
  return (
    <div className="gauge">
      <svg viewBox="0 0 220 130" role="img" aria-label={`المقياس بين ${low} و${high}`}>
        {/* نرسم من اليسار لليمين ثم نعكس أفقياً ليبدأ المقياس من اليمين */}
        <g transform="translate(220 0) scale(-1 1)">
        <path d={arc(0, 0.35)} className="g-low" />
        <path d={arc(0.35, 0.65)} className="g-mid" />
        <path d={arc(0.65, 1)} className="g-high" />
        {value !== null && (
          <>
            <line x1={cx} y1={cy} x2={nx} y2={ny} className="g-needle" />
            <circle cx={cx} cy={cy} r={6} className="g-hub" />
          </>
        )}
        </g>
      </svg>
      <div className="gauge-labels">
        <span>{low}</span>
        <span>{high}</span>
      </div>
      {caption && <div className="gauge-caption">{caption}</div>}
    </div>
  );
}
