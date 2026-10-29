import { QUALITY_INFO } from "../lib/labels";
import type { Quality } from "../lib/types";

/** حالة كل قطب: جيد / مسطّح / مشوّش / مشبع. */
export function QualityChips({ names, quality }: { names: string[]; quality?: Quality[] }) {
  if (!quality) return <p className="muted small">ننتظر الإشارة…</p>;
  const bad = quality.filter((q) => q !== "good").length;
  return (
    <>
      <p className="small muted">
        {bad === 0 ? "كل الأقطاب سليمة." : `${bad} من ${quality.length} أقطاب تحتاج انتباهاً — نتائجها أقل موثوقية.`}
      </p>
      <div className="chips">
        {names.map((n, i) => {
          const q = QUALITY_INFO[quality[i]];
          return (
            <span key={n} className={`chip chip-${q.tone}`} title={q.hint}>
              <span className="ltr">{n}</span> · {q.name}
            </span>
          );
        })}
      </div>
    </>
  );
}
