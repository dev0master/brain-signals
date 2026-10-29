// حسابات الخط الزمني لصفحة الصرع — دوال نقية قابلة للاختبار

export type Range = { from: number; to: number };

/** مدى التكبير حول حدث: دقيقتان قبله ودقيقتان بعده، ضمن حدود التسجيل. */
export function zoomAround(ev: { start: number; end: number }, duration: number, padSec = 120): Range {
  return { from: Math.max(0, ev.start - padSec), to: Math.min(duration, ev.end + padSec) };
}

/** علامات محور الزمن: فاصل "مريح" حسب طول المدى الظاهر. */
export function timeTicks({ from, to }: Range, maxTicks = 8): number[] {
  const steps = [10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const span = to - from;
  const step = steps.find((s) => span / s <= maxTicks) ?? 3600;
  const out: number[] = [];
  for (let t = Math.ceil(from / step) * step; t <= to; t += step) out.push(t);
  return out;
}

/** نقاط المنحنى الواقعة في المدى الظاهر (مع نقطة قبله لئلا ينقطع الخط عند الحافة). */
export function visibleIndices(times: number[], { from, to }: Range): [number, number] {
  let a = 0;
  while (a < times.length && times[a] < from) a++;
  let b = a;
  while (b < times.length && times[b] <= to) b++;
  return [Math.max(0, a - 1), Math.min(times.length, b + 1)];
}

/** أقرب نقطة لزمن معيّن (لعرض قيمة المنحنى تحت مؤشر الفأرة). */
export function nearestIndex(times: number[], t: number): number {
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) return lo - 1;
  return lo;
}
