import type { MentalSegment } from "./types";

/** آخر قيمة متاحة حتى الزمن t (القرار يصدر عند نهاية كل نافذة). */
export function valueAt(times: number[], values: number[], t: number): number | null {
  let lo = 0;
  let hi = times.length - 1;
  if (hi < 0 || t < times[0]) return null;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return values[lo];
}

/** متوسط متحرك لآخر k قيم — يهدّئ اهتزاز المؤشر بين نافذة وأخرى. */
export function smooth(values: number[], k: number): number[] {
  return values.map((_, i) => {
    const from = Math.max(0, i - k + 1);
    const part = values.slice(from, i + 1);
    return part.reduce((a, b) => a + b, 0) / part.length;
  });
}

export function segmentAt(segments: MentalSegment[], t: number): MentalSegment | null {
  // اللحظة الأخيرة من التسجيل تنتمي للمقطع الأخير (وإلا ظهرت "—" عند نهاية التشغيل)
  return segments.find((s, i) => t >= s.start && (t < s.end || (i === segments.length - 1 && t <= s.end))) ?? null;
}

/** وصف لفظي للقيمة على المقياس. */
export function describe(value: number, low: string, high: string): string {
  if (value < 0.35) return low;
  if (value > 0.65) return high;
  return "بين الحالتين";
}
