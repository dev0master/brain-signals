import type { KnownEvent } from "./types";

/**
 * ذاكرة دائرية لعيّنات العرض: لكل قناة آخر N عيّنة.
 * لا تمر عبر React — الرسم على canvas يقرأ منها مباشرة، فيبقى سريعاً مهما كثرت القنوات.
 */
export class SignalBuffer {
  readonly length: number;
  readonly data: Float32Array[];
  private write = 0; // موضع الكتابة التالي
  total = 0; // كم عيّنة دخلت منذ البداية

  constructor(nChannels: number, length: number) {
    this.length = length;
    this.data = Array.from({ length: nChannels }, () => new Float32Array(length));
  }

  push(samples: number[][]): void {
    const n = samples[0]?.length ?? 0;
    for (let i = 0; i < n; i++) {
      const pos = (this.write + i) % this.length;
      for (let ch = 0; ch < this.data.length; ch++) this.data[ch][pos] = samples[ch][i];
    }
    this.write = (this.write + n) % this.length;
    this.total += n;
  }

  /** عيّنات القناة من الأقدم إلى الأحدث. */
  ordered(ch: number, out = new Float32Array(this.length)): Float32Array {
    const src = this.data[ch];
    out.set(src.subarray(this.write), 0);
    out.set(src.subarray(0, this.write), this.length - this.write);
    return out;
  }

  /** كم عيّنة صالحة موجودة (قبل امتلاء الذاكرة أول مرة). */
  get filled(): number {
    return Math.min(this.total, this.length);
  }
}

/** متوسط القدرة النسبية لكل نطاق عبر القنوات الجيدة فقط. */
export function meanBands(bands: number[][], quality?: string[]): number[] {
  const rows = bands.filter((_, i) => !quality || quality[i] === "good");
  const use = rows.length ? rows : bands;
  if (!use.length) return [];
  return use[0].map((_, b) => use.reduce((s, r) => s + r[b], 0) / use.length);
}

/** الحدث المعروف (مثل نوبة مسجّلة طبياً) الذي يقع فيه الزمن t، إن وُجد. */
export function eventAt(events: KnownEvent[], t: number): KnownEvent | null {
  return events.find((e) => t >= e.start && t <= e.end) ?? null;
}

/** الأحداث التي تتقاطع مع النافذة الظاهرة [from, to] — لتلوينها على الرسم. */
export function eventsInView(events: KnownEvent[], from: number, to: number): KnownEvent[] {
  return events.filter((e) => e.end >= from && e.start <= to);
}

export function formatClock(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
