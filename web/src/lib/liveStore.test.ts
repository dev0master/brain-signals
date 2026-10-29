import { describe, expect, it } from "vitest";
import { SignalBuffer, eventAt, eventsInView, formatClock, meanBands } from "./liveStore";

describe("SignalBuffer", () => {
  it("يحفظ آخر N عيّنة بالترتيب من الأقدم للأحدث حتى بعد الالتفاف", () => {
    const b = new SignalBuffer(2, 5);
    b.push([[1, 2, 3], [10, 20, 30]]);
    expect(b.filled).toBe(3);
    b.push([[4, 5, 6, 7], [40, 50, 60, 70]]);
    expect(b.filled).toBe(5);
    expect(Array.from(b.ordered(0))).toEqual([3, 4, 5, 6, 7]);
    expect(Array.from(b.ordered(1))).toEqual([30, 40, 50, 60, 70]);
  });

  it("يتجاهل الإطار الفارغ", () => {
    const b = new SignalBuffer(1, 3);
    b.push([[]]);
    expect(b.total).toBe(0);
  });
});

describe("meanBands", () => {
  it("يتجاهل القنوات غير الجيدة", () => {
    const bands = [[0.5, 0.5], [1, 0]];
    expect(meanBands(bands, ["good", "flat"])).toEqual([0.5, 0.5]);
    expect(meanBands(bands)).toEqual([0.75, 0.25]);
  });

  it("يعود لكل القنوات إن لم تكن أي قناة جيدة", () => {
    expect(meanBands([[1, 0]], ["noisy"])).toEqual([1, 0]);
  });
});

describe("events", () => {
  const events = [{ start: 100, end: 160 }];
  it("يعرف إن كان الزمن داخل حدث", () => {
    expect(eventAt(events, 120)).not.toBeNull();
    expect(eventAt(events, 200)).toBeNull();
  });
  it("يجد الأحداث المتقاطعة مع النافذة الظاهرة", () => {
    expect(eventsInView(events, 150, 170)).toHaveLength(1);
    expect(eventsInView(events, 170, 180)).toHaveLength(0);
  });
});

it("formatClock", () => {
  expect(formatClock(65)).toBe("01:05");
  expect(formatClock(3725)).toBe("1:02:05");
});
