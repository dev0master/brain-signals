import { describe, expect, it } from "vitest";
import { nearestIndex, timeTicks, visibleIndices, zoomAround } from "./timeline";

describe("timeline", () => {
  it("zoomAround يبقى داخل حدود التسجيل", () => {
    expect(zoomAround({ start: 60, end: 100 }, 3600)).toEqual({ from: 0, to: 220 });
    expect(zoomAround({ start: 3550, end: 3590 }, 3600)).toEqual({ from: 3430, to: 3600 });
  });

  it("timeTicks يختار فاصلاً مناسباً", () => {
    expect(timeTicks({ from: 0, to: 3600 })).toEqual([0, 600, 1200, 1800, 2400, 3000, 3600]);
    expect(timeTicks({ from: 2900, to: 3140 })).toEqual([2910, 2940, 2970, 3000, 3030, 3060, 3090, 3120]);
  });

  it("visibleIndices يشمل نقطة على كل حافة", () => {
    const times = [2, 4, 6, 8, 10, 12];
    expect(visibleIndices(times, { from: 5, to: 9 })).toEqual([1, 5]);
  });

  it("nearestIndex", () => {
    const times = [2, 4, 6, 8];
    expect(nearestIndex(times, 4.9)).toBe(1);
    expect(nearestIndex(times, 5.1)).toBe(2);
    expect(nearestIndex(times, 100)).toBe(3);
  });
});
