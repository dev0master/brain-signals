import { describe as d, expect, it } from "vitest";
import { describe, segmentAt, smooth, valueAt } from "./mental";

d("mental helpers", () => {
  it("valueAt يعيد آخر قرار صدر حتى اللحظة", () => {
    const times = [70, 72, 74];
    const values = [0.1, 0.5, 0.9];
    expect(valueAt(times, values, 60)).toBeNull();
    expect(valueAt(times, values, 73)).toBe(0.5);
    expect(valueAt(times, values, 100)).toBe(0.9);
  });

  it("smooth", () => {
    expect(smooth([0, 1, 1, 1], 2)).toEqual([0, 0.5, 1, 1]);
  });

  it("segmentAt و describe", () => {
    const segs = [{ kind: "calm" as const, label: "راحة", start: 60, end: 180 }];
    expect(segmentAt(segs, 100)?.label).toBe("راحة");
    expect(segmentAt(segs, 200)).toBeNull();
    expect(segmentAt(segs, 180)?.label).toBe("راحة"); // نهاية آخر مقطع
    expect(describe(0.8, "مرتاح", "مركّز")).toBe("مركّز");
    expect(describe(0.5, "مرتاح", "مركّز")).toBe("بين الحالتين");
  });
});
