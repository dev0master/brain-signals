import { expect, it } from "vitest";
import { unusable } from "./metrics";

const base = { n_seizures: 19, detected: 19, false_alarms: 3, hours: 36.4, mean_latency_sec: 8 };

it("الكاشف الجيد صالح، والضعيف أو كثير الإنذارات غير صالح", () => {
  expect(unusable({ ...base, sensitivity: 1, false_alarms_per_hour: 0.08 })).toBe(false);
  expect(unusable({ ...base, sensitivity: 0.32, false_alarms_per_hour: 0.1 })).toBe(true);
  expect(unusable({ ...base, sensitivity: 1, false_alarms_per_hour: 7.2 })).toBe(true);
});
