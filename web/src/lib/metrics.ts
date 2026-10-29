import type { EventMetrics } from "./types";

/** حدود عملية تقريبية: كاشف يفوّت أغلب النوبات أو يُنذر كل ساعة سيتجاهله الناس. */
export function unusable(m: EventMetrics): boolean {
  return (m.sensitivity ?? 0) < 0.5 || (m.false_alarms_per_hour ?? 0) > 1;
}
