import { describe, expect, it } from "vitest";
import { calibrationProgress, calibrationPrompt, nameProblems, spellerPrompt } from "./device";
import type { CalibrationLive, SpellerLive } from "./types";

describe("أسماء الأقطاب", () => {
  it("تقبل أسماء مختلفة", () => {
    expect(nameProblems(["Fp1", "Fp2", "C3", "C4"])).toEqual([]);
  });
  it("ترفض الفارغ والمكرر، والاسم القديم مع الحديث لنفس الموضع", () => {
    expect(nameProblems(["C3", ""])).toEqual(["بعض القنوات بلا اسم قطب"]);
    expect(nameProblems(["c3", "C3"])[0]).toContain("مكرر");
    expect(nameProblems(["T3", "T7"])[0]).toContain("مكرر");
  });
});

describe("المعايرة", () => {
  const base: CalibrationLive = { phase: "imagine", n_trials: 40, collected: 10, per_class: 10, class: "feet", imagine: "القدمان" };
  it("تعرض الأمر المطلوب وتلميحه", () => {
    const p = calibrationPrompt(base);
    expect(p.title).toBe("تخيّل: القدمان");
    expect(p.hint).toContain("قدميك");
    expect(p.tone).toBe("go");
  });
  it("التقدّم بعدد المحاولات المكتملة", () => {
    expect(calibrationProgress(base)).toBe(0.25);
  });
  it("الخطأ يظهر برسالته", () => {
    expect(calibrationPrompt({ ...base, phase: "error", error: "قطب مسطّح" }).hint).toBe("قطب مسطّح");
  });
});

describe("الكتابة المباشرة", () => {
  it("تقول إن النظام امتنع حين لم يُنفّذ أمر", () => {
    const s = { phase: "result", last: { decoded: "left", executed: null, confidence: 0.3, probs: {} } } as unknown as SpellerLive;
    expect(spellerPrompt(s).title).toContain("امتنع");
  });
});
