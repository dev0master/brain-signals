import type { CalibrationLive, Connection, ReadinessStatus, SpellerLive } from "./types";

export const READINESS: Record<ReadinessStatus, { text: string; tone: "ok" | "warn" | "bad" | "info" }> = {
  ready: { text: "جاهز", tone: "ok" },
  needs_prepare: { text: "يحتاج تجهيز النماذج", tone: "warn" },
  needs_calibration: { text: "يحتاج معايرتك", tone: "info" },
  unsupported: { text: "غير مدعوم بهذه الأقطاب", tone: "bad" },
};

export const CONNECTION_HINT: Record<Connection, string> = {
  none: "لا يحتاج توصيلاً — يولّد إشارة تجريبية لتختبر كل الصفحات قبل شراء جهاز.",
  serial: "وصّل الـ dongle بمنفذ USB، ثم افتح «إدارة الأجهزة» في ويندوز ← Ports لتعرف رقم المنفذ (مثل COM3).",
  bluetooth: "شغّل البلوتوث في الحاسوب والجهاز. اترك العنوان فارغاً ليبحث عنه BrainFlow تلقائياً.",
};

/** مشكلات أسماء الأقطاب قبل الحفظ: فارغ أو مكرر (القديم T3 = الحديث T7). */
export function nameProblems(names: string[]): string[] {
  const alias: Record<string, string> = { T3: "T7", T4: "T8", T5: "P7", T6: "P8" };
  const norm = (n: string) => {
    const u = n.trim().toUpperCase();
    return alias[u] ?? u;
  };
  const problems: string[] = [];
  if (names.some((n) => !n.trim())) problems.push("بعض القنوات بلا اسم قطب");
  const seen = new Map<string, string>();
  for (const n of names.filter((x) => x.trim())) {
    const k = norm(n);
    if (seen.has(k)) problems.push(`«${n}» مكرر مع «${seen.get(k)}»`);
    else seen.set(k, n);
  }
  return problems;
}

/** ما يُكتب بخط كبير أثناء المعايرة. */
export function calibrationPrompt(c: CalibrationLive): { title: string; hint: string; tone: string } {
  switch (c.phase) {
    case "lead_in":
      return { title: "اجلس مرتاحاً", hint: "أرخِ كتفيك ويديك، وانظر إلى الشاشة. نبدأ بعد لحظات.", tone: "info" };
    case "prepare":
      return { title: "استعد", hint: "ثبّت نظرك هنا ولا تتحرك.", tone: "info" };
    case "imagine":
      return { title: `تخيّل: ${c.imagine}`, hint: imagineHint(c.class ?? ""), tone: "go" };
    case "rest":
      return { title: "استرح", hint: "ارمش وابلع ريقك الآن إن احتجت — لا أثناء التخيّل.", tone: "rest" };
    case "training":
      return { title: "جارٍ تدريب نموذجك…", hint: "ثوانٍ قليلة.", tone: "info" };
    case "done":
      return { title: "انتهت المعايرة", hint: "", tone: "ok" };
    case "error":
      return { title: "تعذّرت المعايرة", hint: c.error ?? "", tone: "bad" };
  }
}

export function imagineHint(cls: string): string {
  return {
    left_hand: "تخيّل أنك تقبض يدك اليسرى وتفتحها — الإحساس بالحركة، دون أن تتحرك فعلاً.",
    right_hand: "تخيّل أنك تقبض يدك اليمنى وتفتحها — الإحساس بالحركة، دون أن تتحرك فعلاً.",
    feet: "تخيّل أنك تضغط بقدميك على الأرض — دون أن تتحرك فعلاً.",
    tongue: "تخيّل أنك تحرك لسانك داخل فمك — دون أن تحركه فعلاً.",
  }[cls] ?? "";
}

export function spellerPrompt(s: SpellerLive): { title: string; tone: string } {
  switch (s.phase) {
    case "lead_in":
      return { title: "نبدأ بعد لحظات — اجلس مرتاحاً", tone: "info" };
    case "prepare":
      return { title: "استعد — قرّر أمرك التالي", tone: "info" };
    case "imagine":
      return { title: "تخيّل الآن", tone: "go" };
    case "result":
      return { title: s.last?.executed ? "نُفّذ" : "امتنع النظام — ثقة منخفضة", tone: "rest" };
  }
}

/** نسبة التقدّم في المعايرة (0–1) بعدد المحاولات المكتملة. */
export function calibrationProgress(c: CalibrationLive): number {
  return c.n_trials ? c.collected / c.n_trials : 0;
}
