import type { CommandMeta, SimStep, SpellerCommand, SpellerInfo } from "./types";

/** كم أمراً نفّذه النظام خطأً حتى الخطوة i (شاملة). */
export function errorsUpTo(steps: SimStep[], i: number): number {
  let n = 0;
  for (let k = 0; k <= i && k < steps.length; k++) {
    if (steps[k].executed !== null && steps[k].executed !== steps[k].intended) n++;
  }
  return n;
}

/** الوقت الحقيقي الذي كانت ستستغرقه الخطوات حتى الآن (لا سرعة العرض). */
export function simulatedSeconds(stepsDone: number, secondsPerCommand: number): number {
  return stepsDone * secondsPerCommand;
}

/** مفاتيح لوحة المفاتيح في وضع «جرّب بنفسك». */
export function commandForKey(key: string): SpellerCommand | null {
  switch (key) {
    case "ArrowLeft": return "left";
    case "ArrowRight": return "right";
    case "Enter": case " ": return "select";
    case "Backspace": case "Escape": return "back";
    default: return null;
  }
}

/** نص قصير للوسيط المحفوظ في التقرير لكل شخص. */
export function speedLabel(cpm: number | null | undefined, completion: number | undefined): string {
  if (cpm == null) return "لا يكمل الجملة";
  const rate = completion !== undefined && completion < 1 ? ` (يكمل ${Math.round(completion * 100)}%)` : "";
  return `${cpm.toFixed(1)} حرف/دقيقة${rate}`;
}

/** الأوامر الأربعة بتوزيع شخص معيّن (أي تخيّل يعني أي أمر)، أو الافتراضي إن لم يوجد. */
export function commandsFor(info: SpellerInfo, mapping?: Record<SpellerCommand, string> | null): CommandMeta[] {
  if (!mapping) return info.commands;
  return info.commands.map((c) => ({ ...c, class: mapping[c.command], imagine: info.class_ar[mapping[c.command]] }));
}

/** هل توزيع الشخص هو نفسه التوزيع الافتراضي؟ */
export function isDefaultMapping(info: SpellerInfo, mapping?: Record<SpellerCommand, string> | null): boolean {
  return !mapping || info.commands.every((c) => mapping[c.command] === c.class);
}
