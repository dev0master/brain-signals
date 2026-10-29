import { expect, it } from "vitest";
import { commandForKey, commandsFor, errorsUpTo, simulatedSeconds, speedLabel } from "./speller";
import type { SimStep, SpellerInfo } from "./types";

const step = (intended: string, executed: string | null) =>
  ({ intended, decoded: executed ?? intended, executed, probs: [], trial: 0, view: {} }) as unknown as SimStep;

it("errorsUpTo يعدّ الأوامر الخاطئة المنفّذة فقط", () => {
  const steps = [step("left", "left"), step("left", "back"), step("select", null), step("select", "right")];
  expect(errorsUpTo(steps, 0)).toBe(0);
  expect(errorsUpTo(steps, 2)).toBe(1); // الامتناع (null) ليس خطأً
  expect(errorsUpTo(steps, 3)).toBe(2);
});

it("simulatedSeconds", () => expect(simulatedSeconds(10, 6)).toBe(60));

it("commandForKey", () => {
  expect(commandForKey("ArrowLeft")).toBe("left");
  expect(commandForKey("Enter")).toBe("select");
  expect(commandForKey("Backspace")).toBe("back");
  expect(commandForKey("a")).toBeNull();
});

it("speedLabel", () => {
  expect(speedLabel(2.345, 1)).toBe("2.3 حرف/دقيقة");
  expect(speedLabel(1.5, 0.6)).toBe("1.5 حرف/دقيقة (يكمل 60%)");
  expect(speedLabel(null, 0)).toBe("لا يكمل الجملة");
});

it("commandsFor يطبّق توزيع الشخص على الأوامر", () => {
  const info = {
    commands: [
      { command: "left", class: "left_hand", imagine: "اليد اليسرى", action: "يسار" },
      { command: "select", class: "feet", imagine: "القدمان", action: "اختيار" },
    ],
    class_ar: { left_hand: "اليد اليسرى", right_hand: "اليد اليمنى", feet: "القدمان", tongue: "اللسان" },
  } as unknown as SpellerInfo;
  expect(commandsFor(info, null)).toBe(info.commands);
  const personal = commandsFor(info, { select: "right_hand", left: "tongue", right: "left_hand", back: "feet" });
  expect(personal.find((c) => c.command === "select")?.imagine).toBe("اليد اليمنى");
  expect(personal.find((c) => c.command === "left")?.action).toBe("يسار");
});

it("isDefaultMapping", async () => {
  const { isDefaultMapping } = await import("./speller");
  const info = { commands: [{ command: "select", class: "feet" }, { command: "back", class: "tongue" }] } as unknown as SpellerInfo;
  expect(isDefaultMapping(info, null)).toBe(true);
  expect(isDefaultMapping(info, { select: "feet", back: "tongue", left: "x", right: "y" })).toBe(true);
  expect(isDefaultMapping(info, { select: "tongue", back: "feet", left: "x", right: "y" })).toBe(false);
});
