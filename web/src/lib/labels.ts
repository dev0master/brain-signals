import type { Quality } from "./types";

/** أسماء نطاقات الترددات ومعناها التقريبي — بلغة بسيطة. */
export const BAND_INFO: Record<string, { name: string; range: string; meaning: string }> = {
  delta: { name: "دلتا", range: "0.5–4Hz", meaning: "النوم العميق" },
  theta: { name: "ثيتا", range: "4–8Hz", meaning: "النعاس والتأمل" },
  alpha: { name: "ألفا", range: "8–13Hz", meaning: "الاسترخاء بعينين مغمضتين" },
  beta: { name: "بيتا", range: "13–30Hz", meaning: "التركيز والتفكير النشط" },
  gamma: { name: "جاما", range: "30–45Hz", meaning: "معالجة معقّدة (تختلط بحركة العضلات)" },
};

export const QUALITY_INFO: Record<Quality, { name: string; tone: "ok" | "warn" | "bad"; hint: string }> = {
  good: { name: "جيدة", tone: "ok", hint: "الإشارة سليمة" },
  flat: { name: "مسطّحة", tone: "bad", hint: "القطب غالباً غير ملامس للجلد" },
  noisy: { name: "مشوّشة", tone: "warn", hint: "حركة أو تلامس ضعيف" },
  saturated: { name: "مشبعة", tone: "bad", hint: "الإشارة تجاوزت حدود الجهاز" },
};

export const SOURCE_KIND: Record<string, string> = {
  synthetic: "إشارة اصطناعية",
  replay: "تسجيل محفوظ يُعاد تشغيله",
  device: "جهاز متصل",
};
