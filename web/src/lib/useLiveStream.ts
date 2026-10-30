import { useEffect, useRef, useState } from "react";
import { SignalBuffer } from "./liveStore";
import type { Frame, Hello, ServerError } from "./types";

const VIEW_SEC = 10; // كم ثانية تظهر على الرسم

export type StreamStatus = "connecting" | "open" | "closed" | "error";

export type LiveState = {
  status: StreamStatus;
  hello: Hello | null;
  error: string | null;
  /** آخر إطار (للنطاقات والجودة ونتائج المحلّلات) — يُحدَّث في React بمعدل منخفض */
  last: Frame | null;
};

/**
 * يفتح اتصال WebSocket بالخادم ويملأ ذاكرة الإشارة.
 * تغيير المصدر أو السرعة = إغلاق الاتصال وفتح اتصال جديد.
 * الإشارة نفسها تُكتب في bufferRef (خارج React) والرسم يقرأها مباشرة.
 *
 * extra: معاملات إضافية للخادم (مثل task=mi_calibration لجلسة معايرة، أو record=1 لحفظ الجلسة).
 * enabled=false: لا اتصال (مثلاً قبل أن يضغط المستخدم «ابدأ المعايرة»).
 */
export function useLiveStream(source: string, speed: number, startSec = 0,
  extra: Record<string, string | number> = {}, enabled = true) {
  const [state, setState] = useState<LiveState>({ status: "connecting", hello: null, error: null, last: null });
  const bufferRef = useRef<SignalBuffer | null>(null);
  const timeRef = useRef(0); // زمن آخر عيّنة في التسجيل (ثانية)
  const query = new URLSearchParams({ source, speed: String(speed), start: String(startSec),
    ...Object.fromEntries(Object.entries(extra).map(([k, v]) => [k, String(v)])) }).toString();

  useEffect(() => {
    if (!enabled) {
      setState({ status: "closed", hello: null, error: null, last: null });
      return;
    }
    let closedByUs = false;
    let lastUiUpdate = 0;
    let lastPhase = "";
    setState({ status: "connecting", hello: null, error: null, last: null });
    bufferRef.current = null;

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws?${query}`);

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as Hello | Frame | ServerError;
      if (msg.type === "hello") {
        bufferRef.current = new SignalBuffer(msg.ch_names.length, Math.round(msg.display_fs * VIEW_SEC));
        setState({ status: "open", hello: msg, error: null, last: null });
      } else if (msg.type === "frame") {
        bufferRef.current?.push(msg.samples);
        timeRef.current = msg.t;
        // الإشارة تُرسم بكل إطار، أما أجزاء React فتكفيها 4 تحديثات في الثانية —
        // إلا عند تغيّر مرحلة المعايرة أو الكتابة: «تخيّل الآن» يجب أن يظهر فوراً
        const now = performance.now();
        const phase = phaseOf(msg);
        if (now - lastUiUpdate > 250 || phase !== lastPhase) {
          lastUiUpdate = now;
          lastPhase = phase;
          setState((s) => ({ ...s, last: msg }));
        }
      } else if (msg.type === "error") {
        setState((s) => ({ ...s, status: "error", error: msg.message }));
      }
    };
    ws.onerror = () => setState((s) => ({ ...s, status: "error", error: s.error ?? "تعذّر الاتصال بالخادم — هل هو يعمل؟" }));
    ws.onclose = () => {
      if (!closedByUs) setState((s) => (s.status === "error" ? s : { ...s, status: "closed" }));
    };

    return () => {
      closedByUs = true;
      ws.close();
    };
  }, [query, enabled]);

  return { ...state, bufferRef, timeRef, viewSec: VIEW_SEC };
}

function phaseOf(frame: Frame): string {
  const task = (frame.calibration ?? frame.speller) as { phase?: string; trial?: number; count?: number } | undefined;
  return task ? `${task.phase}:${task.trial ?? task.count ?? ""}` : "";
}
