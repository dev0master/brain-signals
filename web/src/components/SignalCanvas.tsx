import { useEffect, useRef, type MutableRefObject } from "react";
import { eventsInView, formatClock, type SignalBuffer } from "../lib/liveStore";
import type { KnownEvent } from "../lib/types";

type Props = {
  bufferRef: MutableRefObject<SignalBuffer | null>;
  timeRef: MutableRefObject<number>;
  chNames: string[];
  viewSec: number;
  scaleUv: number; // كم ميكروفولت يملأ نصف ارتفاع مسار القناة
  events: KnownEvent[];
};

const GUTTER = 56; // مساحة أسماء القنوات
const AXIS_H = 22; // مساحة محور الزمن
const LANE_MIN = 22;
const LANE_DENSE = 13;

/**
 * رسم متعدد القنوات بأسلوب أجهزة EEG: كل قناة في مسار أفقي، والزمن يتقدم لليمين.
 * يُرسم على canvas مباشرة من ذاكرة الإشارة، بلا React، مع كل إطار جديد.
 */
export function SignalCanvas({ bufferRef, timeRef, chNames, viewSec, scaleUv, events }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const propsRef = useRef({ scaleUv, events });
  propsRef.current = { scaleUv, events };
  const forceRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    const css = getComputedStyle(document.documentElement);
    const color = (name: string) => css.getPropertyValue(name).trim();
    const colors = {
      line: color("--signal"),
      grid: color("--line-soft"),
      text: color("--muted"),
      event: color("--bad"),
    };
    let width = 0;
    let height = 0;
    let lastTotal = -1;
    let raf = 0;
    let scratch = new Float32Array(0);

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      width = wrap.clientWidth;
      // التسجيلات كثيرة القنوات (64) تأخذ مسارات أنحف حتى لا يطول الرسم كثيراً
      const lane = chNames.length > 32 ? LANE_DENSE : LANE_MIN;
      height = Math.max(240, chNames.length * lane + AXIS_H);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      lastTotal = -1; // أعد الرسم
    };
    const observer = new ResizeObserver(resize);
    observer.observe(wrap);
    resize();

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const buf = bufferRef.current;
      if (!buf || (buf.total === lastTotal && !forceRef.current)) return;
      lastTotal = buf.total;
      forceRef.current = false;
      const { scaleUv: scale, events: evs } = propsRef.current;
      const t = timeRef.current;
      const plotW = width - GUTTER;
      const lanesH = height - AXIS_H;
      const laneH = lanesH / chNames.length;
      const xOf = (sec: number) => GUTTER + ((sec - (t - viewSec)) / viewSec) * plotW;

      ctx.clearRect(0, 0, width, height);

      // تظليل الأحداث المعروفة (نوبات مسجّلة طبياً)
      for (const e of eventsInView(evs, t - viewSec, t)) {
        const x1 = Math.max(GUTTER, xOf(e.start));
        const x2 = Math.min(width, xOf(e.end));
        ctx.fillStyle = colors.event;
        ctx.globalAlpha = 0.14;
        ctx.fillRect(x1, 0, x2 - x1, lanesH);
        ctx.globalAlpha = 1;
      }

      // خطوط الثواني ومحور الزمن
      ctx.strokeStyle = colors.grid;
      ctx.fillStyle = colors.text;
      ctx.font = "11px Consolas, monospace";
      ctx.lineWidth = 1;
      for (let s = Math.max(0, Math.ceil(t - viewSec)); s <= t; s++) {
        const x = Math.round(xOf(s)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, lanesH);
        ctx.stroke();
        if (s % 2 === 0) ctx.fillText(formatClock(s), x - 14, height - 6);
      }

      // القنوات
      if (scratch.length !== buf.length) scratch = new Float32Array(buf.length);
      const n = buf.length;
      const first = n - buf.filled; // العيّنات قبلها لم تصل بعد
      ctx.lineWidth = 1;
      for (let ch = 0; ch < chNames.length; ch++) {
        const mid = laneH * (ch + 0.5);
        ctx.fillStyle = colors.text;
        ctx.fillText(chNames[ch], 6, mid + 4);
        const data = buf.ordered(ch, scratch);
        ctx.strokeStyle = colors.line;
        ctx.beginPath();
        for (let i = first; i < n; i++) {
          const x = GUTTER + (i / (n - 1)) * plotW;
          const v = Math.max(-1.4, Math.min(1.4, data[i] / scale)); // قصّ لئلا تطغى قناة على جاراتها
          const y = mid - v * (laneH / 2);
          if (i === first) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [bufferRef, timeRef, chNames, viewSec]);

  // أي تغيير في المقياس أو الأحداث يحتاج إعادة رسم حتى لو لم تصل بيانات جديدة
  useEffect(() => {
    forceRef.current = true;
  }, [scaleUv, events]);

  return (
    <div ref={wrapRef} className="signal-wrap" dir="ltr">
      <canvas ref={canvasRef} style={{ width: "100%", display: "block" }} />
    </div>
  );
}
