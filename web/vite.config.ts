/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// أثناء التطوير: الواجهة على 5173 وكل طلبات /api و/ws تُحوَّل إلى خادم Python على 8770
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8770",
      "/ws": { target: "ws://127.0.0.1:8770", ws: true },
    },
  },
  // اختبارات الوحدات فقط؛ اختبارات المتصفح في e2e/ يشغّلها Playwright (npm run e2e)
  test: { include: ["src/**/*.test.ts"] },
});
