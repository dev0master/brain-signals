import { defineConfig } from "@playwright/test";

// اختبار من البداية للنهاية: الخادم الحقيقي + الواجهة المبنية (npm run build أولاً) في متصفح Edge
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8770",
    channel: "msedge",
    viewport: { width: 1440, height: 900 },
    locale: "ar",
  },
  webServer: {
    command: "C:\\Users\\MSI\\neurolens-venv\\Scripts\\python.exe -m uvicorn server.main:app --port 8770",
    cwd: "..", // مجلد المشروع
    url: "http://127.0.0.1:8770/api/health",
    reuseExistingServer: true,
    timeout: 60_000,
    env: { PYTHONUTF8: "1" },
  },
});
