import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    viewport: { width: 1440, height: 1000 },
    channel: process.env.USAMI_BROWSER_CHANNEL || 'chrome',
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      command: '../../.venv/bin/python -m uvicorn web.backend.app:app --app-dir ../.. --host 127.0.0.1 --port 8001',
      url: 'http://127.0.0.1:8001/api/health',
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'USAMI_BACKEND_URL=http://127.0.0.1:8001 npm run dev',
      url: 'http://127.0.0.1:5173/api/health',
      reuseExistingServer: !process.env.CI,
    },
  ],
});
