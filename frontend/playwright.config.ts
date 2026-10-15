import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', fullyParallel: false, workers: 1, timeout: 30000,
  reporter: [['list'], ['json', { outputFile: '../results/phase3/browser-tests.json' }]],
  use: { baseURL: 'http://127.0.0.1:5173', channel: 'chrome', viewport: { width: 1440, height: 1100 }, reducedMotion: 'reduce', trace: 'retain-on-failure' },
  webServer: [
    { command: '../.venv/bin/python -m uvicorn backend.main:app --app-dir .. --host 127.0.0.1 --port 8000', url: 'http://127.0.0.1:8000/api/ready', reuseExistingServer: !process.env.CI, timeout: 30000 },
    { command: 'npm run dev', url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI, timeout: 30000 },
  ],
});
