import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', fullyParallel: false, workers: 1, timeout: 90000,
  reporter: [['list'], ['json', { outputFile: '../results/arcade-ux/browser-tests.json' }]],
  use: { baseURL: 'http://127.0.0.1:5174', channel: 'chrome', viewport: { width: 1440, height: 1100 }, reducedMotion: 'reduce', trace: 'retain-on-failure' },
  webServer: [
    { command: '../.venv/bin/python -m uvicorn backend.main:app --app-dir .. --host 127.0.0.1 --port 8001', url: 'http://127.0.0.1:8001/api/ready', reuseExistingServer: false, timeout: 30000 },
    { command: 'npm run preview -- --port 5174', env: {FEATUREDIVE_API_ORIGIN:'http://127.0.0.1:8001'}, url: 'http://127.0.0.1:5174', reuseExistingServer: false, timeout: 30000 },
  ],
});
