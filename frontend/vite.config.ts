import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode}) => {
  const apiOrigin = loadEnv(mode,'.','FEATUREDIVE_').FEATUREDIVE_API_ORIGIN ?? 'http://127.0.0.1:8000';
  return {
    plugins: [react()],
    server: { port: 5173, strictPort: true, proxy: { '/api': apiOrigin } },
    preview: { port: 4173, strictPort: true, proxy: { '/api': apiOrigin } },
  };
});
