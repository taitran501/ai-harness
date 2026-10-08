import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: loadEnv(mode, '.', '').HARNESS_DASHBOARD_API || 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
}));
