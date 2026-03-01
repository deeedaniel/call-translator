import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            if ((err as NodeJS.ErrnoException).code === 'EPIPE' ||
                (err as NodeJS.ErrnoException).code === 'ECONNREFUSED') {
              return;
            }
            console.error('ws proxy error:', err.message);
          });
        },
      },
      '/api': {
        target: 'http://localhost:8000',
      },
    },
  },
})
