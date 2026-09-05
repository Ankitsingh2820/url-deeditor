import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The backend owns /api (JSON), /media (artifacts, with Range support for
// <video> seeking) and the health probes. Proxying keeps the frontend
// same-origin in dev so EventSource and video seeking behave exactly as they
// will behind one reverse proxy in production.
const target = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Bind every interface: the default `localhost` resolves to ::1 only on
    // Windows, so 127.0.0.1 (and any other device on the LAN, handy when
    // recording a demo) would otherwise refuse the connection.
    host: true,
    proxy: {
      '/api': { target, changeOrigin: true },
      '/media': { target, changeOrigin: true },
      '/healthz': { target, changeOrigin: true },
      '/readyz': { target, changeOrigin: true },
      // the header links straight to the live OpenAPI docs
      '/docs': { target, changeOrigin: true },
      '/openapi.json': { target, changeOrigin: true },
    },
  },
})
