import { defineConfig } from 'vite'
import solid from 'vite-plugin-solid'

export default defineConfig({
  plugins: [solid()],
  server: {
    port: 5173,
    proxy: {
      '/pages': 'http://localhost:8787',
      '/image': 'http://localhost:8787',
      '/analyze': 'http://localhost:8787',
      '/health': 'http://localhost:8787',
      '/folder': 'http://localhost:8787',
      '/status': 'http://localhost:8787',
    }
  }
})
