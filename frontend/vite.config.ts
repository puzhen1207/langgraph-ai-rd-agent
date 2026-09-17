import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Split the 1 MB single bundle into cacheable vendor groups.
        // Order matters: the markdown/highlight checks run before the
        // bare `react` match so react-markdown doesn't land in the
        // react chunk.
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (/react-syntax-highlighter|refractor|highlight\.js/.test(id)) {
            return 'highlight'
          }
          if (
            /react-markdown|remark|micromark|mdast|unified|unist|hast|vfile|property-information/.test(
              id,
            )
          ) {
            return 'markdown'
          }
          if (id.includes('lucide-react')) return 'icons'
          if (/[\\/]node_modules[\\/]react(-dom)?[\\/]/.test(id) || id.includes('scheduler')) {
            return 'react'
          }
          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
