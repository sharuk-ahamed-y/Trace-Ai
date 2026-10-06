/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          dark: '#050505',
          card: '#0A0D14',
          border: '#1E293B',
          neon: '#00FF88',
          blue: '#00E5FF',
          purple: '#B026FF',
          pink: '#FF007F',
          muted: '#94A3B8'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'Segoe UI', 'sans-serif']
      }
    },
  },
  plugins: [],
}
