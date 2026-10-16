/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pulse: {
          bg: '#070909',
          secondary: '#0D1210',
          card: '#111916',
          elevated: '#171E1A',
          border: '#1A2822',
          borderLight: '#23382F',
          neon: '#39FF88',
          lime: '#D7FF32',
          cyan: '#35D9FF',
          blue: '#2B82FE',
          red: '#FF4D4D',
          text: '#F4F7F5',
          muted: '#9AA8A0',
          subtle: '#5F7067'
        },
        cyber: {
          dark: '#070909',
          card: '#111916',
          border: '#1A2822',
          neon: '#39FF88',
          blue: '#35D9FF',
          lime: '#D7FF32',
          purple: '#B026FF',
          pink: '#FF007F',
          muted: '#9AA8A0'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'Outfit', 'Segoe UI', 'sans-serif']
      },
      boxShadow: {
        'neon-glow': '0 0 25px rgba(57, 255, 136, 0.25)',
        'lime-glow': '0 0 25px rgba(215, 255, 50, 0.25)',
        'cyan-glow': '0 0 25px rgba(53, 217, 255, 0.25)',
        'card-dark': '0 10px 30px rgba(0, 0, 0, 0.5)'
      }
    },
  },
  plugins: [],
}
