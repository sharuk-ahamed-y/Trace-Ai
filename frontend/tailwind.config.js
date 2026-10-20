/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: '#030508',
          900: '#05070B',
          850: '#080C14',
          800: '#0C101B',
          750: '#101624',
          700: '#141C2E',
          600: '#1C2740',
        },
        gold: {
          primary: '#D4AF37',
          radiant: '#E5C158',
          light: '#F5E5B8',
          dark: '#A88424',
          muted: '#997D29',
          border: 'rgba(212, 175, 55, 0.25)',
          glow: 'rgba(212, 175, 55, 0.35)',
        },
        cyan: {
          electric: '#00F2FE',
          neon: '#00C2FF',
          dark: '#0084B4',
          glow: 'rgba(0, 242, 254, 0.35)',
        },
        // Backward-compatible color aliases
        pulse: {
          bg: '#05070B',
          secondary: '#080C14',
          card: '#0C101B',
          elevated: '#101624',
          border: 'rgba(212, 175, 55, 0.2)',
          borderLight: 'rgba(212, 175, 55, 0.35)',
          neon: '#D4AF37',
          lime: '#E5C158',
          cyan: '#00F2FE',
          blue: '#00C2FF',
          red: '#EF4444',
          text: '#F8FAFC',
          muted: '#94A3B8',
          subtle: '#64748B'
        },
        cyber: {
          dark: '#05070B',
          card: '#0C101B',
          border: 'rgba(212, 175, 55, 0.2)',
          neon: '#D4AF37',
          blue: '#00C2FF',
          lime: '#E5C158',
          purple: '#A855F7',
          pink: '#EC4899',
          muted: '#94A3B8'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'Outfit', 'Segoe UI', 'sans-serif'],
        display: ['Outfit', 'Inter', 'sans-serif']
      },
      boxShadow: {
        'gold-glow': '0 0 25px rgba(212, 175, 55, 0.25)',
        'gold-glow-lg': '0 0 40px rgba(212, 175, 55, 0.4)',
        'cyan-glow': '0 0 25px rgba(0, 242, 254, 0.25)',
        'cyan-glow-lg': '0 0 40px rgba(0, 194, 255, 0.4)',
        'card-dark': '0 10px 30px rgba(0, 0, 0, 0.6)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
      },
      backgroundImage: {
        'gold-gradient': 'linear-gradient(135deg, #F5E5B8 0%, #E5C158 40%, #D4AF37 70%, #A88424 100%)',
        'gold-subtle': 'linear-gradient(135deg, rgba(229, 193, 88, 0.15) 0%, rgba(212, 175, 55, 0.05) 100%)',
        'cyan-gradient': 'linear-gradient(135deg, #00F2FE 0%, #00C2FF 100%)',
        'dark-glass': 'linear-gradient(180deg, rgba(16, 22, 36, 0.7) 0%, rgba(8, 12, 20, 0.85) 100%)'
      }
    },
  },
  plugins: [],
}
