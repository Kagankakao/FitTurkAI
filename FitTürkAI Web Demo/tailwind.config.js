/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#36656B',
          green: '#75B06F',
          soft: '#DAD887',
          light: '#F0F8A4',
        },
        emerald: {
          900: '#36656B',
          600: '#75B06F',
          500: '#75B06F',
          400: '#DAD887',
          300: '#F0F8A4',
        },
        'neutral-light': '#f8fafc',
        'neutral-dark': '#0f172a',
        'primary': '#1f2937',
        'secondary': '#64748b',
        'fitness-green': '#059669',
        'fitness-blue': '#2563eb',
        'fitness-orange': '#f59e0b',
        'fitness-pink': '#db2777',
        'fitness-dark': '#0f172a',
      },
      backgroundImage: {
        'hero-gradient': 'linear-gradient(120deg, #F0F8A4 0%, #DAD887 45%, #75B06F 100%)',
        'produce-glow':
          'radial-gradient(700px circle at 20% 15%, rgba(117, 176, 111, 0.35), transparent 50%), radial-gradient(600px circle at 85% 20%, rgba(218, 216, 135, 0.35), transparent 50%), radial-gradient(600px circle at 80% 80%, rgba(240, 248, 164, 0.45), transparent 55%)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        'pulse-heart': {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.2)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 1s ease-in',
        'pulse-heart': 'pulse-heart 1s infinite',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}

