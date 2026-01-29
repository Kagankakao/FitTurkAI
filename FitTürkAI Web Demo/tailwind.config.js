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
        'hero-gradient': 'linear-gradient(120deg, #0f172a 0%, #111827 55%, #0b1220 100%)',
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

