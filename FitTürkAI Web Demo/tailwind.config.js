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
        'hero-gradient': 'linear-gradient(120deg, #ecfdf3 0%, #fef3c7 45%, #ffe4e6 100%)',
        'produce-glow':
          'radial-gradient(700px circle at 20% 15%, rgba(16, 185, 129, 0.35), transparent 50%), radial-gradient(600px circle at 85% 20%, rgba(251, 146, 60, 0.28), transparent 50%), radial-gradient(600px circle at 80% 80%, rgba(236, 72, 153, 0.22), transparent 55%)',
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

