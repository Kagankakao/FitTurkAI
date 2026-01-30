import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1f2937',
          light: '#475569',
          dark: '#0f172a',
        },
        secondary: {
          DEFAULT: '#64748b',
          light: '#94a3b8',
          dark: '#334155',
        },
        accent: {
          DEFAULT: '#0f766e',
          light: '#5eead4',
          dark: '#115e59',
        },
        neutral: {
          light: '#f8fafc',
          dark: '#0f172a',
        },
        'fitness-green': '#059669',
        'fitness-blue': '#2563eb',
        'fitness-orange': '#f59e0b',
        'fitness-pink': '#db2777',
        'fitness-dark': '#0f172a',
      },
      backgroundImage: {
        'hero-gradient':
          'linear-gradient(120deg, #ecfdf3 0%, #fef3c7 45%, #ffe4e6 100%)',
        'produce-glow':
          'radial-gradient(700px circle at 20% 15%, rgba(16, 185, 129, 0.35), transparent 50%), radial-gradient(600px circle at 85% 20%, rgba(251, 146, 60, 0.28), transparent 50%), radial-gradient(600px circle at 80% 80%, rgba(236, 72, 153, 0.22), transparent 55%)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        inter: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        heading: ['Poppins', 'Inter', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-in': 'slideIn 0.5s ease-in-out',
        'scale-up': 'scaleUp 0.3s ease-in-out',
        'bounce-in': 'bounceIn 0.5s ease-in-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleUp: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        bounceIn: {
          '0%': { transform: 'scale(0.3)', opacity: '0' },
          '50%': { transform: 'scale(1.05)', opacity: '0.8' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    function({ addUtilities }: any) {
      const newUtilities = {
        '.scrollbar-none': {
          /* Firefox */
          'scrollbar-width': 'none',
          /* Safari and Chrome */
          '&::-webkit-scrollbar': {
            display: 'none',
          },
        },
      }
      addUtilities(newUtilities)
    }
  ],
};

export default config;
