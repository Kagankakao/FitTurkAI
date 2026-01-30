import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';

export default function Hero() {
  const router = useRouter();
  return (
    <section className="relative w-full min-h-[60vh] flex flex-col items-center justify-center bg-hero-gradient text-brand-dark py-16 overflow-hidden">
      <div className="absolute inset-0 bg-produce-glow opacity-80" aria-hidden="true" />
      <div className="absolute -top-6 left-8 hidden md:flex items-center gap-2 rounded-full border border-brand-soft/60 bg-white/70 px-4 py-2 text-xs text-brand-dark/80 backdrop-blur">
        🥦 Brokoli
      </div>
      <div className="absolute top-20 right-10 hidden md:flex items-center gap-2 rounded-full border border-brand-soft/60 bg-white/70 px-4 py-2 text-xs text-brand-dark/80 backdrop-blur">
        🥕 Havuç
      </div>
      <div className="absolute bottom-10 left-12 hidden md:flex items-center gap-2 rounded-full border border-brand-soft/60 bg-white/70 px-4 py-2 text-xs text-brand-dark/80 backdrop-blur">
        🍅 Domates
      </div>
      <div className="relative z-10 flex flex-col items-center">
        <motion.h1
          className="text-4xl md:text-6xl font-semibold text-center tracking-tight text-brand-dark"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          Hayalindeki Vücuda Ulaş!
        </motion.h1>
        <motion.p
          className="mt-6 text-lg md:text-2xl text-center max-w-2xl mx-auto text-brand-dark/80"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 1 }}
        >
          Yapay Zeka Destekli Kişisel Sağlık ve Fitness Asistanı
        </motion.p>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-3 text-xs font-medium text-brand-dark/90">
          <span className="rounded-full border border-brand-soft/70 bg-brand-light/80 px-3 py-1">🥬 Ispanak</span>
          <span className="rounded-full border border-brand-soft/70 bg-brand-light/80 px-3 py-1">🥕 Havuç</span>
          <span className="rounded-full border border-brand-soft/70 bg-brand-light/80 px-3 py-1">🍅 Domates</span>
          <span className="rounded-full border border-brand-soft/70 bg-brand-light/80 px-3 py-1">🥒 Salatalık</span>
        </div>
        <motion.div
          className="mt-10 flex flex-col md:flex-row gap-4 justify-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 1, duration: 0.7 }}
        >
          <motion.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-3 rounded-full bg-brand-green text-white font-semibold text-lg shadow-lg transition-all duration-300 hover:bg-brand-soft"
            onClick={() => router.push('/chat')}
          >
            Demoyu Başlat
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-3 rounded-full border border-brand-soft text-brand-dark font-semibold text-lg shadow-lg transition-all duration-300 hover:border-brand-green hover:bg-brand-light/70"
            onClick={() => router.push('/chat')}
          >
            Asistanı Dene
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
