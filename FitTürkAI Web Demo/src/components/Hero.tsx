import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';

export default function Hero() {
  const router = useRouter();
  return (
    <section className="relative w-full min-h-[60vh] flex flex-col items-center justify-center bg-hero-gradient text-white py-16 overflow-hidden">
      <div className="absolute inset-0 bg-produce-glow opacity-70" aria-hidden="true" />
      <div className="absolute -top-6 left-8 hidden md:flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-2 text-xs text-white/80 backdrop-blur">
        🥦 Brokoli
      </div>
      <div className="absolute top-20 right-10 hidden md:flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-2 text-xs text-white/80 backdrop-blur">
        🥕 Havuç
      </div>
      <div className="absolute bottom-10 left-12 hidden md:flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-2 text-xs text-white/80 backdrop-blur">
        🍅 Domates
      </div>
      <div className="relative z-10 flex flex-col items-center">
        <motion.h1
          className="text-4xl md:text-6xl font-semibold text-center tracking-tight"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          Hayalindeki Vücuda Ulaş!
        </motion.h1>
        <motion.p
          className="mt-6 text-lg md:text-2xl text-center max-w-2xl mx-auto text-white/80"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 1 }}
        >
          Yapay Zeka Destekli Kişisel Sağlık ve Fitness Asistanı
        </motion.p>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-3 text-xs font-medium text-white/85">
          <span className="rounded-full border border-emerald-200/30 bg-emerald-400/20 px-3 py-1">🥬 Ispanak</span>
          <span className="rounded-full border border-orange-200/30 bg-orange-400/20 px-3 py-1">🥕 Havuç</span>
          <span className="rounded-full border border-rose-200/30 bg-rose-400/20 px-3 py-1">🍅 Domates</span>
          <span className="rounded-full border border-lime-200/30 bg-lime-400/20 px-3 py-1">🥒 Salatalık</span>
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
            className="px-8 py-3 rounded-full bg-white text-slate-900 font-semibold text-lg shadow-lg transition-all duration-300 hover:bg-slate-100"
            onClick={() => router.push('/chat')}
          >
            Demoyu Başlat
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.95 }}
            className="px-8 py-3 rounded-full border border-white/40 text-white font-semibold text-lg shadow-lg transition-all duration-300 hover:bg-white/10"
            onClick={() => router.push('/chat')}
          >
            Asistanı Dene
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
}
