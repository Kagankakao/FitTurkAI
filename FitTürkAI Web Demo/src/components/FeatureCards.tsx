import { motion } from 'framer-motion';

const features = [
  {
    title: 'Akıllı Planlama',
    desc: 'Hedefine uygun kişiselleştirilmiş planlar ve öneriler.',
    icon: '🧠',
  },
  {
    title: 'Beslenme Takibi',
    desc: 'Günlük beslenmeni kolayca kaydet ve analiz et.',
    icon: '🥗',
  },
  {
    title: 'İlerleme Analizi',
    desc: 'Grafiklerle gelişimini takip et, motivasyonunu artır.',
    icon: '📈',
  },
  {
    title: 'Topluluk',
    desc: 'Başarı hikâyelerini paylaş, destek al.',
    icon: '🤝',
  },
];

export default function FeatureCards() {
  return (
    <section className="py-12 bg-slate-50 dark:bg-slate-950">
      <div className="max-w-5xl mx-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 px-4">
        {features.map((f, i) => (
          <motion.div
            key={f.title}
            className="rounded-2xl bg-white dark:bg-slate-900 shadow-sm p-6 flex flex-col items-center text-center cursor-pointer border border-slate-200/70 dark:border-slate-800/70 hover:shadow-md transition-all"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            whileHover={{ y: -4, boxShadow: '0 10px 30px 0 rgba(15, 23, 42, 0.08)' }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            viewport={{ once: true }}
          >
            <span className="text-4xl mb-3">{f.icon}</span>
            <h3 className="font-semibold text-lg mb-2 text-slate-900 dark:text-slate-100">
              {f.title}
            </h3>
            <p className="text-slate-600 dark:text-slate-300 text-sm">{f.desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
