import { motion } from 'framer-motion';

const stories = [
  {
    name: 'Zeynep Y.',
    story:
      'Düzenli olarak uygulamayı kullandım ve 3 ayda hedefime ulaştım. Artık çok daha sağlıklıyım!',
    image: 'https://randomuser.me/api/portraits/women/65.jpg',
  },
  {
    name: 'Ali V.',
    story: 'FitTurkAI ile kas kütlemi artırdım ve motivasyonumu hiç kaybetmedim. Teşekkürler!',
    image: 'https://randomuser.me/api/portraits/men/43.jpg',
  },
];

export default function SuccessStories() {
  return (
    <section className="py-16 bg-slate-50 dark:bg-slate-950">
      <h2 className="text-3xl font-semibold text-center mb-10 text-slate-900 dark:text-slate-100">
        Hedefe Ulaşanlar
      </h2>
      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 px-4">
        {stories.map((s, i) => (
          <motion.div
            key={s.name}
            className="flex flex-col md:flex-row items-center bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm p-6 gap-6 animate-fade-in"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: i * 0.2 }}
            viewport={{ once: true }}
          >
            <img
              src={s.image}
              alt={s.name}
              className="w-24 h-24 rounded-full object-cover border-2 border-slate-200 dark:border-slate-700 mb-4 md:mb-0 shadow-sm"
              onError={(e) =>
                (e.currentTarget.src =
                  'https://ui-avatars.com/api/?name=' + encodeURIComponent(s.name))
              }
            />
            <div>
              <p className="text-lg text-slate-700 dark:text-slate-200 mb-2">{s.story}</p>
              <span className="font-semibold text-slate-900 dark:text-slate-100">
                {s.name}
              </span>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
