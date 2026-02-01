/* eslint-disable @next/next/no-img-element */
import { Swiper, SwiperSlide } from 'swiper/react';
import { Pagination, Autoplay } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/pagination';

const testimonials = [
  {
    name: 'Ayşe K.',
    text: 'FitTurkAI sayesinde 6 ayda 12 kilo verdim ve çok daha enerjik hissediyorum! Planlar ve topluluk desteği harika.',
    image: 'https://randomuser.me/api/portraits/women/44.jpg',
  },
  {
    name: 'Mehmet T.',
    text: 'Uygulamanın beslenme takibi ve ilerleme analizleri motivasyonumu hep yüksek tuttu. Herkese tavsiye ederim!',
    image: 'https://randomuser.me/api/portraits/men/32.jpg',
  },
  {
    name: 'Elif S.',
    text: 'Kişisel planlar ve önerilerle hedefime çok daha hızlı ulaştım. Arayüzü de çok kullanışlı.',
    image: 'https://randomuser.me/api/portraits/women/68.jpg',
  },
];

export default function Testimonials() {
  return (
    <section className="py-16 bg-white dark:bg-slate-950">
      <h2 className="text-3xl font-semibold text-center mb-10 text-brand-dark dark:text-slate-100">
        Kullanıcı Başarı Hikâyeleri
      </h2>
      <div className="max-w-2xl mx-auto">
        <Swiper
          modules={[Pagination, Autoplay]}
          spaceBetween={30}
          slidesPerView={1}
          pagination={{ clickable: true }}
          autoplay={{ delay: 4000 }}
          loop
        >
          {testimonials.map((t, i) => (
            <SwiperSlide key={i}>
              <div className="flex flex-col items-center bg-slate-50 dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm p-8 animate-fade-in">
                <img
                  src={t.image}
                  alt={t.name}
                  className="w-20 h-20 rounded-full object-cover mb-4 border-2 border-slate-200 dark:border-slate-700 shadow-sm"
                  onError={(e) =>
                    (e.currentTarget.src =
                      'https://ui-avatars.com/api/?name=' + encodeURIComponent(t.name))
                  }
                />
                <p className="text-lg text-brand-dark/80 dark:text-slate-200 mb-2 text-center">
                  “{t.text}”
                </p>
                <span className="font-semibold text-brand-dark dark:text-slate-100">
                  {t.name}
                </span>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
    </section>
  );
}
