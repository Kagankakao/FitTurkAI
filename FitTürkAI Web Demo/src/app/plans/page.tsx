export default function PlansPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 dark:bg-slate-950">
      <div className="bg-white/90 dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm p-10 w-full max-w-2xl flex flex-col items-center animate-fade-in">
        <h1 className="text-4xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
          Planlarım
        </h1>
        <p className="mb-6 text-slate-600 dark:text-slate-300 text-center text-lg max-w-xl">
          Kişisel antrenman ve beslenme planlarını burada görüntüleyebilir, yeni planlar
          oluşturabilir veya mevcut planlarını düzenleyebilirsin.
        </p>
        <div className="flex flex-wrap gap-6 justify-center mt-4">
          <div className="bg-slate-900 rounded-xl p-6 shadow-sm text-white w-56 text-center hover:scale-105 transition">
            <span className="text-3xl">🏋️‍♂️</span>
            <div className="font-bold mt-2">Antrenman Planı</div>
            <div className="text-sm mt-1">Haftalık ve günlük antrenmanlarını takip et.</div>
          </div>
          <div className="bg-slate-800 rounded-xl p-6 shadow-sm text-white w-56 text-center hover:scale-105 transition">
            <span className="text-3xl">🥗</span>
            <div className="font-bold mt-2">Beslenme Planı</div>
            <div className="text-sm mt-1">Kişisel diyet ve öğünlerini planla.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
