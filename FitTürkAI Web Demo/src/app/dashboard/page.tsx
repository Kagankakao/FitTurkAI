'use client';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings, Bell, Eye, Moon, Sun } from 'react-feather';

export default function DashboardPage() {
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('darkMode') === 'true';
    }
    return false;
  });

  const [settings, setSettings] = useState({
    emailNotifications: true,
    pushNotifications: true,
    profileVisibility: true
  });

  const handleSettingChange = (setting: string) => {
    setSettings(prev => ({
      ...prev,
      [setting]: !prev[setting as keyof typeof prev]
    }));
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <main className="md:ml-64 p-6">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-semibold text-slate-900 dark:text-slate-100 mb-8">
            Ayarlar
          </h1>

          <div className="space-y-6">
            {/* Tema Ayarları */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <Moon className="w-6 h-6 text-brand-green" />
                <h2 className="text-xl font-semibold">Tema Ayarları</h2>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-700 dark:text-slate-300">Karanlık Mod</p>
                  <p className="text-sm text-slate-500 dark:text-slate-400">
                    Uygulamanın görünümünü karanlık temaya geçirir
                  </p>
                </div>
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 ${
                    darkMode ? 'bg-emerald-600' : 'bg-slate-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      darkMode ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>

            {/* Bildirim Ayarları */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <Bell className="w-6 h-6 text-brand-dark" />
                <h2 className="text-xl font-semibold">Bildirim Ayarları</h2>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-700 dark:text-slate-300">E-posta Bildirimleri</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Önemli güncellemeler için e-posta al
                    </p>
                  </div>
                  <button
                    onClick={() => handleSettingChange('emailNotifications')}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 ${
                    settings.emailNotifications ? 'bg-emerald-600' : 'bg-slate-200'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        settings.emailNotifications ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-700 dark:text-slate-300">Push Bildirimleri</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Anlık bildirimler al
                    </p>
                  </div>
                  <button
                    onClick={() => handleSettingChange('pushNotifications')}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 ${
                    settings.pushNotifications ? 'bg-emerald-600' : 'bg-slate-200'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        settings.pushNotifications ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>

            {/* Gizlilik Ayarları */}
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <Eye className="w-6 h-6 text-brand-green" />
                <h2 className="text-xl font-semibold">Gizlilik Ayarları</h2>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-700 dark:text-slate-300">Profil Görünürlüğü</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Profilinizi diğer kullanıcılara göster
                    </p>
                  </div>
                  <button
                    onClick={() => handleSettingChange('profileVisibility')}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 ${
                    settings.profileVisibility ? 'bg-emerald-600' : 'bg-slate-200'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        settings.profileVisibility ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
