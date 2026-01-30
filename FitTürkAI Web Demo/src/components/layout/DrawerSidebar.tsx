'use client';

import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Settings } from 'react-feather';
import {
  BookOpenIcon,
  FlagIcon,
  ChartBarIcon,
  FireIcon,
  UserIcon,
  HomeIcon,
} from '@heroicons/react/24/outline';
import { useRouter } from 'next/navigation';

const links = [
  { name: 'Profil', href: '/profile', icon: UserIcon, color: 'text-brand-dark' },
  { name: 'Notlar', href: '/notes', icon: BookOpenIcon, color: 'text-brand-green' },
  { name: 'Hedefler', href: '/goals', icon: FlagIcon, color: 'text-brand-dark' },
  { name: 'İlerleme', href: '/progress', icon: ChartBarIcon, color: 'text-brand-green' },
  { name: 'Tarifler', href: '/recipes', icon: FireIcon, color: 'text-brand-dark' },
  { name: 'Ayarlar', href: '/dashboard', icon: Settings, color: 'text-brand-dark' },
];

export default function DrawerSidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const handleLogout = () => {
    document.cookie = 'token=; Max-Age=0; path=/';
    localStorage.removeItem('userEmail');
    onClose();
    router.push('/');
  };
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ x: '-100%' }}
          animate={{ x: 0 }}
          exit={{ x: '-100%' }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="fixed inset-0 z-50 flex"
        >
          {/* Overlay */}
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
          {/* Drawer */}
          <aside className="relative w-72 max-w-full h-full bg-white dark:bg-slate-950 shadow-xl flex flex-col p-6 border-r border-slate-200/60 dark:border-slate-800/60">
            <button
              className="absolute top-4 right-4 p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
              onClick={onClose}
              aria-label="Kapat"
            >
              <X size={24} />
            </button>
            <div className="mb-8 mt-2 text-center">
              <span className="font-semibold text-xl text-slate-900 dark:text-slate-100 select-none tracking-tight">
                Menü
              </span>
            </div>
            <nav className="flex flex-col gap-2 mt-6">
              {links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all duration-200 hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 text-base"
                  onClick={onClose}
                >
                  <link.icon className={`w-6 h-6 ${link.color}`} />
                  {link.name}
                </Link>
              ))}
            </nav>
            <button
              onClick={handleLogout}
              className="mt-8 w-full py-3 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold transition"
            >
              Çıkış Yap
            </button>
          </aside>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
