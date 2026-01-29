'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Eye, EyeOff } from 'react-feather';
import Link from 'next/link';
import Logo from '@/components/Logo';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const userEmail = localStorage.getItem('userEmail');
      if (userEmail) {
        router.replace('/dashboard');
      } else {
        setChecked(true);
      }
    }
  }, [router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      // Basit localStorage tabanlı auth
      if (!email.trim()) {
        throw new Error('E-posta adresi boş olamaz');
      }
      
      if (!password.trim()) {
        throw new Error('Şifre boş olamaz');
      }

      if (password.length < 6) {
        throw new Error('Şifre en az 6 karakter olmalıdır');
      }

      // Demo kullanıcı kontrolü
      const demoEmail = 'fitturkai@demo.com';
      const demoPassword = '123456';

      if (email === demoEmail && password === demoPassword) {
        // Demo kullanıcı girişi
        const token = 'demo-token-' + Date.now();
        document.cookie = `token=${token}; path=/; max-age=2592000`;
        localStorage.setItem('userEmail', email);
        localStorage.setItem('userName', 'Demo Kullanıcı');
        router.push('/chat');
        return;
      }

      // Normal giriş - sadece e-posta kaydet
      const token = 'user-token-' + Date.now();
      document.cookie = `token=${token}; path=/; max-age=2592000`;
      localStorage.setItem('userEmail', email);
      localStorage.setItem('userName', email.split('@')[0]);
      
      router.push('/chat');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Bir hata oluştu');
    } finally {
      setIsLoading(false);
    }
  };

  if (!checked) return null;

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-950 p-4 font-inter">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="text-center mb-8">
          <Link href="/" className="inline-block">
            <Logo />
          </Link>
          <h1 className="text-3xl font-semibold mt-4 text-slate-900 dark:text-slate-100">
            Giriş Yap
          </h1>
          <p className="text-slate-600 dark:text-slate-300 mt-2">
            FitTürkAI sağlık asistanınıza erişin
          </p>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-sm border border-slate-200/70 dark:border-slate-800/70 p-8">
          {error && (
            <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-xl border border-red-200 dark:border-red-800">
              {error}
            </div>
          )}

          <div className="mb-6 p-4 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-200 rounded-xl border border-slate-200/70 dark:border-slate-700/70">
            <p className="text-sm">
              <strong>Demo Giriş:</strong><br />
              E-posta: fitturkai@demo.com<br />
              Şifre: 123456
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                E-posta
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 rounded-xl border border-slate-300/70 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-slate-400/20 focus:border-slate-400 transition-all duration-200 placeholder-slate-500 dark:placeholder-slate-400"
                placeholder="ornek@email.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Şifre
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full px-4 py-3 rounded-xl border border-slate-300/70 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-slate-400/20 focus:border-slate-400 transition-all duration-200 pr-12 placeholder-slate-500 dark:placeholder-slate-400"
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="remember"
                  type="checkbox"
                  className="h-4 w-4 text-slate-900 focus:ring-slate-400 border-slate-300 dark:border-slate-600 rounded"
                />
                <label htmlFor="remember" className="ml-2 block text-sm text-slate-700 dark:text-slate-300">
                  Beni hatırla
                </label>
              </div>
              <Link
                href="/auth/forgot-password"
                className="text-sm text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 transition-colors"
              >
                Şifremi unuttum
              </Link>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-slate-900 hover:bg-slate-800 text-white rounded-xl py-3 font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm hover:shadow-md"
            >
              {isLoading ? 'Giriş yapılıyor...' : 'Giriş Yap'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Hesabınız yok mu?{' '}
              <Link
                href="/auth/register"
                className="text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 font-medium transition-colors"
              >
                Kayıt Ol
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
