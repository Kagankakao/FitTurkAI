'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const router = useRouter();

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const demoEmail = 'demo@fitturk.ai';
      const token = 'demo-token-' + Date.now();
      document.cookie = `token=${token}; path=/; max-age=2592000`;
      localStorage.setItem('userEmail', demoEmail);
      localStorage.setItem('userName', 'Demo Kullanıcı');
      router.replace('/chat');
    }
  }, [router]);

  return null;
}
