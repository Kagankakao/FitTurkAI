"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      let userEmail = localStorage.getItem('userEmail');
      let token = document.cookie.split('; ').find(row => row.startsWith('token='))?.split('=')[1];
      if (!userEmail || !token) {
        const demoEmail = 'demo@fitturk.ai';
        token = 'demo-token-' + Date.now();
        document.cookie = `token=${token}; path=/; max-age=2592000`;
        localStorage.setItem('userEmail', demoEmail);
        localStorage.setItem('userName', 'Demo Kullanıcı');
        userEmail = demoEmail;
      }
      if (userEmail && token) setChecked(true);
    }
  }, [router]);

  if (!checked) return null;
  return <>{children}</>;
} 