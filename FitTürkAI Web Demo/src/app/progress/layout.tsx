"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function ProgressLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      let userEmail = localStorage.getItem('userEmail');
      if (!userEmail) {
        const demoEmail = 'demo@fitturk.ai';
        const token = 'demo-token-' + Date.now();
        document.cookie = `token=${token}; path=/; max-age=2592000`;
        localStorage.setItem('userEmail', demoEmail);
        localStorage.setItem('userName', 'Demo Kullanıcı');
        userEmail = demoEmail;
      }
      if (userEmail) setChecked(true);
    }
  }, [router]);

  if (!checked) return null;
  return <>{children}</>;
} 