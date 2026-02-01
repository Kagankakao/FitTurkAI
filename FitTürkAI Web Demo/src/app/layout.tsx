import type { Metadata } from 'next';
import './globals.css';
import MotionLayout from '@/components/MotionLayout';
import ClientLayout from './ClientLayout';

export const metadata: Metadata = {
  title: 'FitTurkAI - Sağlıklı Yaşam Takip Uygulaması',
  description:
    'Sağlıklı yaşam ve fitness hedeflerinize ulaşmanıza yardımcı olan kişisel sağlık takip uygulaması.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr">
      <body className="font-sans bg-brand-light text-brand-dark dark:bg-slate-950 dark:text-slate-100">
        <div className="min-h-screen flex flex-col">
          <ClientLayout>
            <MotionLayout>{children}</MotionLayout>
          </ClientLayout>
        </div>
      </body>
    </html>
  );
}
