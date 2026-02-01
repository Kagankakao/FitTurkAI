import Image from 'next/image';
import { useState } from 'react';

const Logo = () => {
  const [imgError, setImgError] = useState(false);
  return (
    <div className="flex items-center gap-2 pt-1">
      {!imgError && (
        <Image src="/logo.png" alt="FitTurkAI Logo" width={36} height={36} onError={() => setImgError(true)} />
      )}
      <span className="font-semibold text-2xl text-brand-dark dark:text-slate-100 select-none">
        FitTurkAI
      </span>
    </div>
  );
};

export default Logo; 