import React from 'react';

interface RecipeCardProps {
  title: string;
  image?: string;
  description?: string;
  category?: string;
  onClick?: () => void;
}

const RecipeCard: React.FC<RecipeCardProps> = ({
  title,
  image,
  description,
  category,
  onClick,
}) => {
  return (
    <div
      className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/70 dark:border-slate-800/70 shadow-sm overflow-hidden hover:shadow-md transition-all cursor-pointer group"
      onClick={onClick}
    >
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-bold text-gray-900 dark:text-white truncate">{title}</h3>
          {category && (
            <span className="px-2 py-1 bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200 rounded-full text-xs font-semibold">
              {category}
            </span>
          )}
        </div>
        {description && (
          <p className="text-slate-600 dark:text-slate-300 text-sm line-clamp-2">{description}</p>
        )}
      </div>
    </div>
  );
};

export default RecipeCard;
