import React, { useState } from 'react';
import { getImageUrl } from '../utils';
import { RefreshCw } from 'lucide-react';
import clsx from 'clsx';

export default function ProductCard({ item, onSwap, isSwapping }) {
  const [isHovered, setIsHovered] = useState(false);

  if (!item) return null;

  return (
    <div 
      className="relative group rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800 shadow-lg transition-all hover:shadow-purple-500/10 hover:border-zinc-700"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="aspect-[3/4] w-full overflow-hidden bg-zinc-800">
        <img 
          src={getImageUrl(item.id)} 
          alt={item.name} 
          className={clsx(
            "w-full h-full object-cover transition-transform duration-500",
            isHovered && "scale-105"
          )}
        />
      </div>

      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 via-black/60 to-transparent pt-12">
        <p className="text-xs text-purple-400 font-medium uppercase tracking-wider mb-1">
          {item.category_label || item.category}
        </p>
        <h3 className="text-sm text-zinc-100 font-medium line-clamp-1">{item.name}</h3>
        {item.price_inr && (
          <p className="text-sm text-zinc-400 mt-1">₹{item.price_inr}</p>
        )}
      </div>

      {onSwap && (
        <button
          onClick={() => onSwap(item)}
          disabled={isSwapping}
          className={clsx(
            "absolute top-3 right-3 p-2 bg-black/60 backdrop-blur-md rounded-full text-zinc-200 border border-zinc-700 transition-all",
            "hover:bg-purple-600 hover:text-white hover:border-purple-500",
            isHovered ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-2",
            isSwapping && "animate-spin"
          )}
          title="Swap this item"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
