import React, { useState } from 'react';
import ProductCard from './ProductCard';
import { Sparkles, Info } from 'lucide-react';

export default function OutfitBoard({ outfit, onSwapItem, isSwapping }) {
  if (!outfit) return null;

  const score = outfit.score?.percentage || 0;
  
  return (
    <div className="flex flex-col space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-light text-zinc-50 flex items-center gap-2">
            {outfit.theme}
            <Sparkles className="w-5 h-5 text-purple-400" />
          </h2>
          <p className="text-zinc-400 mt-1">Total: ₹{outfit.total_price}</p>
        </div>
        
        <div className="flex flex-col items-end">
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-400">Compatibility</span>
            <div className="px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full text-green-400 font-medium">
              {score}% Match
            </div>
          </div>
          {outfit.palette && outfit.palette !== 'mixed' && (
            <div className="flex gap-1 mt-2">
              <span className="text-xs text-zinc-500 mr-2">Palette:</span>
              {outfit.palette.split(' / ').map((color, i) => (
                <div key={i} className="px-2 py-0.5 bg-zinc-800 rounded text-xs text-zinc-300">
                  {color}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 auto-rows-max">
        {outfit.items.map((item) => (
          <ProductCard 
            key={item.id} 
            item={item} 
            onSwap={onSwapItem}
            isSwapping={isSwapping === item.id}
          />
        ))}
      </div>

      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5 mt-4">
        <h3 className="text-sm font-medium text-zinc-200 mb-2 flex items-center gap-2">
          <Info className="w-4 h-4 text-purple-400" />
          Why It Works
        </h3>
        <p className="text-zinc-400 text-sm leading-relaxed">
          {outfit.explanation}
        </p>
      </div>
    </div>
  );
}
