import React, { useState } from 'react';
import ChatWidget from './components/ChatWidget';
import UploadZone from './components/UploadZone';
import OutfitBoard from './components/OutfitBoard';
import { chatOutfits, uploadImage, swapItem } from './api';
import { Sparkles, ArrowRight } from 'lucide-react';

function App() {
  const [outfits, setOutfits] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentQuery, setCurrentQuery] = useState('');
  const [intent, setIntent] = useState(null);
  const [swappingItemId, setSwappingItemId] = useState(null);
  const [viewIndex, setViewIndex] = useState(0);

  const handleChat = async (query) => {
    setIsLoading(true);
    setCurrentQuery(query);
    try {
      const data = await chatOutfits(query);
      setOutfits(data.outfits || []);
      setIntent(data.intent);
      setViewIndex(0);
    } catch (err) {
      console.error(err);
      alert("Failed to generate outfits.");
    }
    setIsLoading(false);
  };

  const handleUpload = async (file) => {
    setIsLoading(true);
    try {
      const data = await uploadImage(file);
      setOutfits(data.outfits || []);
      setIntent(data.intent);
      setCurrentQuery("Visual Match");
      setViewIndex(0);
    } catch (err) {
      console.error(err);
      alert("Failed to process image.");
    }
    setIsLoading(false);
  };

  const handleSwap = async (itemToSwap) => {
    const currentOutfit = outfits[viewIndex];
    setSwappingItemId(itemToSwap.id);
    
    // figure out slot
    // We pass the slot name but the backend logic is flexible
    try {
      const data = await swapItem(
        currentOutfit.items,
        itemToSwap.category, // using category as slot hint
        intent?.occasion || 'casual',
        intent?.gender,
        currentQuery
      );
      
      // Update outfit
      const newOutfits = [...outfits];
      newOutfits[viewIndex] = {
        ...currentOutfit,
        items: data.items,
        score: data.score,
        explanation: data.explanation,
        total_price: data.total_price
      };
      setOutfits(newOutfits);
    } catch (err) {
      console.error(err);
      alert("Could not swap item.");
    }
    setSwappingItemId(null);
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 selection:bg-purple-500/30 font-sans">
      <header className="fixed top-0 left-0 right-0 h-16 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md z-50 flex items-center px-6">
        <h1 className="text-xl font-semibold tracking-wide flex items-center gap-2">
          Dare <span className="text-purple-400 font-light">XAI</span>
        </h1>
      </header>

      <main className="pt-24 pb-20 px-6 lg:px-12 w-full max-w-[1800px] mx-auto flex flex-col lg:flex-row gap-10 lg:gap-16">
        
        {/* Left Sidebar: Controls */}
        <div className="w-full lg:w-1/3 flex flex-col gap-6">
          <div className="mb-4">
            <h2 className="text-4xl font-light leading-tight mb-4">
              Your Personal <br />
              <span className="font-medium text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                AI Stylist
              </span>
            </h2>
            <p className="text-zinc-400 text-lg">
              Describe your occasion or upload an inspiration piece to instantly generate curated, shoppable outfits.
            </p>
          </div>

          <ChatWidget onSendMessage={handleChat} isLoading={isLoading} />
          
          <div className="flex items-center gap-4 my-2">
            <div className="flex-1 h-px bg-zinc-800"></div>
            <span className="text-zinc-600 text-sm font-medium uppercase">or</span>
            <div className="flex-1 h-px bg-zinc-800"></div>
          </div>

          <UploadZone onUpload={handleUpload} isLoading={isLoading} />
        </div>

        {/* Right Content: Outfit Boards */}
        <div className="w-full lg:w-2/3">
          {outfits.length > 0 ? (
            <div className="flex flex-col h-full">
              {/* Tabs for multiple outfits */}
              {outfits.length > 1 && (
                <div className="flex gap-2 mb-6 p-1 bg-zinc-900 rounded-lg w-fit">
                  {outfits.map((outfit, idx) => (
                    <button
                      key={idx}
                      onClick={() => setViewIndex(idx)}
                      className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                        viewIndex === idx 
                          ? 'bg-zinc-800 text-white shadow-sm' 
                          : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                      }`}
                    >
                      {outfit.theme}
                    </button>
                  ))}
                </div>
              )}

              <OutfitBoard 
                outfit={outfits[viewIndex]} 
                onSwapItem={handleSwap} 
                isSwapping={swappingItemId} 
              />
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-12 border border-zinc-800/50 rounded-2xl bg-zinc-900/20 border-dashed">
              <Sparkles className="w-12 h-12 text-zinc-700 mb-4" />
              <h3 className="text-xl text-zinc-300 font-medium mb-2">Awaiting your styling prompt</h3>
              <p className="text-zinc-500 max-w-sm">
                Use the chat or upload an image on the left to start generating curated fashion boards.
              </p>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}

export default App;
