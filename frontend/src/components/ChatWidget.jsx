import React, { useState } from 'react';
import { Send, MessageSquare } from 'lucide-react';
import clsx from 'clsx';

export default function ChatWidget({ onSendMessage, isLoading }) {
  const [query, setQuery] = useState('');
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSendMessage(query);
      setQuery('');
    }
  };

  const suggestions = [
    "I need a formal outfit for an interview",
    "Summer vacation in Goa",
    "Casual friday at a tech startup",
    "Wedding reception guest"
  ];

  return (
    <div className="w-full bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden shadow-2xl">
      <div className="p-4 border-b border-zinc-800 bg-zinc-900/50 flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
          <MessageSquare className="w-4 h-4 text-purple-400" />
        </div>
        <div>
          <h3 className="text-sm font-medium text-zinc-100">Dare Stylist</h3>
          <p className="text-xs text-zinc-500">AI Fashion Assistant</p>
        </div>
      </div>
      
      <div className="p-4 space-y-3">
        <p className="text-sm text-zinc-400">Try asking:</p>
        <div className="flex flex-wrap gap-2">
          {suggestions.map((s, i) => (
            <button
              key={i}
              onClick={() => onSendMessage(s)}
              disabled={isLoading}
              className="text-xs px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-full transition-colors border border-zinc-700 hover:border-zinc-500"
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="p-4 bg-zinc-950 border-t border-zinc-800">
        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Tell me what you're dressing for..."
            className="w-full bg-zinc-900 border border-zinc-700 rounded-full py-3 pl-4 pr-12 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className={clsx(
              "absolute right-2 p-2 rounded-full transition-colors",
              query.trim() && !isLoading ? "bg-purple-600 text-white hover:bg-purple-500" : "bg-zinc-800 text-zinc-500"
            )}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
}
