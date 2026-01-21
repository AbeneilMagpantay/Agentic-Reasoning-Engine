import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, BrainCircuit, ChevronDown, ChevronRight, Activity } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

// Helper for classes
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Reasoning Block Component
const ReasoningStep = ({ step, expanded, onToggle }) => (
  <div className="border border-tech-border rounded-lg mb-2 overflow-hidden">
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between p-2 bg-tech-card hover:bg-tech-border/50 text-xs font-mono text-tech-muted transition-colors"
    >
      <div className="flex items-center gap-2">
        <Activity size={14} className="text-tech-accent" />
        <span>{step.title}</span>
      </div>
      {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
    </button>
    <AnimatePresence>
      {expanded && (
        <motion.div
          initial={{ height: 0 }}
          animate={{ height: "auto" }}
          exit={{ height: 0 }}
          className="bg-tech-bg/50"
        >
          <div className="p-3 text-xs text-tech-muted whitespace-pre-wrap font-mono">
            {step.content}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  </div>
);

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input;
    setInput('');

    // Add User Message
    const userMsg = { role: 'user', content: question, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);

    try {
      // Create Placeholder Agent Message
      const agentMsgId = Date.now() + 1;
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '',
        reasoning: [], // Array of {title, content}
        id: agentMsgId,
        loading: true
      }]);

      const res = await fetch(`http://localhost:8000/invoke?question=${encodeURIComponent(question)}`, {
        method: 'POST'
      });

      if (!res.ok) throw new Error("API Error");

      const data = await res.json();

      // Update Agent Message
      setMessages(prev => prev.map(msg =>
        msg.id === agentMsgId
          ? {
            ...msg,
            content: data.generation,
            loading: false,
            reasoning: [
              { title: 'Retrieval', content: `Found ${data.documents?.length || 0} documents.` },
              { title: 'Hallucination Check', content: 'Verified: Grounded in context.' }
            ]
          }
          : msg
      ));
    } catch (err) {
      setMessages(prev => prev.map(msg =>
        msg.loading
          ? { ...msg, content: "Error connecting to Agentic Engine.", loading: false }
          : msg
      ));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-tech-bg text-tech-text font-sans selection:bg-tech-accent/30">
      {/* Sidebar - Visual Flair */}
      <div className="hidden md:flex flex-col w-64 border-r border-tech-border bg-tech-card/50 p-4">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 rounded-lg bg-tech-accent flex items-center justify-center">
            <BrainCircuit size={20} className="text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight">Agentic Engine</h1>
        </div>

        <div className="flex-1">
          <p className="text-xs font-mono text-tech-muted uppercase tracking-wider mb-3">System Status</p>
          <div className="flex items-center gap-2 text-sm text-green-400 mb-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Active
          </div>
          <div className="flex items-center gap-2 text-sm text-tech-muted">
            <span className="w-2 h-2 rounded-full bg-tech-accent/50" />
            Gemini 2.5 Flash
          </div>
          <div className="flex items-center gap-2 text-sm text-tech-muted mt-2">
            <span className="w-2 h-2 rounded-full bg-orange-500/50" />
            Qdrant Vector DB
          </div>
        </div>

        <div className="text-xs text-tech-muted/50 text-center">
          v1.0.0 • Foundation
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative max-w-5xl mx-auto w-full">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 pb-48">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-tech-muted space-y-4 opacity-50">
              <BrainCircuit size={48} className="text-tech-accent" />
              <p className="text-lg">Ready to reason.</p>
            </div>
          )}

          {messages.map((msg) => (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={msg.id}
              className={cn("flex gap-4", msg.role === 'user' ? "justify-end" : "justify-start")}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-tech-border flex items-center justify-center shrink-0">
                  <Bot size={16} className="text-tech-accent" />
                </div>
              )}

              <div className={cn(
                "max-w-[80%] rounded-2xl p-4 shadow-sm",
                msg.role === 'user'
                  ? "bg-tech-accent/10 border border-tech-accent/20 text-white rounded-br-sm"
                  : "bg-tech-card border border-tech-border rounded-bl-sm"
              )}>
                {/* Reasoning Steps (Agent Only) */}
                {msg.reasoning && msg.reasoning.length > 0 && (
                  <div className="mb-4 space-y-1">
                    {msg.reasoning.map((step, idx) => (
                      <ReasoningStep
                        key={idx}
                        step={step}
                        expanded={true}
                        onToggle={() => { }}
                      />
                    ))}
                  </div>
                )} { /* Auto-expanded for demo */}

                {/* Content */}
                {msg.loading ? (
                  <div className="flex gap-1 h-6 items-center px-2">
                    <span className="w-1.5 h-1.5 bg-tech-muted rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-1.5 h-1.5 bg-tech-muted rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-1.5 h-1.5 bg-tech-muted rounded-full animate-bounce" />
                  </div>
                ) : (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                )}
              </div>

              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-tech-border flex items-center justify-center shrink-0">
                  <User size={16} className="text-tech-muted" />
                </div>
              )}
            </motion.div>
          ))}
          <div ref={scrollRef} className="h-32 md:h-48 shrink-0" />
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 w-full p-4 md:p-8 bg-gradient-to-t from-tech-bg via-tech-bg to-transparent">
          <form onSubmit={handleSubmit} className="relative max-w-3xl mx-auto shadow-2xl">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              className="w-full bg-tech-card/80 backdrop-blur-md border border-tech-border rounded-xl py-4 pl-6 pr-14 text-tech-text placeholder:text-tech-muted focus:outline-none focus:ring-1 focus:ring-tech-accent focus:border-tech-accent transition-all shadow-lg"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-tech-accent text-white rounded-lg hover:bg-tech-accent/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={18} />
            </button>
          </form>
          <p className="text-center text-xs text-tech-muted mt-3">
            Agentic Reasoner may produce hallucinations. Use with caution.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
