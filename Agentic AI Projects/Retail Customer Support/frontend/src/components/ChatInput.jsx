import { useState } from 'react';

const QUICK_PROMPTS = [
  'Where is order ORD-10002?',
  'I want a refund for ORD-10001',
  'Recommend electronics',
  'What is your return policy?',
];

export default function ChatInput({ onSend, disabled }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || disabled) return;
    onSend(input.trim());
    setInput('');
  };

  return (
    <div className="chat-input-area">
      <div className="quick-prompts">
        {QUICK_PROMPTS.map((p) => (
          <button key={p} type="button" className="quick-btn" onClick={() => onSend(p)} disabled={disabled}>
            {p}
          </button>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about orders, refunds, products..."
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
