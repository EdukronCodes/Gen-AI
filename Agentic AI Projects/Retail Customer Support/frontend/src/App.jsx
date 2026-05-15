import { useState, useRef, useEffect } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { sendChatMessage } from './services/api';
import './styles/chat.css';

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      content: "Hi! I'm your retail support assistant. I can help track orders, process refunds, recommend products, and answer policy questions. How can I help?",
      agent: 'customer_support',
    },
  ]);
  const [sessionId, setSessionId] = useState(null);
  const [email, setEmail] = useState('alice@example.com');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (text) => {
    setMessages((m) => [...m, { role: 'user', content: text }]);
    setLoading(true);
    try {
      const data = await sendChatMessage(text, email, sessionId);
      if (!sessionId) setSessionId(data.session_id);
      setMessages((m) => [
        ...m,
        {
          role: 'bot',
          content: data.reply,
          agent: data.agent,
          sentiment: data.sentiment,
        },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: 'bot', content: 'Sorry, I had trouble connecting. Please ensure the backend is running on port 8000.', agent: 'error' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <span className="logo">🛍️</span>
          <div>
            <h1>Retail Support</h1>
            <p>AI-powered customer assistance</p>
          </div>
        </div>
        <div className="header-user">
          <label>Demo email</label>
          <select value={email} onChange={(e) => setEmail(e.target.value)}>
            <option value="alice@example.com">alice@example.com</option>
            <option value="bob@example.com">bob@example.com</option>
            <option value="carol@example.com">carol@example.com</option>
          </select>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.map((msg, i) => (
            <ChatMessage key={i} {...msg} />
          ))}
          {loading && (
            <div className="typing">
              <span></span><span></span><span></span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <ChatInput onSend={handleSend} disabled={loading} />
      </main>

      <aside className="sidebar">
        <h3>Demo data</h3>
        <ul>
          <li><strong>Orders:</strong> ORD-10001, ORD-10002, ORD-10003</li>
          <li><strong>Refunds:</strong> REF-5001, REF-5002</li>
          <li><strong>Try:</strong> Track ORD-10002, Refund for ORD-10001, Show electronics</li>
        </ul>
      </aside>
    </div>
  );
}
