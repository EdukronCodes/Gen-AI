export default function ChatMessage({ role, content, agent, sentiment }) {
  const isUser = role === 'user';
  return (
    <div className={`message ${isUser ? 'message-user' : 'message-bot'}`}>
      {!isUser && agent && (
        <span className="agent-badge">{agent.replace(/_/g, ' ')}</span>
      )}
      <div className="message-content" dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }} />
      {!isUser && sentiment && (
        <span className={`sentiment sentiment-${sentiment.label}`}>{sentiment.label}</span>
      )}
    </div>
  );
}

function formatMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>');
}
