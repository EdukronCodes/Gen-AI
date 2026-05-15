const API_BASE = import.meta.env.VITE_API_URL || '';

export async function sendChatMessage(message, customerEmail, sessionId) {
  const res = await fetch(`${API_BASE}/api/v1/customer/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      customer_email: customerEmail || undefined,
      session_id: sessionId || undefined,
    }),
  });
  if (!res.ok) throw new Error('Chat request failed');
  return res.json();
}

export async function trackOrder(orderNumber) {
  const res = await fetch(`${API_BASE}/api/v1/orders/${orderNumber}`);
  if (!res.ok) throw new Error('Order not found');
  return res.json();
}

export async function searchProducts(query) {
  const res = await fetch(`${API_BASE}/api/v1/products/search?q=${encodeURIComponent(query)}`);
  return res.json();
}
