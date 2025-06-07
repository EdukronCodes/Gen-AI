document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const userInput = document.getElementById('user-input');
    const message = userInput.value;
    userInput.value = '';

    // Display user message
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML += `<p><strong>You:</strong> ${message}</p>`;

    // Send message to server
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
    .then(response => response.json())
    .then(data => {
        // Display bot response
        chatMessages.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}); 