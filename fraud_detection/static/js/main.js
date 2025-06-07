document.getElementById('detect-button').addEventListener('click', function() {
    // Send request to server
    fetch('/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
    })
    .then(response => response.json())
    .then(data => {
        // Display detection result
        document.getElementById('detection-result').innerText = data.result;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}); 