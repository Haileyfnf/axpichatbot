import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState([]);

  const handleSendMessage = async () => {
    try {
      const res = await axios.post('http://127.0.0.1:5000/chatbot', { message });
      setResponse(res.data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h1>F&F 챗봇</h1>
      <input 
        type="text" 
        value={message} 
        onChange={(e) => setMessage(e.target.value)} 
        placeholder="메시지를 입력하세요..." 
      />
      <button onClick={handleSendMessage}>전송</button>
      <ul>
        {response.map((item, index) => (
          <li key={index}>
            {item.name} - {item.distance} ({item.type})
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
