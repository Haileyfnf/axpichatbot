import React, { useState } from 'react';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState([]);

  const handleSendMessage = () => {
    if (message === '점심 먹을만한 곳 추천해줘') {
      // Flask에서 반환하던 값 React에서 직접 처리
      setResponse([
        { name: '서브웨이', distance: '도보 2분', type: '샌드위치' },
        { name: '샐러디', distance: '도보 3분', type: '샐러드' },
        { name: '김밥천국', distance: '도보 3분', type: '한식' }
      ]);
    } else {
      setResponse([{ response: `입력한 메시지: ${message}` }]);
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
            {item.name 
              ? `${item.name} - ${item.distance} (${item.type})` 
              : item.response}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
