import React, { useState, useEffect } from 'react';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState([]);

  // 네이버 지도 초기화
  useEffect(() => {
    if (window.naver) {
      const map = new window.naver.maps.Map('map', {
        center: new window.naver.maps.LatLng(37.5665, 126.978),  // 서울 시청 좌표
        zoom: 10,
      });

      // 마커 추가
      new window.naver.maps.Marker({
        position: new window.naver.maps.LatLng(37.5665, 126.978),
        map: map,
      });
    }
  }, []);

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

      {/* 네이버 지도 컴포넌트 */}
      <div id="map" style={{ width: '100%', height: '400px' }}></div>
    </div>
  );
}

export default App;
