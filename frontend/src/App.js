import React, { useState, useEffect } from 'react';
import './App.css';
import SearchChat from './components/SearchChat';
import NaverMap from './components/NaverMap';

// 백엔드 서버 URL 설정
const BACKEND_URL = 'http://localhost:5000';

function App() {
  const [searchResult, setSearchResult] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('unknown');
  // F&F 신사옥 좌표로 초기 위치 설정
  const [userLocation, setUserLocation] = useState({
    lat: 37.4982314,
    lng: 127.0379665
  });

  // 백엔드 서버 상태 확인
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        console.log('백엔드 서버 상태 확인 중...');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch(`${BACKEND_URL}/health`, { 
          signal: controller.signal 
        });

        clearTimeout(timeoutId);

        if (response.ok) {
          setBackendStatus('online');
          console.log('백엔드 서버 연결 성공');
        } else {
          setBackendStatus('error');
          console.error('백엔드 서버 응답 오류:', await response.text());
        }
      } catch (error) {
        console.error('백엔드 서버 연결 실패:', error);
        if (error.name === 'AbortError') {
          console.error('백엔드 서버 연결 시간 초과');
          setBackendStatus('timeout');
        } else {
          setBackendStatus('offline');
        }
      }
    };

    checkBackendStatus();
    const intervalId = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(intervalId);
  }, []);

  // 백엔드 서버 상태에 따른 UI 처리
  useEffect(() => {
    if (backendStatus === 'timeout' || backendStatus === 'error') {
      // 상태 메시지를 UI의 상태 표시줄에만 표시하고, 채팅 메시지로는 추가하지 않음
      console.error('서버 상태 오류:', backendStatus);
    }
  }, [backendStatus]);

  // 사용자 위치 정보 처리 - 실제 위치 대신 F&F 신사옥 위치 사용
  useEffect(() => {
    setUserLocation({
      lat: 37.4982517,
      lng: 127.0310195
    });
  }, []);

  // 위치 기반 검색 함수
  const handleSearch = async (query) => {
    setIsLoading(true);
    setMessages(prev => [...prev, { type: 'user', content: query }]);

    try {
      const response = await fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, location: userLocation }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = '검색 중 오류가 발생했습니다.';
        
        if (response.status === 404) {
          errorMessage = '검색 결과를 찾을 수 없습니다.';
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
        
        setMessages(prev => [...prev, { type: 'error', content: errorMessage }]);
        setIsLoading(false);
        return;
      }

      const data = await response.json();
      setSearchResult(data);
      
      if (data.response) {
        setMessages(prev => [...prev, { type: 'bot', content: data.response }]);
      } else if (data.places && data.places.length > 0) {
        const resultMessage = data.places.map(place => 
          `${place.name}\n주소: ${place.address}\n${place.rating ? `평점: ${place.rating}` : ''}`
        ).join('\n\n');
        setMessages(prev => [...prev, { type: 'bot', content: resultMessage }]);
      } else {
        setMessages(prev => [...prev, { type: 'bot', content: '검색 결과가 없습니다.' }]);
      }
    } catch (error) {
      console.error('Search error:', error);
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: '서버와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // 백엔드 상태에 따른 메시지 렌더링
  const getBackendStatusMessage = () => {
    switch(backendStatus) {
      case 'offline':
        return '백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.';
      case 'timeout':
        return '백엔드 서버 응답 시간이 초과되었습니다. 서버가 과부하 상태일 수 있습니다.';
      case 'error':
        return '백엔드 서버 연결 오류가 발생했습니다. 서버 로그를 확인하세요.';
      default:
        return '백엔드 서버 연결 상태를 확인할 수 없습니다.';
    }
  };

  return (
    <div className="app">
      {(backendStatus !== 'online' && backendStatus !== 'unknown') && (
        <div className="backend-status-warning">
          <p>{getBackendStatusMessage()}</p>
        </div>
      )}
      {userLocation && (
        <div className="location-status">
          <p>현재 위치 정보 사용 중</p>
        </div>
      )}
      <div className="app-container">
        <div className="map-container">
          <NaverMap 
            searchResult={searchResult} 
            userLocation={userLocation}
          />
        </div>
        <div className="chat-container">
          <SearchChat 
            onSearch={handleSearch}
            messages={messages} 
            isLoading={isLoading}
            isBackendOnline={backendStatus === 'online'}
            hasLocation={!!userLocation}
          />
        </div>
      </div>
    </div>
  );
}

export default App; 